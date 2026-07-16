"""Reference example: write your own LLM-driven optimizer (a custom BaseAgenticOptimizer).

This is a template for teams adding a NEW *LLM-driven* optimization algorithm — the
"Path B" extension point. Instead of hand-coding an update rule, you inherit a Bedrock
reflection agent that reads the rollout traces and rewrites the Formula's params itself.

``BaseAgenticOptimizer`` gives you, for free, all the reusable helpers:
  - a Bedrock agent with the `shell` tool and a `submit_optimized_params` tool
    (``_create_agent``), plus a ``_get_extra_tools()`` hook to add your own
  - stratified trace sampling (``_sample_traces``), temp-folder trace writing
    (``_write_traces_to_temp`` / ``_cleanup_temp``), an output guardrail, checkpointing

You supply ``step()`` — composing those helpers into your algorithm. The canonical flow
(sample → write traces → reflect → read submitted params → update formula → cleanup) is
shown below and is worth copying verbatim; the interesting variation is what you override
around it. Here we override ``_get_extra_tools()`` to give the reflection agent an
**LLM-as-judge** tool: before submitting, the agent can call ``judge_instructions`` to
score a candidate rewrite (0–1, with a critique) and iterate. This is the same
``_get_extra_tools()`` hook ``MultiAgentOptimizer`` uses to add ``swarm`` — a good pattern
for any tool-augmented reflection (a scorer/judge, retrieval, code exec, …).

The reflection prompt templates and submit protocol are reused from the built-in
contrastive_reflection setup, which is param-agnostic and therefore works unchanged on a
``SkillFormula`` (it optimizes the skill's ``instructions``).

After training, the optimized skill is written back to ``optimized_skills/<name>/SKILL.md``
so the update persists (training only mutates the in-memory Skill); it reloads via
``Skill.from_file(...)``.

Compare with:
- skill_optimization.py — same task, but with the stock ContrastiveReflectionOptimizer
- gsm8k_trainer.py       — the standard Trainer loop with a SystemPromptFormula

If your algorithm is programmatic (no LLM), subclass ``FormulaOptimizer`` directly and
implement ``step()`` instead — see the "Path A" notes in the extending guide.

Requirements:
    pip install strands-harness-optimizer datasets
    A Strands version that ships the AgentSkills plugin (from strands import AgentSkills, Skill)
    AWS credentials configured for Bedrock access
"""

import os
import re

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "This example requires the 'datasets' package. "
        "Install it with: pip install datasets"
    )

from strands import Agent, AgentSkills, Skill, tool
from strands.models import BedrockModel
from strands_tools import shell

from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SkillFormula
from strands_harness_optimizer.adapters import apply_formulas_on_strands_agent
from strands_harness_optimizer.optimizers import BaseAgenticOptimizer
from strands_harness_optimizer.rewards import RewardFunction
from strands_harness_optimizer.rollout_engines import LocalRolloutEngine
from strands_harness_optimizer.trainer import Trainer
from strands_harness_optimizer.utils import load_builtin_template


# ---------------------------------------------------------------------------
# THE PART TO STUDY: a custom LLM-driven optimizer.
# ---------------------------------------------------------------------------

class LLMJReferenceSkillOptimizer(BaseAgenticOptimizer):
    """Reference skill optimizer: a BaseAgenticOptimizer whose reflection agent has an
    LLM-as-judge (LLMJ) tool.

    ``step()`` composes the inherited helpers into the canonical reflection flow
    (sample traces → write to a temp folder → run the Bedrock agent → read the params
    it submits via ``submit_optimized_params`` → ``formula.update_params(...)`` →
    cleanup in a ``finally``). The two customizations that make this a *new* optimizer:

    1. ``_get_extra_tools()`` — add a ``judge_instructions`` tool: a lightweight
       LLM-as-judge the reflection agent can call to score a candidate rewrite (0–1,
       with a critique) before submitting, so it can iterate toward a higher-scoring
       version. This is the same hook ``MultiAgentOptimizer`` uses to add ``swarm``.
    2. ``_run_reflection()`` — render the templates and run the (judge-equipped) agent.

    Everything else — the submit protocol, guardrail, sampling, checkpointing — is
    inherited unchanged.

    Args:
        formula: The Formula whose params will be optimized (any Formula works).
        system_prompt_template / task_message_template: Jinja templates for the
            reflection agent. The built-in param-agnostic contrastive_reflection
            templates work as-is; pass your own to change the reflection strategy.
        **kwargs: Forwarded to BaseAgenticOptimizer (model_config, region_name,
            n_sample_traces, stratified_sampling, success_threshold, ...).
    """

    def __init__(self, formula, system_prompt_template, task_message_template, **kwargs):
        super().__init__(formula, **kwargs)
        # create_template accepts str or Template; store rendered-ready templates.
        from strands_harness_optimizer.utils.templates import create_template

        self._system_prompt_template = (
            create_template(system_prompt_template)
            if isinstance(system_prompt_template, str)
            else system_prompt_template
        )
        self._task_message_template = (
            create_template(task_message_template)
            if isinstance(task_message_template, str)
            else task_message_template
        )

    def step(self) -> None:
        """The reflection loop, composed from inherited helpers.

        This is the canonical BaseAgenticOptimizer flow — copy it verbatim and
        customize the hooks (``_run_reflection`` / ``_get_extra_tools``) around it.
        """
        if not self._rollouts:  # always handle the empty buffer
            return
        try:
            indices = self._sample_traces()               # stratified by reward
            traces_folder = self._write_traces_to_temp(indices)
            params = self.formula.get_tunable_params()     # what to optimize
            self._run_reflection(traces_folder, params)    # run the reflection agent
            optimized = self._get_submitted_params()       # what it submitted
            if optimized:
                self.formula.update_params(optimized)
            else:
                raise RuntimeError(
                    "Reflection agent did not call submit_optimized_params."
                )
        finally:
            self._cleanup_temp()                           # never leak temp dirs

    def _get_extra_tools(self) -> list:
        """Give the reflection agent an LLM-as-judge tool (on top of shell + submit).

        The tool spins up a small Bedrock judge agent that scores a candidate
        instruction rewrite. It reuses this optimizer's ``model_config`` /
        ``region_name`` so no extra configuration is needed. Returning it here is all
        it takes for the reflection agent to be able to call it.
        """
        model_config = self.model_config
        region_name = self.region_name

        @tool
        def judge_instructions(candidate: str) -> str:
            """Score a candidate skill-instruction rewrite for quality.

            Args:
                candidate: The proposed instruction text to evaluate.

            Returns:
                A line like "score=0.8 | <one-sentence critique>".
            """
            judge = Agent(
                model=BedrockModel(region_name=region_name, **model_config),
                system_prompt=(
                    "You are a strict rubric-based judge of agent *instructions* for a "
                    "math word-problem solver. Reply with ONLY one line: "
                    "'score=<0..1> | <one-sentence critique>'. Reward instructions that "
                    "are specific and actionable (clear final-answer formatting, "
                    "step-by-step guidance) and penalize vague or contradictory ones."
                ),
                callback_handler=None,
            )
            return extract_response_text(judge(f"Instructions to score:\n{candidate}"))

        return [judge_instructions]

    def _run_reflection(self, traces_folder: str, params: dict) -> None:
        """Render the templates and run the (judge-equipped) reflection agent.

        Mirrors ContrastiveReflectionOptimizer._run_reflection — the base ``step()``
        calls this after writing traces and before reading submitted params.
        """
        import os

        template_vars = {"traces_folder": os.path.abspath(traces_folder), "params": params}
        system_prompt = self._system_prompt_template.render(**template_vars)
        task_message = self._task_message_template.render(**template_vars)

        agent = self._create_agent(system_prompt)  # gets shell + submit + judge_instructions
        agent(task_message)


# ---------------------------------------------------------------------------
# Standard harness (task, reward, agent) — see skill_optimization.py for detail.
# ---------------------------------------------------------------------------

class GSM8KReward(RewardFunction):
    """Reward 1.0 if the agent's final number matches the expected answer."""

    def __call__(self, rollout: Rollout) -> Reward:
        response_text = rollout.metadata.get("response_text", "")
        expected = rollout.data_sample.get("expected_answer", "")
        predicted = self._extract_final_number(response_text)
        return Reward(
            reward=1.0 if predicted is not None and predicted == expected else 0.0,
            metadata={"predicted": predicted, "expected": expected},
        )

    def _extract_final_number(self, text: str) -> str | None:
        numbers = re.findall(r'-?\d[\d,]*\.?\d*', text.replace(",", ""))
        if not numbers:
            return None
        last = numbers[-1].rstrip(".")
        return last.rstrip("0").rstrip(".") if "." in last else last


def parse_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', answer_text)
    return match.group(1).replace(",", "") if match else ""


def prepare_gsm8k(split: str):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {"question": row["question"], "expected_answer": parse_gsm8k_answer(row["answer"])}
        for row in ds
    ]


def extract_response_text(response) -> str:
    try:
        return response.message["content"][0]["text"]
    except Exception:
        return str(response)


def save_skill(skill: Skill, directory: str) -> str:
    """Persist a Skill to ``<directory>/SKILL.md`` (frontmatter + instructions).

    Training only mutates the in-memory Skill; ``Skill`` has loaders
    (``from_file``/``from_content``) but no serializer, so we write the canonical
    SKILL.md ourselves. The result reloads via ``Skill.from_file(directory)``.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "SKILL.md")
    content = (
        "---\n"
        f"name: {skill.name}\n"
        f"description: {skill.description}\n"
        "---\n\n"
        f"{skill.instructions}\n"
    )
    with open(path, "w") as f:
        f.write(content)
    return path


def main():
    print("Loading GSM8K dataset...")
    train_data = prepare_gsm8k("test[:10]")
    test_data = prepare_gsm8k("test[10:15]")
    train_loader = DataLoader(train_data, batch_size=10)
    print(f"Train: {len(train_data)} problems, Test: {len(test_data)} problems")

    # The Skill we optimize (its instructions are the tunable param).
    math_skill = Skill(
        name="math-solver",
        description="Solve arithmetic word problems and report a single final number.",
        instructions="Solve the math word problem and end with the final numeric answer.",
    )
    formula = SkillFormula(math_skill)

    model = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
    )
    skills_plugin = AgentSkills(skills=[math_skill])

    def create_agent():
        agent = Agent(
            model=model,
            system_prompt=(
                "You are a math assistant. Activate the `math-solver` skill and follow "
                "its instructions. You may use the shell tool as a calculator."
            ),
            plugins=[skills_plugin],
            tools=[shell],
            callback_handler=None,
        )
        apply_formulas_on_strands_agent(agent, [formula])
        return agent

    def invoke_agent(agent, data_sample) -> Rollout:
        agent.messages.clear()
        try:
            response_text = extract_response_text(agent(data_sample["question"]))
        except Exception as e:
            response_text = f"Error: {e}"
        return Rollout(
            data_sample=data_sample,
            messages=list(agent.messages),
            metadata={"response_text": response_text},
        )

    engine = LocalRolloutEngine(
        formula=formula, agent_creator=create_agent, agent_invoker=invoke_agent
    )

    # Our custom optimizer, plugged in exactly like any built-in one. It reuses the
    # param-agnostic contrastive_reflection templates (they optimize whatever params
    # the formula exposes — here the skill's "instructions").
    optimizer = LLMJReferenceSkillOptimizer(
        formula,
        system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
        task_message_template=load_builtin_template(
            "contrastive_reflection/task_message_system_prompt.jinja"
        ),
        model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        n_sample_traces=-1,
    )

    trainer = Trainer(
        formula=formula,
        optimizer=optimizer,
        reward_fn=GSM8KReward(),
        engine=engine,
        dataloader=train_loader,
        n_epochs=2,
    )

    print("\n=== Training the skill with LLMJReferenceSkillOptimizer ===")
    print(f"Initial instructions: {formula.get_tunable_params()['instructions'][:100]}...")
    stats = trainer.fit()
    for s in stats:
        print(f"  Epoch {s['epoch']}: avg_reward = {s['avg_reward']:.2f}")

    # Confirm the optimization on a held-out split: the same agent now runs with the
    # skill's optimized instructions (the formula re-syncs them into the plugin).
    print("\n=== Held-out evaluation (optimized skill) ===")
    reward_fn = GSM8KReward()
    eval_agent = create_agent()
    correct = 0
    for i, sample in enumerate(test_data):
        reward = reward_fn(invoke_agent(eval_agent, sample))
        correct += reward.reward
        status = "correct" if reward.reward == 1.0 else "wrong"
        print(f"  [{i + 1}/{len(test_data)}] {status} "
              f"(predicted={reward.metadata['predicted']}, expected={reward.metadata['expected']})")
    print(f"Test accuracy: {correct / len(test_data):.0%}")

    print(f"\nFinal optimized skill instructions:\n{formula.get_tunable_params()['instructions']}")

    # Persist the optimized skill so the update outlives this process (training only
    # mutated the in-memory Skill). It reloads with Skill.from_file(<dir>).
    out_dir = os.path.join("optimized_skills", math_skill.name)
    path = save_skill(math_skill, out_dir)
    reloaded = Skill.from_file(out_dir)
    assert reloaded.instructions == math_skill.instructions  # round-trip check
    print(f"\nSaved optimized skill to {path} (reload via Skill.from_file('{out_dir}'))")


if __name__ == "__main__":
    main()
