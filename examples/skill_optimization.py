"""Example: Optimize a Strands Agent *Skill* on GSM8K.

Strands Agent Skills (https://strandsagents.com/docs/user-guide/concepts/plugins/skills/)
are modular instruction packages loaded via the ``AgentSkills`` plugin using
progressive disclosure: a skill's ``name`` + ``description`` are injected into the
system prompt, and its full ``instructions`` load on demand when the agent activates
the skill via the ``skills`` tool.

Both the ``description`` (the discovery hook) and the ``instructions`` (the payload)
are optimizable context. This example uses the built-in ``SkillFormula`` to make a
``Skill``'s text a *tunable parameter*, then drives the automated ``Trainer`` loop
(``LocalRolloutEngine`` + ``ContrastiveReflectionOptimizer``) to rewrite the skill's
instructions from success/failure patterns on GSM8K.

The Formula is the single interface the optimizer talks to, and the control flow is
explicit: optimizer -> Formula.update_params() (mutates the Skill) -> the adapter
calls Formula.process() each invocation and pushes the returned skill into the
AgentSkills plugin via set_available_skills(). The formula never touches the agent's
system prompt, and correctness does not depend on the plugin and formula sharing the
same Skill object.

Compare with:
- gsm8k_optimization.py — optimizes a SystemPromptFormula (manual loop)
- gsm8k_trainer.py       — optimizes a SystemPromptFormula (Trainer loop)

Requirements:
    pip install strands-harness-optimizer datasets
    A Strands version that ships the AgentSkills plugin (from strands import AgentSkills, Skill)
    AWS credentials configured for Bedrock access
"""

import re

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "This example requires the 'datasets' package. "
        "Install it with: pip install datasets"
    )

from strands import Agent, AgentSkills, Skill
from strands.models import BedrockModel
from strands_tools import shell

from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SkillFormula
from strands_harness_optimizer.adapters import apply_formulas_on_strands_agent
from strands_harness_optimizer.rewards import RewardFunction
from strands_harness_optimizer.optimizers import ContrastiveReflectionOptimizer
from strands_harness_optimizer.rollout_engines import LocalRolloutEngine
from strands_harness_optimizer.trainer import Trainer
from strands_harness_optimizer.utils import load_builtin_template


# --- Reward function: extract final number and compare ---

class GSM8KReward(RewardFunction):
    """Reward 1.0 if the agent's final number matches the expected answer."""

    def __call__(self, rollout: Rollout) -> Reward:
        response_text = rollout.metadata.get("response_text", "")
        expected = rollout.data_sample.get("expected_answer", "")

        predicted = self._extract_final_number(response_text)
        correct = predicted is not None and predicted == expected

        return Reward(
            reward=1.0 if correct else 0.0,
            metadata={"predicted": predicted, "expected": expected},
        )

    def _extract_final_number(self, text: str) -> str | None:
        numbers = re.findall(r'-?\d[\d,]*\.?\d*', text.replace(",", ""))
        if not numbers:
            return None
        # Normalize a trailing decimal point / zeros (e.g. "460." -> "460",
        # "460.0" -> "460") so formatting quirks don't cause false negatives.
        last = numbers[-1].rstrip(".")
        if "." in last:
            last = last.rstrip("0").rstrip(".")
        return last


# --- Helpers ---

def parse_gsm8k_answer(answer_text: str) -> str:
    """Extract the final answer number from GSM8K format (#### <number>)."""
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', answer_text)
    return match.group(1).replace(",", "") if match else ""


def prepare_gsm8k(split: str):
    """Load GSM8K and extract question + expected_answer."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {"question": row["question"], "expected_answer": parse_gsm8k_answer(row["answer"])}
        for row in ds
    ]


def extract_response_text(response) -> str:
    """Best-effort extraction of the agent's final text response."""
    try:
        return response.message["content"][0]["text"]
    except Exception:
        return str(response)


def main():
    # --- Load GSM8K ---
    print("Loading GSM8K dataset...")
    train_data = prepare_gsm8k("test[:15]")
    test_data = prepare_gsm8k("test[15:20]")
    train_loader = DataLoader(train_data, batch_size=15)
    print(f"Train: {len(train_data)} problems, Test: {len(test_data)} problems")

    # --- The Skill we will optimize ---
    # Start deliberately thin: the optimizer will grow these instructions from
    # observed success/failure patterns.
    math_skill = Skill(
        name="math-solver",
        description="Solve arithmetic word problems and report a single final number.",
        instructions=(
            "Solve the math word problem. Show your reasoning and end with the "
            "final numeric answer on the last line."
        ),
    )

    # Wrap the skill so its instructions are a tunable parameter.
    formula = SkillFormula(math_skill, tune_description=False, tune_instructions=True)

    # --- Agent setup ---
    model = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
    )

    # The AgentSkills plugin injects skill metadata and exposes the `skills` tool.
    # The adapter re-syncs the formula's skill into this plugin on every
    # invocation via set_available_skills(), so optimizer updates take effect
    # regardless of whether the plugin initially shared the same Skill object.
    skills_plugin = AgentSkills(skills=[math_skill])

    def create_agent():
        agent = Agent(
            model=model,
            system_prompt=(
                "You are a math assistant. For every arithmetic word problem, "
                "activate the `math-solver` skill and follow its instructions. "
                "You may use the shell tool as a calculator."
            ),
            plugins=[skills_plugin],
            tools=[shell],
            callback_handler=None,
        )
        # Attach the formula so the harness treats the skill's text as tunable.
        apply_formulas_on_strands_agent(agent, [formula])
        return agent

    def invoke_agent(agent, data_sample) -> Rollout:
        agent.messages.clear()
        try:
            response = agent(data_sample["question"])
            response_text = extract_response_text(response)
        except Exception as e:
            response_text = f"Error: {e}"
        return Rollout(
            data_sample=data_sample,
            messages=list(agent.messages),
            metadata={"response_text": response_text},
        )

    reward_fn = GSM8KReward()

    # LocalRolloutEngine manages an agent pool and calls invoke_agent per sample.
    engine = LocalRolloutEngine(
        formula=formula,
        agent_creator=create_agent,
        agent_invoker=invoke_agent,
    )

    # The built-in contrastive_reflection templates are param-agnostic: they loop
    # over whatever tunable params the formula exposes (here, the skill's
    # "instructions") and submit each back under its original key. The same
    # optimizer + templates work for SystemPromptFormula, SkillFormula, etc.
    optimizer = ContrastiveReflectionOptimizer(
        formula,
        system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
        task_message_template=load_builtin_template(
            "contrastive_reflection/task_message_system_prompt.jinja"
        ),
        model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        n_sample_traces=-1,
    )

    # --- Automated training loop ---
    # The Trainer iterates the DataLoader, generates rollouts via the engine,
    # scores them with reward_fn, and calls optimizer.step()/zero() per epoch —
    # each step rewrites the skill's instructions in place.
    trainer = Trainer(
        formula=formula,
        optimizer=optimizer,
        reward_fn=reward_fn,
        engine=engine,
        dataloader=train_loader,
        n_epochs=3,
    )

    print("\n=== Training the skill with Trainer ===")
    print(f"Initial instructions: {formula.get_tunable_params()['instructions'][:100]}...")
    stats = trainer.fit()
    for s in stats:
        print(f"  Epoch {s['epoch']}: avg_reward = {s['avg_reward']:.2f}")

    # --- Test evaluation with the optimized skill ---
    print(f"\n{'='*60}")
    print("Test Evaluation (with optimized skill)")
    print(f"{'='*60}")

    test_agent = create_agent()
    test_rewards = []
    for i, sample in enumerate(test_data):
        rollout = invoke_agent(test_agent, sample)
        reward = reward_fn(rollout)
        test_rewards.append(reward.reward)

        status = "correct" if reward.reward == 1.0 else "wrong"
        print(f"  [{i+1}/{len(test_data)}] {status} "
              f"(predicted={reward.metadata['predicted']}, expected={reward.metadata['expected']})")

    test_acc = sum(test_rewards) / len(test_rewards)
    print(f"\nTest accuracy: {test_acc:.0%}")
    print(f"\nFinal optimized skill instructions:\n{formula.get_tunable_params()['instructions']}")


if __name__ == "__main__":
    main()
