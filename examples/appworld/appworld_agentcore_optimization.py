"""Example: optimize an AppWorld agent's system prompt via a deployed AgentCore runtime.

AppWorld (https://appworld.dev) tasks run inside an AgentCore *runtime* — a deployed
Strands agent that talks to the AppWorld environment/APIs, executes the task, and
returns an evaluation. That heavy machinery (the ``appworld`` package, MCP/environment
servers, the ``agent_customizer`` dataset code) lives server-side in the runtime, so
this optimization script needs **only** ``strands_harness_optimizer`` plus AWS creds.

The loop:

    task_ids ─▶ AgentCore{HTTP,}RolloutEngine.invoke(runtime) ─▶ Rollout(eval_result)
                     │  payload_mapper turns the engine's                 │
                     │  {data_sample, params} into the runtime's          │
                     │  {task_id, system_prompt, exp_id}                  ▼
                     └───────────────────────────────────▶ AppWorldReward (success → 1.0)
                                                                          │
                                              ContrastiveReflectionOptimizer.step()
                                                → SystemPromptFormula.update_params()

The runtime already applies the submitted ``system_prompt`` to its agent (its
``invoke`` entrypoint reads ``payload["system_prompt"]``), so tuning the
``SystemPromptFormula`` here changes the prompt the deployed agent runs with on the
next epoch.

Prerequisites:
- An AppWorld AgentCore runtime is deployed (see the runtime's app.py entrypoint) and
  you have its ARN.
- AWS credentials with permission to call ``bedrock-agentcore:InvokeAgentRuntime``.

Runtime I/O contract (matches the deployed app.py):
    request  payload : {"task_id": str, "system_prompt": str, "exp_id": str}
    response payload : {"result", "messages", "stop_reason", "eval_result"}
        eval_result  : {"reward": float, "metrics": {"success": bool, "difficulty",
                        "num_tests", "passes", "failures"}, ...}   (make_eval_result)

Requirements:
    pip install strands-harness-optimizer

Usage — point it at the runtime one of two ways:

    # A) a LOCAL runtime container over HTTP (see examples/appworld/appworld_runtime):
    docker run -p 8080:8080 ... appworld-runtime      # start the container
    export APPWORLD_BASE_URL="http://localhost:8080"  # or APPWORLD_BASE_URLS=url1,url2 for a pool
    python examples/appworld/appworld_agentcore_optimization.py

    # B) a DEPLOYED AgentCore runtime by ARN (needs bedrock-agentcore:InvokeAgentRuntime):
    export APPWORLD_AGENT_ARN="arn:aws:bedrock-agentcore:us-west-2:123:runtime/appworld-xyz"
    python examples/appworld/appworld_agentcore_optimization.py

    export APPWORLD_TASK_IDS="task_1,task_2,task_3"   # optional; else uses the sample list
"""

import csv
import os

from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SystemPromptFormula
from strands_harness_optimizer.optimizers import ContrastiveReflectionOptimizer
from strands_harness_optimizer.rewards import RewardFunction
from strands_harness_optimizer.rollout_engines import (
    AgentCoreHTTPRolloutEngine,
    AgentCoreRolloutEngine,
)
from strands_harness_optimizer.trainer import Trainer
from strands_harness_optimizer.utils import load_builtin_template


# The starting system prompt is AppWorld's canonical code-instructions prompt (the
# operating protocol + a worked ReAct example). It's the `system_prompt` field of
# appworld_runtime/prompts/appworld_code_instructions.yaml, minus the trailing per-task
# block ("Using these APIs, now generate code... Task: {{input_str}}") — the runtime
# supplies that as the per-invocation user message. The optimizer tunes this prompt.
_PROMPT_YAML = os.path.join(
    os.path.dirname(__file__), "appworld_runtime", "prompts", "appworld_code_instructions.yaml"
)


def _load_baseline_system_prompt() -> str:
    """Read the canonical prompt YAML and drop the trailing per-task USER block."""
    import yaml

    text = yaml.safe_load(open(_PROMPT_YAML).read())["system_prompt"]
    # Split off the final "USER:\n  Using these APIs, now generate code..." block,
    # which is the per-task message the runtime renders — keep everything before it.
    marker = "Using these APIs, now generate code to solve the actual task:"
    head = text.split(marker)[0]
    # Trim a dangling trailing "USER:" label left before the marker, if present.
    return head.rstrip().removesuffix("USER:").rstrip() + "\n"


APPWORLD_SYSTEM_PROMPT = _load_baseline_system_prompt()


# --- Reward: read the runtime's evaluation result ---

class AppWorldReward(RewardFunction):
    """Reward 1.0 when the AppWorld task succeeded.

    The AgentCore runtime returns ``eval_result`` (a make_eval_result dict); the
    engine stores it at ``rollout.metadata["eval_result"]``. Task success lives at
    ``eval_result["metrics"]["success"]``. We fall back to the top-level ``reward``
    field if metrics are absent.
    """

    def __call__(self, rollout: Rollout) -> Reward:
        eval_result = rollout.metadata.get("eval_result") or {}
        metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
        success = bool(metrics.get("success", False))
        reward = 1.0 if success else float(eval_result.get("reward", 0.0) or 0.0)
        return Reward(
            reward=reward,
            metadata={
                "success": success,
                "passes": metrics.get("passes"),
                "failures": metrics.get("failures"),
                "difficulty": metrics.get("difficulty"),
            },
        )


# --- Payload mapping: canonical engine payload -> this runtime's shape ---

def appworld_payload_mapper(payload: dict) -> dict:
    """Map the engine's canonical payload to what the AppWorld runtime expects.

    AgentCoreRolloutEngine builds ``{"data_sample": {...}, "params": {...}}``. The
    deployed runtime's ``invoke`` reads flat ``task_id`` / ``system_prompt`` / ``exp_id``
    instead, so we flatten here. ``params`` carries the tuned SystemPromptFormula value.
    """
    data_sample = payload.get("data_sample", {})
    params = payload.get("params", {})
    return {
        "task_id": data_sample.get("task_id", data_sample.get("id")),
        "system_prompt": params.get("system_prompt", ""),
        "exp_id": data_sample.get("exp_id", "harness-optimizer"),
    }


# --- Task data ---

def load_task_samples() -> list[dict]:
    """Load the task ids to train on, in priority order:

    1. APPWORLD_TASK_IDS — an explicit comma-separated list (a quick subset).
    2. APPWORLD_TASK_CSV — a split CSV (its `task_id` column); this is the full
       split. run_example.sh extracts <split>.csv from the runtime container into
       ./datasets/appworld/ and points this at it, so training uses the whole split.

    One of the two must be set — there is no baked-in sample list (a hardcoded handful
    is not a real training run).
    """
    env_ids = os.getenv("APPWORLD_TASK_IDS")
    if env_ids:
        task_ids = [t.strip() for t in env_ids.split(",") if t.strip()]
    else:
        csv_path = os.getenv("APPWORLD_TASK_CSV")
        if not csv_path:
            raise SystemExit(
                "Set APPWORLD_TASK_CSV to a split CSV (full split), or APPWORLD_TASK_IDS "
                "to a comma-separated subset. run_example.sh does the CSV extraction for you."
            )
        with open(csv_path, newline="") as f:
            task_ids = [row["task_id"] for row in csv.DictReader(f) if row.get("task_id")]
        if not task_ids:
            raise SystemExit(f"No task_id rows found in {csv_path}")

    return [{"task_id": tid, "id": tid} for tid in task_ids]


def main():
    # Two ways to reach the AppWorld runtime:
    #   - APPWORLD_BASE_URL(S): a local runtime *container* over HTTP (what
    #     examples/appworld/appworld_runtime builds) — AgentCoreHTTPRolloutEngine.
    #   - APPWORLD_AGENT_ARN: a *deployed* AgentCore runtime — AgentCoreRolloutEngine.
    base_urls = os.getenv("APPWORLD_BASE_URLS") or os.getenv("APPWORLD_BASE_URL")
    agent_arn = os.getenv("APPWORLD_AGENT_ARN")
    if not base_urls and not agent_arn:
        raise SystemExit(
            "Set one of:\n"
            "  APPWORLD_BASE_URL(S)  — local runtime container(s), e.g. 'http://localhost:8080'\n"
            "                          (comma-separated for a pool of containers)\n"
            "  APPWORLD_AGENT_ARN    — a deployed AgentCore runtime ARN"
        )

    task_samples = load_task_samples()
    train_loader = DataLoader(task_samples, batch_size=len(task_samples))
    print(f"AppWorld tasks: {len(task_samples)}")

    # The prompt we optimize. The runtime applies whatever system_prompt we send.
    # Seed it with AppWorld's required operating instructions (adapted from AppWorld's
    # official full_code agent prompt) — the `apis` object, the supervisor-login →
    # access_token idiom, the no-OS-modules rule, and the mandatory complete_task call.
    # Without these the agent can't authenticate (401s) or submit an answer. The
    # optimizer then refines this working baseline.
    formula = SystemPromptFormula(system_prompt=APPWORLD_SYSTEM_PROMPT)

    # payload_mapper reshapes the canonical {data_sample, params} payload into the
    # runtime's {task_id, system_prompt, exp_id} for either transport.
    if base_urls:
        urls = [u.strip() for u in base_urls.split(",") if u.strip()]
        print(f"Driving {len(urls)} local runtime container(s) over HTTP: {urls}")
        engine = AgentCoreHTTPRolloutEngine(
            formula=formula,
            base_urls=urls,
            num_workers=len(urls),  # one in-flight request per container
            payload_mapper=appworld_payload_mapper,
        )
    else:
        print("Driving a deployed AgentCore runtime by ARN")
        engine = AgentCoreRolloutEngine(
            formula=formula,
            agent_arn=agent_arn,
            region_name=os.getenv("AWS_REGION", "us-west-2"),
            num_workers=4,
            payload_mapper=appworld_payload_mapper,
        )

    optimizer = ContrastiveReflectionOptimizer(
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
        reward_fn=AppWorldReward(),
        engine=engine,
        dataloader=train_loader,
        n_epochs=1,
    )

    print("\n=== Optimizing AppWorld system prompt (via AgentCore runtime) ===")
    print(f"Initial prompt: {formula.get_tunable_params()['system_prompt'][:100]}...")
    stats = trainer.fit()
    for s in stats:
        print(f"  Epoch {s['epoch']}: avg_reward (success rate) = {s['avg_reward']:.2f}")

    print(f"\nFinal optimized prompt:\n{formula.get_tunable_params()['system_prompt']}")


if __name__ == "__main__":
    main()
