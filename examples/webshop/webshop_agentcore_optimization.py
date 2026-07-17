"""Example: optimize a WebShop agent's system prompt with a **multi-agent** optimizer.

WebShop (https://webshop-pnlp.github.io) tasks run inside an AgentCore *runtime* — a
deployed Strands agent that talks to the WebShop gym environment (search / click /
buy), completes the shopping task, and returns an evaluation. That heavy machinery
(the WebShop gym, its search index, the dual-venv setup) lives server-side in the
runtime, so this optimization script needs **only** ``strands_harness_optimizer`` plus
AWS creds.

The loop:

    task_ids ─▶ AgentCore{HTTP,}RolloutEngine.invoke(runtime) ─▶ Rollout(eval_result)
                     │  payload_mapper turns the engine's                 │
                     │  {data_sample, params} into the runtime's          │
                     │  {task_id, system_prompt, exp_id}                  ▼
                     └───────────────────────────────────▶ WebShopReward (score → 1.0)
                                                                          │
                                              MultiAgentOptimizer.step()
                                                → swarm of sub-agents analyzes rollouts
                                                → SystemPromptFormula.update_params()

**Sub-agents.** Unlike the AppWorld example (which uses a single reflection agent),
this one uses ``MultiAgentOptimizer``: the reflection step is an orchestrator agent
that spins up a **swarm of sub-agents** (via the strands ``swarm`` tool) to analyze
the WebShop rollouts in parallel — each sub-agent digs into a subset of trajectories
and reports contrastive success/failure signals, which the orchestrator aggregates
into the updated system prompt. This mirrors the internal ``webshop_swarm_*``
optimization configs.

Runtime I/O contract (matches the deployed app.py):
    request  payload : {"task_id": str, "system_prompt": str, "exp_id": str}
    response payload : {"result", "messages", "stop_reason", "eval_result"}
        eval_result  : {"reward": float, "metrics": {"success": bool, "score": float,
                        "done": bool}, ...}   (make_eval_result)

Requirements:
    pip install strands-harness-optimizer

Usage — point it at the runtime one of two ways:

    # A) a LOCAL runtime container over HTTP (see examples/webshop/webshop_runtime):
    docker run -p 8080:8080 ... webshop-runtime      # start the container
    export WEBSHOP_BASE_URL="http://localhost:8080"  # or WEBSHOP_BASE_URLS=url1,url2 for a pool
    python examples/webshop/webshop_agentcore_optimization.py

    # B) a DEPLOYED AgentCore runtime by ARN (needs bedrock-agentcore:InvokeAgentRuntime):
    export WEBSHOP_AGENT_ARN="arn:aws:bedrock-agentcore:us-west-2:123:runtime/webshop-xyz"
    python examples/webshop/webshop_agentcore_optimization.py

    export WEBSHOP_TASK_IDS="0,1,2,3"   # optional subset; else the whole split range
"""

import os

from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SystemPromptFormula
from strands_harness_optimizer.optimizers import MultiAgentOptimizer
from strands_harness_optimizer.rewards import RewardFunction
from strands_harness_optimizer.rollout_engines import (
    AgentCoreHTTPRolloutEngine,
    AgentCoreRolloutEngine,
)
from strands_harness_optimizer.trainer import Trainer
from strands_harness_optimizer.utils import load_builtin_template


# The starting system prompt is WebShop's operating instructions (tool protocol +
# shopping strategy + a worked example). It's the `system_prompt` field of
# webshop_runtime/prompts/webshop_instructions.yaml. The optimizer tunes this prompt;
# the per-task shopping goal is supplied by the runtime as the user message.
_PROMPT_YAML = os.path.join(
    os.path.dirname(__file__), "webshop_runtime", "prompts", "webshop_instructions.yaml"
)


def _load_baseline_system_prompt() -> str:
    import yaml

    return yaml.safe_load(open(_PROMPT_YAML).read())["system_prompt"]


WEBSHOP_SYSTEM_PROMPT = _load_baseline_system_prompt()


# --- Reward: read the runtime's evaluation result ---

class WebShopReward(RewardFunction):
    """Reward = WebShop's task score in [0, 1].

    The AgentCore runtime returns ``eval_result`` (a make_eval_result dict); the engine
    stores it at ``rollout.metadata["eval_result"]``. WebShop pays a partial-match score
    in [0, 1] on the final "buy now"; a full match is 1.0. We use that score directly as
    the reward, and expose ``success`` (score == 1.0) in the metadata.
    """

    def __call__(self, rollout: Rollout) -> Reward:
        eval_result = rollout.metadata.get("eval_result") or {}
        metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
        score = float(metrics.get("score", eval_result.get("reward", 0.0)) or 0.0)
        return Reward(
            reward=score,
            metadata={
                "success": bool(metrics.get("success", score >= 1.0)),
                "score": score,
                "done": metrics.get("done"),
            },
        )


# --- Payload mapping: canonical engine payload -> this runtime's shape ---

def webshop_payload_mapper(payload: dict) -> dict:
    """Map the engine's canonical payload to what the WebShop runtime expects.

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
    """Load the task ids to train on.

    1. WEBSHOP_TASK_IDS — an explicit comma-separated list of goal indices (quick subset).
    2. Otherwise the whole split range: WEBSHOP_SPLIT (train => 0..99, eval => 0..199).

    WebShop tasks are integer goal indices, so no CSV extraction is needed — the range
    is defined by the split (see webshop_runtime/dataset/dataset.py).
    """
    env_ids = os.getenv("WEBSHOP_TASK_IDS")
    if env_ids:
        task_ids = [t.strip() for t in env_ids.split(",") if t.strip()]
    else:
        split = os.getenv("WEBSHOP_SPLIT", "train")
        ranges = {"train": range(0, 100), "eval": range(0, 200)}
        if split not in ranges:
            raise SystemExit(f"Unknown WEBSHOP_SPLIT '{split}'. Use 'train' or 'eval'.")
        task_ids = [str(i) for i in ranges[split]]

    return [{"task_id": tid, "id": tid} for tid in task_ids]


def main():
    # Two ways to reach the WebShop runtime:
    #   - WEBSHOP_BASE_URL(S): a local runtime *container* over HTTP (what
    #     examples/webshop/webshop_runtime builds) — AgentCoreHTTPRolloutEngine.
    #   - WEBSHOP_AGENT_ARN: a *deployed* AgentCore runtime — AgentCoreRolloutEngine.
    base_urls = os.getenv("WEBSHOP_BASE_URLS") or os.getenv("WEBSHOP_BASE_URL")
    agent_arn = os.getenv("WEBSHOP_AGENT_ARN")
    if not base_urls and not agent_arn:
        raise SystemExit(
            "Set one of:\n"
            "  WEBSHOP_BASE_URL(S)  — local runtime container(s), e.g. 'http://localhost:8080'\n"
            "                         (comma-separated for a pool of containers)\n"
            "  WEBSHOP_AGENT_ARN    — a deployed AgentCore runtime ARN"
        )

    task_samples = load_task_samples()
    train_loader = DataLoader(task_samples, batch_size=len(task_samples))
    print(f"WebShop tasks: {len(task_samples)}")

    # The prompt we optimize. The runtime applies whatever system_prompt we send.
    formula = SystemPromptFormula(system_prompt=WEBSHOP_SYSTEM_PROMPT)

    # payload_mapper reshapes the canonical {data_sample, params} payload into the
    # runtime's {task_id, system_prompt, exp_id} for either transport.
    if base_urls:
        urls = [u.strip() for u in base_urls.split(",") if u.strip()]
        print(f"Driving {len(urls)} local runtime container(s) over HTTP: {urls}")
        engine = AgentCoreHTTPRolloutEngine(
            formula=formula,
            base_urls=urls,
            num_workers=len(urls),  # one in-flight request per container
            payload_mapper=webshop_payload_mapper,
        )
    else:
        print("Driving a deployed AgentCore runtime by ARN")
        engine = AgentCoreRolloutEngine(
            formula=formula,
            agent_arn=agent_arn,
            region_name=os.getenv("AWS_REGION", "us-west-2"),
            num_workers=4,
            payload_mapper=webshop_payload_mapper,
        )

    # The multi-agent optimizer: the reflection step spins up a swarm of sub-agents
    # (via the strands `swarm` tool) to analyze the WebShop rollouts in parallel, then
    # aggregates their contrastive findings into the updated system prompt. The
    # `rollout_analyzer_template` is the role each sub-agent is given.
    optimizer = MultiAgentOptimizer(
        formula,
        system_prompt_template=load_builtin_template("multi_agent/system_prompt.jinja"),
        task_message_template=load_builtin_template(
            "multi_agent/task_message_system_prompt.jinja"
        ),
        rollout_analyzer_template=load_builtin_template("multi_agent/rollout_analyzer.jinja"),
        model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        n_sample_traces=-1,
    )

    trainer = Trainer(
        formula=formula,
        optimizer=optimizer,
        reward_fn=WebShopReward(),
        engine=engine,
        dataloader=train_loader,
        n_epochs=1,
    )

    print("\n=== Optimizing WebShop system prompt (multi-agent, via AgentCore runtime) ===")
    print(f"Initial prompt: {formula.get_tunable_params()['system_prompt'][:100]}...")
    stats = trainer.fit()
    for s in stats:
        print(f"  Epoch {s['epoch']}: avg_reward (task score) = {s['avg_reward']:.2f}")

    print(f"\nFinal optimized prompt:\n{formula.get_tunable_params()['system_prompt']}")


if __name__ == "__main__":
    main()
