"""Example: optimize WebShop's system prompt, skill library and tool descriptions TOGETHER.

One reflector reads the rollout traces and places each finding in the one surface
that reaches the agent when it matters: a rule every task needs goes in the system
prompt, a situation-specific workflow becomes a skill, a fact about one tool's inputs
or results goes in that tool's description. The next iteration runs with all three.

```
webshop_multi_surface_optimization.py         webshop_runtime/  (the deployable runtime)
  DataLoader(task_ids)                    app.py — AgentCore /invocations entrypoint
  AgentCoreHTTPRolloutEngine ──HTTP POST──▶  → applies system_prompt, installs skills_folder,
  (or AgentCoreRolloutEngine, by ARN)         patches tool_descriptions, runs the task,
        │  payload_mapper: {data_sample,params}  returns {messages, eval_result,
        ▼            → {task_id, system_prompt,   skills_applied, tool_descriptions_applied}
  WebShopReward       skills_folder, tool_descriptions, exp_id}
        │
  MultiSurfaceOptimizer.step()  ← reflector reads the traces, writes into step_NNNN/:
        │                         skills/{create,update}/, system_prompt/, tool_descriptions/,
        ▼                         findings.json
  MultiSurfaceFormula.update_params()  → prompt text, skill set (materialized), tool overrides
```

Compare with ``webshop_agentcore_optimization.py`` (prompt only) and
``webshop_skill_library_optimization.py`` (skills only) on the same runtime.

Delivery is INLINE for all three surfaces: the prompt as text, the skill set packed as
``[{path, content}]``, and the tool descriptions as the SPARSE map of edited tools. The
runtime patches only the tools it is sent; every other tool keeps its own docstring.

Requires the runtime to be reachable. ``run_example.sh --multi`` builds it, starts it,
and runs this script against it.
"""

import json
import os
import sys

# Reward, task loader and baseline prompt are the prompt example's; the inline packer
# is the skill-library example's. One definition each rather than copies that drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from webshop_agentcore_optimization import (  # noqa: E402
    WEBSHOP_SYSTEM_PROMPT,
    WebShopReward,
    load_task_samples,
)
from webshop_skill_library_optimization import pack_inline  # noqa: E402

from strands_harness_optimizer.data import DataLoader  # noqa: E402
from strands_harness_optimizer.formulas import (  # noqa: E402
    MultiSurfaceFormula,
    SkillLibraryFormula,
    SystemPromptFormula,
    ToolDescriptionFormula,
)
from strands_harness_optimizer.optimizers import MultiSurfaceOptimizer  # noqa: E402
from strands_harness_optimizer.rollout_engines import (  # noqa: E402
    AgentCoreHTTPRolloutEngine,
    AgentCoreRolloutEngine,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.getenv("MULTI_OUTPUT_ROOT", "./webshop_multi_surface_runs")
ITERATIONS = int(os.getenv("MULTI_ITERATIONS", "1"))
OPTIMIZER_MODEL = os.getenv("MULTI_OPTIMIZER_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")
# -1 = give the reflector every trace. Raise the rollout count and lower this to
# sample instead (stratified 50/50 success/failure).
N_SAMPLE_TRACES = int(os.getenv("MULTI_N_SAMPLE_TRACES", "-1"))
# The runtime's own tool descriptions, generated from its @tool docstrings by
# webshop_runtime/dump_tool_descriptions.py. The reflector starts from this map.
TOOL_DESCRIPTIONS_YAML = os.getenv(
    "TOOL_DESCRIPTIONS_YAML",
    os.path.join(HERE, "webshop_runtime", "prompts", "tool_descriptions.yaml"),
)


def make_payload_mapper(formula: MultiSurfaceFormula, delivery: dict):
    """Ship the CURRENT state of all three surfaces on every batch.

    ``formula`` is read at call time (prompt text and tool overrides come straight
    from ``get_tunable_params``); ``delivery["skills"]`` holds the packed skill set,
    refreshed after each step. The engine constructs the mapper once, so capturing
    values here would keep sending iteration N-1's artifacts forever.
    """

    def mapper(payload: dict) -> dict:
        data_sample = payload.get("data_sample", {})
        params = formula.get_tunable_params()
        return {
            "task_id": data_sample.get("task_id", data_sample.get("id")),
            "system_prompt": params["system_prompt"],
            "skills_folder": delivery.get("skills", []),
            # Sparse: only the tools the optimizer edited. The runtime patches these
            # and leaves every other tool's description as its docstring gave it.
            "tool_descriptions": params["tool_descriptions"],
            "exp_id": data_sample.get("exp_id", "harness-optimizer-multi-surface"),
        }

    return mapper


def build_engine(formula, delivery):
    """HTTP to a local container, or ARN to a deployed runtime."""
    base_urls = os.getenv("WEBSHOP_BASE_URLS") or os.getenv("WEBSHOP_BASE_URL")
    agent_arn = os.getenv("WEBSHOP_AGENT_ARN")
    if not base_urls and not agent_arn:
        raise SystemExit(
            "Set one of:\n"
            "  WEBSHOP_BASE_URL(S)  — local runtime container(s), e.g. 'http://localhost:8080'\n"
            "                          (comma-separated for a pool of containers)\n"
            "  WEBSHOP_AGENT_ARN    — a deployed AgentCore runtime ARN"
        )
    mapper = make_payload_mapper(formula, delivery)
    if base_urls:
        urls = [u.strip() for u in base_urls.split(",") if u.strip()]
        print(f"Driving {len(urls)} local runtime container(s) over HTTP: {urls}")
        return AgentCoreHTTPRolloutEngine(
            formula=formula,
            base_urls=urls,
            num_workers=len(urls),  # one in-flight request per container
            payload_mapper=mapper,
        )
    print("Driving a deployed AgentCore runtime by ARN")
    return AgentCoreRolloutEngine(
        formula=formula,
        agent_arn=agent_arn,
        region_name=os.getenv("AWS_REGION", "us-west-2"),
        num_workers=4,
        payload_mapper=mapper,
    )


def report_activation(rollouts, n_skills: int, tool_overrides: list) -> None:
    """Say whether the skills and tool edits actually REACHED the agent.

    A runtime that predates a payload key ignores it silently, and the resulting
    flat reward is indistinguishable from "the change did not help". The runtime
    echoes what it applied so that case is visible.
    """
    if n_skills:
        seen = [r.metadata.get("skills_applied") for r in rollouts]
        reported = [s for s in seen if s is not None]
        if not reported:
            print(
                "  WARNING no rollout reported `skills_applied` -- the runtime ignored "
                "`skills_folder`. Rebuild the runtime image."
            )
        elif all(s == 0 for s in reported):
            print(f"  WARNING every rollout reported skills_applied=0 of {n_skills} sent.")
        else:
            print(f"  skills loaded in-runtime: {max(reported)} of {n_skills} sent")
    if tool_overrides:
        seen = [r.metadata.get("tool_descriptions_applied") for r in rollouts]
        reported = [s for s in seen if s is not None]
        if not reported:
            print(
                "  WARNING no rollout reported `tool_descriptions_applied` -- the runtime "
                "ignored `tool_descriptions`. Rebuild the runtime image."
            )
        else:
            applied = sorted({name for s in reported for name in (s or [])})
            missing = sorted(set(tool_overrides) - set(applied))
            print(f"  tool descriptions patched in-runtime: {applied or 'none'}")
            if missing:
                print(f"  WARNING sent but never applied: {missing}")


def rollout(engine, reward_fn, loader):
    rollouts, rewards = [], []
    for batch in loader:
        for r in engine.generate_batch(batch):
            rollouts.append(r)
            rewards.append(reward_fn(r))
    scored = [w.reward for w in rewards]
    mean = sum(scored) / len(scored) if scored else 0.0
    return rollouts, rewards, mean


def main():
    task_samples = load_task_samples()
    print(f"WebShop tasks: {len(task_samples)}")

    formula = MultiSurfaceFormula(
        system_prompt=SystemPromptFormula(system_prompt=WEBSHOP_SYSTEM_PROMPT),
        # Start from whatever library is on disk (SKILL_DIR), or cold.
        skills=SkillLibraryFormula(skill_dir=os.getenv("SKILL_DIR") or None),
        tool_descriptions=ToolDescriptionFormula.from_yaml(TOOL_DESCRIPTIONS_YAML),
    )
    print(f"Starting skills: {formula.skills.skill_names or '(empty — cold start)'}")
    print(f"Toolset: {formula.tool_descriptions.tool_names}")

    # Mutable holder for the packed skill set; refreshed after every step.
    delivery: dict = {
        "skills": pack_inline(formula.skills.skill_dir) if formula.skills.skill_dir else []
    }

    engine = build_engine(formula, delivery)
    reward_fn = WebShopReward()
    loader = DataLoader(task_samples, batch_size=len(task_samples))

    # ONE optimizer for the whole run: each step() writes its own step_NNNN/ under
    # OUTPUT_ROOT, and earlier steps are never read again.
    optimizer = MultiSurfaceOptimizer(
        formula,
        output_folder=OUTPUT_ROOT,
        model_config={"model_id": OPTIMIZER_MODEL},
        n_sample_traces=N_SAMPLE_TRACES,
    )

    history = []
    for it in range(ITERATIONS):
        print(f"\n{'=' * 60}\n  ITERATION {it}\n{'=' * 60}")

        overrides = sorted(formula.tool_descriptions.overrides)
        print(
            f"[rollout] {len(task_samples)} task(s) with {len(formula.skills.skill_names)} "
            f"skill(s) and {len(overrides)} tool override(s)"
        )
        rollouts, rewards, mean = rollout(engine, reward_fn, loader)
        report_activation(rollouts, len(formula.skills.skill_names), overrides)
        print(f"[rollout] mean reward: {mean:.4f}")
        history.append(
            {
                "iteration": it,
                "skills": list(formula.skills.skill_names),
                "tool_overrides": overrides,
                "prompt_chars": len(formula.system_prompt.system_prompt or ""),
                "mean_reward": mean,
            }
        )

        optimizer.zero()
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)

        print("[reflect] running the multi-surface reflector ...")
        optimizer.step()
        n_findings = len((optimizer.last_findings or {}).get("findings", []))
        print(
            f"[reflect] applied {optimizer.last_applied or 'nothing'}; "
            f"dropped {sorted(optimizer.last_dropped) or 'nothing'}; "
            f"{n_findings} finding(s) -> {optimizer.last_step_dir}"
        )

        # Skills were materialized into this step's skill_set/ by the optimizer.
        delivery["skills"] = pack_inline(formula.skills.skill_dir) if formula.skills.members else []
        print(
            f"[reflect] now: {len(formula.skills.skill_names)} skill(s) "
            f"{formula.skills.skill_names}, tool overrides "
            f"{sorted(formula.tool_descriptions.overrides)}, prompt "
            f"{len(formula.system_prompt.system_prompt or '')} chars"
        )

    # A final rollout so the last iteration's artifacts are actually measured.
    print(f"\n{'=' * 60}\n  FINAL EVALUATION\n{'=' * 60}")
    final_rollouts, _, final_mean = rollout(engine, reward_fn, loader)
    report_activation(
        final_rollouts,
        len(formula.skills.skill_names),
        sorted(formula.tool_descriptions.overrides),
    )
    history.append(
        {
            "iteration": "final",
            "skills": list(formula.skills.skill_names),
            "tool_overrides": sorted(formula.tool_descriptions.overrides),
            "prompt_chars": len(formula.system_prompt.system_prompt or ""),
            "mean_reward": final_mean,
        }
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nreward by iteration:")
    for h in history:
        print(
            f"  {str(h['iteration']):>5}  {h['mean_reward']:.4f}  "
            f"({len(h['skills'])} skill(s), {len(h['tool_overrides'])} tool override(s), "
            f"prompt {h['prompt_chars']} chars)"
        )
    print(f"\nartifacts: {OUTPUT_ROOT}")
    print(
        f"\n  NOTE {len(task_samples)} task(s), 1 rollout each. A reward delta this "
        "size is directional at best -- raise the task count and repeat the eval "
        "before treating it as a result."
    )


if __name__ == "__main__":
    main()
