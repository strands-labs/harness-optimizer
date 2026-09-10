"""Example: curate a skill LIBRARY from AppWorld rollout traces.

Optimizes the SET of skills an agent has — which skills should exist at all —
rather than the wording of a system prompt. An agent reads the rollout traces,
decides what to create / revise / retire, and the next iteration runs with the
curated library.

```
appworld_skill_library_optimization.py             appworld_runtime/  (the deployable runtime)
  DataLoader(task_ids)                        app.py — AgentCore /invocations entrypoint
  AgentCoreHTTPRolloutEngine ──HTTP POST──▶      → installs skills_folder, runs the task,
  (or AgentCoreRolloutEngine, by ARN)              returns {messages, eval_result,
        │  payload_mapper: {data_sample,params}                skills_applied}
        ▼            → {task_id, system_prompt, skills_folder, exp_id}
  AppWorldReward (eval_result.metrics.success → 1.0)
        │
  SkillLibraryOptimizer.step()  ← curator agent reads the traces, writes
        │                          create/ optimize/ retire.txt + manifest.json
        ▼
  SkillLibraryFormula.update_params()  → existing − retired + created
  formula.materialize(...)             → the flat <name>/SKILL.md folder
```

Compare with ``appworld_agentcore_optimization.py``, which tunes the system prompt
on the same runtime and the same traces.

Skills are delivered INLINE (``skills_folder`` = ``[{path, content}]``) rather than
via an S3 pointer, so this needs no bucket and no extra IAM: the skill text travels
in the payload, which also means a saved trace records exactly what the agent read.

Requires the runtime to be reachable. ``run_example.sh --skills`` builds it, starts
it, and runs this script against it.
"""

import json
import os
import sys

# The reward and the task loader are identical to the system-prompt example on this
# runtime -- importing them keeps one definition rather than a copy that can drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from appworld_agentcore_optimization import (  # noqa: E402
    APPWORLD_SYSTEM_PROMPT,
    AppWorldReward,
    load_task_samples,
)

from strands_harness_optimizer.data import DataLoader  # noqa: E402
from strands_harness_optimizer.formulas import SkillLibraryFormula  # noqa: E402
from strands_harness_optimizer.optimizers import SkillLibraryOptimizer  # noqa: E402
from strands_harness_optimizer.rollout_engines import (  # noqa: E402
    AgentCoreHTTPRolloutEngine,
    AgentCoreRolloutEngine,
)
from strands_harness_optimizer.utils import load_builtin_template  # noqa: E402

OUTPUT_ROOT = os.getenv("SKILL_OUTPUT_ROOT", "./appworld_skill_runs")
ITERATIONS = int(os.getenv("SKILL_ITERATIONS", "1"))
OPTIMIZER_MODEL = os.getenv(
    "SKILL_OPTIMIZER_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
# -1 = give the curator every trace. Raise the rollout count and lower this to
# sample instead (stratified 50/50 success/failure).
N_SAMPLE_TRACES = int(os.getenv("SKILL_N_SAMPLE_TRACES", "-1"))


def pack_inline(skill_dir: str) -> list[dict]:
    """Pack a materialized skill folder as ``[{"path", "content"}]``.

    Walks the whole tree, so a skill's ``scripts/`` or ``resources/`` travel with
    it. Raises on a file that is not text rather than skipping it -- a silently
    dropped resource is a skill that fails at runtime for no visible reason.
    """
    out = []
    for root, _, files in os.walk(skill_dir):
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, skill_dir)
            try:
                with open(full) as f:
                    out.append({"path": rel, "content": f.read()})
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"{full} is not text and cannot be inlined. Drop it, or deliver "
                    "this skill set via S3 instead."
                ) from e
    return out


def make_payload_mapper(skills_ref: dict):
    """Build a payload_mapper that ships the CURRENT skill set every batch.

    ``skills_ref`` is read at call time, not captured by value: the engine calls
    ``ensure_sync_params()`` per batch but the mapper is constructed once, so
    binding the packed skills here would keep sending iteration N-1's library
    forever -- the run would report iterating while measuring the same thing twice.
    """

    def mapper(payload: dict) -> dict:
        data_sample = payload.get("data_sample", {})
        return {
            "task_id": data_sample.get("task_id", data_sample.get("id")),
            # The library is a skills-only intervention, so the prompt stays the
            # shipped baseline. Sending an optimized prompt too would mix two
            # surfaces and make the delta unattributable.
            "system_prompt": APPWORLD_SYSTEM_PROMPT,
            "skills_folder": skills_ref.get("packed", []),
            "exp_id": data_sample.get("exp_id", "harness-optimizer-skills"),
        }

    return mapper


def build_engine(formula, skills_ref):
    """HTTP to a local container, or ARN to a deployed runtime."""
    base_urls = os.getenv("APPWORLD_BASE_URLS") or os.getenv("APPWORLD_BASE_URL")
    agent_arn = os.getenv("APPWORLD_AGENT_ARN")
    if not base_urls and not agent_arn:
        raise SystemExit(
            "Set one of:\n"
            "  APPWORLD_BASE_URL(S)  — local runtime container(s), e.g. 'http://localhost:8080'\n"
            "                          (comma-separated for a pool of containers)\n"
            "  APPWORLD_AGENT_ARN    — a deployed AgentCore runtime ARN"
        )
    mapper = make_payload_mapper(skills_ref)
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


def report_activation(rollouts, expected: int) -> None:
    """Say whether the skills actually LOADED, not just whether they were sent.

    A runtime that predates the ``skills_folder`` key ignores it silently, and the
    resulting flat reward is indistinguishable from "the skills did not help". The
    runtime echoes ``skills_applied`` so that case is visible.
    """
    if expected == 0:
        return
    seen = [r.metadata.get("skills_applied") for r in rollouts]
    reported = [s for s in seen if s is not None]
    if not reported:
        print(
            "  WARNING no rollout reported `skills_applied` -- the runtime ignored "
            "`skills_folder`, so this iteration measured the UNCHANGED agent. "
            "Rebuild the runtime image."
        )
    elif all(s == 0 for s in reported):
        print(f"  WARNING every rollout reported skills_applied=0 of {expected} sent.")
    else:
        print(f"  skills loaded in-runtime: {max(reported)} of {expected} sent")


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
    print(f"AppWorld tasks: {len(task_samples)}")

    # Start from whatever library is on disk (SKILL_DIR), or cold -- an empty
    # library is the normal first iteration, where every action is a CREATE.
    formula = SkillLibraryFormula(skill_dir=os.getenv("SKILL_DIR") or None)
    print(f"Starting library: {formula.skill_names or '(empty — cold start)'}")

    # Mutable holder so the payload_mapper always ships the CURRENT set.
    skills_ref: dict = {"packed": pack_inline(formula.skill_dir) if formula.skill_dir else []}

    engine = build_engine(formula, skills_ref)
    reward_fn = AppWorldReward()
    loader = DataLoader(task_samples, batch_size=len(task_samples))

    history = []
    for it in range(ITERATIONS):
        print(f"\n{'=' * 60}\n  ITERATION {it}\n{'=' * 60}")

        print(f"[rollout] {len(task_samples)} task(s) with "
              f"{len(formula.skill_names)} skill(s)")
        rollouts, rewards, mean = rollout(engine, reward_fn, loader)
        report_activation(rollouts, len(formula.skill_names))
        print(f"[rollout] mean reward: {mean:.4f}")
        history.append({"iteration": it, "skills": list(formula.skill_names),
                        "mean_reward": mean})

        out_dir = os.path.join(OUTPUT_ROOT, f"iter_{it}")
        optimizer = SkillLibraryOptimizer(
            formula,
            system_prompt_template=load_builtin_template(
                "skill_library/system_prompt.jinja"),
            task_message_template=load_builtin_template(
                "skill_library/task_message.jinja"),
            output_folder=out_dir,
            model_config={"model_id": OPTIMIZER_MODEL},
            n_sample_traces=N_SAMPLE_TRACES,
        )
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)

        print("[curate] running the curator agent ...")
        optimizer.step()
        print(f"[curate] decisions: "
              f"{ {k: len(v) for k, v in optimizer.last_decisions.items()} }")

        if not any(optimizer.last_decisions.values()):
            # Not a failure. Once the recurring patterns are covered, stopping is
            # the right move -- the curator says why in manifest.json.
            print(f"[curate] SKIP — library unchanged; see "
                  f"{os.path.join(out_dir, 'manifest.json')}")
            continue

        skill_set = formula.materialize(os.path.join(out_dir, "skill_set"))
        skills_ref["packed"] = pack_inline(skill_set)
        print(f"[curate] library now {formula.skill_names}")
        print(f"[curate] materialized -> {skill_set} "
              f"({len(skills_ref['packed'])} file(s) inlined)")

    # A final rollout so the last iteration's library is actually measured; without
    # it the run reports a library nobody ran.
    print(f"\n{'=' * 60}\n  FINAL EVALUATION\n{'=' * 60}")
    _, _, final_mean = rollout(engine, reward_fn, loader)
    history.append({"iteration": "final", "skills": list(formula.skill_names),
                    "mean_reward": final_mean})

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nreward by iteration:")
    for h in history:
        print(f"  {str(h['iteration']):>5}  {h['mean_reward']:.4f}  "
              f"({len(h['skills'])} skill(s))")
    print(f"\nartifacts: {OUTPUT_ROOT}")
    print(f"final library: {formula.skill_names}")
    print(
        f"\n  NOTE {len(task_samples)} task(s), 1 rollout each. A reward delta this "
        "size is directional at best -- raise the task count and repeat the eval "
        "before treating it as a result."
    )


if __name__ == "__main__":
    main()
