# AppWorld optimization example

Optimize an agent's system prompt on [AppWorld](https://appworld.dev) tasks with
`strands_harness_optimizer`. AppWorld tasks run inside a deployable **AgentCore runtime**
(a Strands agent that talks to the AppWorld environment); the **optimizer** drives that
runtime over HTTP (local container) or by ARN (deployed), scores each task by success,
and rewrites the system prompt with `ContrastiveReflectionOptimizer`.

```
appworld_agentcore_optimization.py         appworld_runtime/  (the deployable runtime)
  DataLoader(task_ids)                        app.py  — AgentCore /invocations entrypoint
  AgentCoreHTTPRolloutEngine ──HTTP POST──▶      → AppWorldDataset runs the task,
  (or AgentCoreRolloutEngine, by ARN)              returns {messages, eval_result}
        │  payload_mapper: {data_sample,params}
        ▼            → {task_id, system_prompt, exp_id}
  AppWorldReward (eval_result.metrics.success → 1.0)
        │
  ContrastiveReflectionOptimizer.step() → SystemPromptFormula.update_params()
```

## Two pieces

| Path | What it is |
|------|-----------|
| [`appworld_agentcore_optimization.py`](appworld_agentcore_optimization.py) | The **optimizer** (client). Runs the Trainer loop against the runtime. Needs only `strands_harness_optimizer` + a runtime endpoint. |
| [`appworld_runtime/`](appworld_runtime/) | The **deployable runtime** (container). Builds a Strands+AppWorld agent as an AgentCore runtime. See its [README](appworld_runtime/README.md). |
| [`appworld_skill_library_optimization.py`](appworld_skill_library_optimization.py) | The **skill-library** optimizer (client). Curates the SET of skills the agent has, instead of tuning the system prompt. |
| [`run_example.sh`](run_example.sh) | One-command driver: build the runtime, start it, wait for health, run the optimizer, clean up. |

## Quick start (local container)

One script builds the runtime, starts it, waits until healthy, and runs the optimizer
against it (then cleans up the container). Run it from the repo root:

```bash
./examples/appworld/run_example.sh                 # defaults: Dockerfile.amd64, full train split, port 8080
./examples/appworld/run_example.sh -t 82e2fac_1,82e2fac_2 -p 8090   # custom tasks/port
./examples/appworld/run_example.sh --build-only    # just build the image
```

It resolves AWS credentials from the environment/role (needed at runtime for Bedrock).
See `run_example.sh --help`.

Under the hood it's just these steps (do them manually if you prefer):

```bash
# 1. Build + run the runtime container (see appworld_runtime/README.md for prerequisites).
docker build -f examples/appworld/appworld_runtime/Dockerfile.amd64 -t appworld-runtime .
docker run -d -p 8080:8080 -e AWS_REGION=us-west-2 \
  -e APPWORLD_DATASET_CSV=/app/datasets/appworld -e APPWORLD_DATASET_SPLIT=train \
  --name appworld-runtime appworld-runtime   # + AWS creds envs

# 2. Optimize its system prompt against the container.
export APPWORLD_BASE_URL="http://localhost:8080"
export APPWORLD_TASK_IDS="82e2fac_1"          # optional; else the sample list
python examples/appworld/appworld_agentcore_optimization.py
```

To drive a **deployed** AgentCore runtime instead, set `APPWORLD_AGENT_ARN` (and omit
`APPWORLD_BASE_URL`); the example auto-selects `AgentCoreRolloutEngine`.

## Two surfaces on one runtime

The same container serves both optimizers — it applies whatever the payload carries:

| surface | client | what it optimizes | payload key |
|---|---|---|---|
| system prompt | `appworld_agentcore_optimization.py` | the prompt text | `system_prompt` |
| skill library | `appworld_skill_library_optimization.py` | which skills exist (create / revise / retire) | `skills_folder` |

```bash
./examples/appworld/run_example.sh              # system prompt (default)
./examples/appworld/run_example.sh --skills     # skill library
```

Skills travel **inline** in the payload as `[{path, content}]`, not as an S3 pointer,
so no bucket and no extra IAM are needed — and a saved trace records the skill text the
agent actually read. Measured skill sets are 8–20 KB, well inside a payload. For a
library that outgrows that (or carries binary resources), switch to uploading the
materialized folder and sending a URI instead; `install_skills` in the runtime's
`_local.py` is the only place that would change.

The runtime echoes `skills_applied` in its response, and the client checks it. That is
what distinguishes *"the skills did not help"* from *"the skills never loaded"* — a
runtime built before the `skills_folder` key ignores it silently, and the resulting flat
reward looks identical to a real null result.

Reward for the curator comes from the same place as the prompt optimizer's:
`eval_result.metrics.success` — success (binary).
