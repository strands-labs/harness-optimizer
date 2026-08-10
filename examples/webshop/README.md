# WebShop optimization example (multi-agent optimizer)

Optimize an agent's system prompt on [WebShop](https://webshop-pnlp.github.io) shopping
tasks with `strands_harness_optimizer`. WebShop tasks run inside a deployable **AgentCore
runtime** (a Strands agent that shops against the WebShop gym via `search` / `click` /
`buy now`); the **optimizer** drives that runtime over HTTP (local container) or by ARN
(deployed), scores each task by WebShop's match score, and rewrites the system prompt.

Unlike the [AppWorld example](../appworld) (single-agent reflection), this one uses the
**`MultiAgentOptimizer`**: the reflection step spins up a **swarm of sub-agents** to
analyze the rollouts in parallel before rewriting the prompt (see "Sub-agents" below).

```
webshop_agentcore_optimization.py          webshop_runtime/  (the deployable runtime)
  DataLoader(task_ids)                        app.py  — AgentCore /invocations entrypoint
  AgentCoreHTTPRolloutEngine ──HTTP POST──▶      → WebShopDataset runs the task,
  (or AgentCoreRolloutEngine, by ARN)              returns {messages, eval_result}
        │  payload_mapper: {data_sample,params}
        ▼            → {task_id, system_prompt, exp_id}
  WebShopReward (eval_result.metrics.score)
        │
  MultiAgentOptimizer.step()   ← swarm of sub-agents analyzes rollouts in parallel
        → SystemPromptFormula.update_params()
```

## Sub-agents

The optimizer here is `MultiAgentOptimizer`. Its reflection step is an **orchestrator**
agent that uses the strands [`swarm`](https://strandsagents.com/latest/user-guide/concepts/multi-agent/agents-as-tools/)
tool to create a team of **sub-agents**, each assigned a subset of the WebShop rollout
traces. Every sub-agent (role: the `multi_agent/rollout_analyzer` template) extracts
contrastive success/failure signals from its traces; the orchestrator aggregates those
findings and appends them to the system prompt. This mirrors the internal
`webshop_swarm_*` optimization configs. To use single-agent reflection instead, swap
`MultiAgentOptimizer` for `ContrastiveReflectionOptimizer` (as the AppWorld example does).

## Two pieces

| Path | What it is |
|------|-----------|
| [`webshop_agentcore_optimization.py`](webshop_agentcore_optimization.py) | The **optimizer** (client). Runs the Trainer loop against the runtime with the multi-agent optimizer. Needs only `strands_harness_optimizer` + a runtime endpoint. |
| [`webshop_runtime/`](webshop_runtime/) | The **deployable runtime** (container). Builds a Strands + WebShop-gym agent as an AgentCore runtime. See its [README](webshop_runtime/README.md). |
| [`webshop_skill_library_optimization.py`](webshop_skill_library_optimization.py) | The **skill-library** optimizer (client). Curates the SET of skills the agent has, instead of tuning the system prompt. |
| [`run_example.sh`](run_example.sh) | One-command driver: build the runtime, start it, wait for health, run the optimizer, clean up. |

## Quick start (local container)

One script builds the runtime, starts it, waits until healthy, and runs the optimizer
against it (then cleans up the container). Run it from the repo root:

```bash
./examples/webshop/run_example.sh                    # defaults: Dockerfile.amd64, train split, port 8080
./examples/webshop/run_example.sh -t 0,1,2 -p 8090   # custom tasks/port
./examples/webshop/run_example.sh --build-only       # just build the image
```

It resolves AWS credentials from the environment/role (needed at runtime for Bedrock).
See `run_example.sh --help`.

Under the hood it's just these steps (do them manually if you prefer):

```bash
# 1. Build + run the runtime container (see webshop_runtime/README.md for prerequisites).
docker build -f examples/webshop/webshop_runtime/Dockerfile.amd64 -t webshop-runtime .
docker run -d -p 8080:8080 -e AWS_REGION=us-west-2 \
  -e WEBSHOP_DATASET_SPLIT=train \
  --name webshop-runtime webshop-runtime   # + AWS creds envs

# 2. Optimize its system prompt against the container (multi-agent optimizer).
export WEBSHOP_BASE_URL="http://localhost:8080"
export WEBSHOP_TASK_IDS="0,1,2"     # optional subset; else the whole 'train' range (0-99)
python examples/webshop/webshop_agentcore_optimization.py
```

To drive a **deployed** AgentCore runtime instead, set `WEBSHOP_AGENT_ARN` (and omit
`WEBSHOP_BASE_URL`); the example auto-selects `AgentCoreRolloutEngine`.

## Tasks & splits

WebShop tasks are integer goal indices. A split is a contiguous range:

- `train` → indices `0..99`
- `eval`  → indices `0..199` (the held-out goals)

Set `WEBSHOP_TASK_IDS` for a quick subset, or `WEBSHOP_SPLIT` to train on the whole
range. The default dataset in the runtime is WebShop's small **1000-product** set.

## Two surfaces on one runtime

The same container serves both optimizers — it applies whatever the payload carries:

| surface | client | what it optimizes | payload key |
|---|---|---|---|
| system prompt | `webshop_agentcore_optimization.py` | the prompt text | `system_prompt` |
| skill library | `webshop_skill_library_optimization.py` | which skills exist (create / revise / retire) | `skills_folder` |

```bash
./examples/webshop/run_example.sh              # system prompt (default)
./examples/webshop/run_example.sh --skills     # skill library
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
`eval_result.metrics.score` — match score in [0,1] (partial credit).
