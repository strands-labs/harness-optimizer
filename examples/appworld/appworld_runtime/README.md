# AppWorld AgentCore runtime

This directory contains the **deployable runtime** for the AppWorld optimization
example. It packages a Strands agent that solves [AppWorld](https://appworld.dev)
tasks and exposes it as a [Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)
runtime. Optimize its system prompt from the OSS harness with
[`../appworld_agentcore_optimization.py`](../appworld_agentcore_optimization.py), which
drives this runtime over HTTP (local container, `AgentCoreHTTPRolloutEngine`) or by ARN
(deployed, `AgentCoreRolloutEngine`).

```
optimizer (this repo)  ──InvokeAgentRuntime──▶  AppWorld runtime (this dir)
   AgentCoreRolloutEngine                          app.py  →  AppWorldDataset
   ContrastiveReflectionOptimizer                          →  appworld env + eval
        ▲   tuned system_prompt in payload                  │
        └───────────── eval_result (success) ───────────────┘
```

## Layout

This runtime depends only on **`strands_harness_optimizer`** + **`strands`** + the
**`appworld`** package — there is no `agent_customizer` dependency. The few helpers it
would otherwise import from that internal package are reproduced locally in `_local.py`.

| Path | What it is |
|------|-----------|
| `app.py` | AgentCore entrypoint. `invoke(payload)` reads `task_id` / `system_prompt` / `exp_id`, runs the task, returns `{result, messages, stop_reason, eval_result}`. Uses `SystemPromptFormula` + `StrandsAdapter` from the OSS package. |
| `_local.py` | Self-contained helpers (no `agent_customizer`): `make_eval_result`, `ManagedProcess`, `create_strands_agent`. |
| `dataset/` | `AppWorldDataset` (`dataset.py`, on top of `strands_harness_optimizer.data.Dataset` — task rows only), `AppWorldEnvironment` (`environment.py` — runs/evaluates a task, owns the env server + `execute` tool), plus `appworld_strands_tool.py` / `appworld_utils.py`. |
| `generate_csvs.py` | Builds the task CSVs from the downloaded AppWorld data (run at image build). |
| `Dockerfile*` | Build variants (see below). |
| `*_requirements.txt` | Python deps per virtualenv. |

## Runtime I/O contract

Request payload (what `AgentCoreRolloutEngine` + the example's `payload_mapper` send):
```json
{"task_id": "<appworld task id>", "system_prompt": "<prompt to run>", "exp_id": "<label>"}
```
Response payload:
```json
{"result": ..., "messages": [...], "stop_reason": null,
 "eval_result": {"reward": 0.0, "metrics": {"success": false, "difficulty": ...,
                 "num_tests": ..., "passes": ..., "failures": ...}}}
```
The optimizer's `AppWorldReward` reads `eval_result.metrics.success`.

## Dockerfile variants

| File | Target | Notes |
|------|--------|-------|
| `Dockerfile.amd64` | linux/amd64 | **Fully public** (no ECR/S3): `python:3.12-slim` base, uv from PyPI, appworld source from public GitHub. **Dual venv** — `aux` runs the appworld **0.1.3** env server (pydantic v1), `main` runs strands + appworld 0.2.x client (pydantic v2). Bedrock backend. |
| `Dockerfile.arm64` | linux/arm64 | Same fully-public dual-venv build for arm64 (what AgentCore runs). Differs from `.amd64` only in the `--platform` pin. |

## Build-time patches to AppWorld

`Dockerfile.arm64`/`Dockerfile.amd64` build appworld from the pinned upstream source (commit
`11e8d183`, the 0.2.x line) and apply two **one-line `sed` edits** to it — each is the
only change vs upstream, so we edit in place rather than shipping full-file overrides
(which would drift with the upstream files):

```dockerfile
# pin the on-disk data version to the bundle this image downloads (0.2.0 -> 0.1.0)
RUN sed -i 's/^DATA_VERSION = .*/DATA_VERSION = "0.1.0"/' ./appworld/src/appworld/common/constants.py
# drop the .dev0 suffix so the built wheel has a release version
RUN sed -i 's/^version = "0.2.0.dev0"/version = "0.2.0"/' ./appworld/pyproject.toml
```

A third patch (`compat.py`, a botocore override forcing real wall-clock time for SigV4
signing under AppWorld's frozen clock) was **removed**: the AppWorld environment runs in
a separate server process (`appworld serve environment` + `AppWorld(remote_environment_url=…)`),
so freezegun freezes the *server's* clock, not the agent process where boto signs. Only
needed if you create the world without a `remote_environment_url` in boto's own process;
see git history for the old file.

## ⚠️ Build prerequisites

Both Dockerfiles are **fully public** — no private ECR/S3. They use the public
`python:3.12-slim` base, install `strands_harness_optimizer` from **this repo** (copied
into the image, not PyPI, so the runtime tracks the current working code), and fetch the
appworld source by cloning `github.com/stonybrooknlp/appworld` at commit `11e8d1832...`
with `git lfs pull`. Remaining build-time needs are just network access:

1. **AppWorld package + data** — each installs `appworld` and runs `appworld install` +
   `appworld download data` (needs network + git-lfs). The env server runs appworld
   0.1.3 (pydantic v1) in `VENV_AUX`; the agent side runs appworld 0.2.x in `VENV_MAIN`.
2. **AppWorld task CSVs** are generated at build time by `generate_csvs.py` from the
   downloaded data — not committed to the repo (`datasets/appworld/` is gitignored).

## Build & deploy (sketch)

```bash
# From this repo root. All variants are fully public (no ECR/S3). AgentCore runs arm64:
docker build -f examples/appworld/appworld_runtime/Dockerfile.arm64 -t appworld-runtime .

# Push to ECR and register as an AgentCore runtime, then grab the runtime ARN.
# (Use your standard AgentCore deployment flow.)

# Optimize its system prompt from this repo:
export APPWORLD_AGENT_ARN="arn:aws:bedrock-agentcore:us-west-2:<acct>:runtime/<id>"
export APPWORLD_TASK_IDS="<task_id_1>,<task_id_2>,..."
python examples/appworld/appworld_agentcore_optimization.py
```

## Note on imports

`app.py` runs as a top-level module (`python -m app` from this directory, per the
Dockerfiles' `WORKDIR /app`), so it imports its siblings directly: `from _local import
create_strands_agent` and `from dataset import AppWorldDataset`. If you relocate these
files, keep them importable from the working directory (or turn the folder into a
package and switch to relative imports).
