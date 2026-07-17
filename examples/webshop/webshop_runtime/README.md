# WebShop AgentCore runtime

This directory contains the **deployable runtime** for the WebShop optimization
example. It packages a Strands agent that solves [WebShop](https://webshop-pnlp.github.io)
shopping tasks and exposes it as a [Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)
runtime. Optimize its system prompt from the OSS harness with
[`../webshop_agentcore_optimization.py`](../webshop_agentcore_optimization.py), which
drives this runtime over HTTP (local container, `AgentCoreHTTPRolloutEngine`) or by ARN
(deployed, `AgentCoreRolloutEngine`).

```
optimizer (this repo)  ──InvokeAgentRuntime──▶  WebShop runtime (this dir)
   AgentCoreRolloutEngine                          app.py  →  WebShopDataset
   MultiAgentOptimizer (swarm)                             →  WebShop gym + eval
        ▲   tuned system_prompt in payload                  │
        └───────────── eval_result (score) ─────────────────┘
```

## Layout

This runtime depends only on **`strands_harness_optimizer`** + **`strands`** + the
public **WebShop** source — there is no `agent_customizer` dependency. The few helpers
it would otherwise import from that internal package are reproduced locally in
`_local.py`.

| Path | What it is |
|------|-----------|
| `app.py` | AgentCore entrypoint. `invoke(payload)` reads `task_id` / `system_prompt` / `exp_id`, runs the task, returns `{result, messages, stop_reason, eval_result}`. Uses `SystemPromptFormula` + `StrandsAdapter` from the OSS package. |
| `_local.py` | Self-contained helpers (no `agent_customizer`): `make_eval_result`, `ManagedProcess`, `create_strands_agent`. |
| `dataset/` | `WebShopDataset` (`dataset.py`, on top of `strands_harness_optimizer.data.Dataset` — task rows only), `WebShopEnvironment` (`environment.py` — runs/evaluates a task, owns the gym server + `search`/`click`/`get_available_actions` tools). |
| `app_simple.py` | The WebShop gym server: a Flask wrapper over `WebAgentTextEnv` exposing `/reset`, `/step`, `/get_actions`, `/close`. Runs in `VENV_AUX`. |
| `build_index.py` | Builds the Lucene search-index documents (run at image build). |
| `prompts/webshop_instructions.yaml` | The baseline system prompt (tool protocol + shopping strategy) the optimizer starts from. |
| `Dockerfile*` | Build variants (see below). |
| `*_requirements.txt` | Python deps per virtualenv. |

## Runtime I/O contract

Request payload (what the engine + the example's `payload_mapper` send):
```json
{"task_id": "<goal index>", "system_prompt": "<prompt to run>", "exp_id": "<label>"}
```
Response payload:
```json
{"result": ..., "messages": [...], "stop_reason": null,
 "eval_result": {"reward": 0.0, "metrics": {"success": false, "score": 0.0, "done": true}}}
```
The optimizer's `WebShopReward` reads `eval_result.metrics.score` (WebShop's [0,1]
partial-match score; `success` = a full 1.0 match).

## Dockerfile variants

| File | Target | Notes |
|------|--------|-------|
| `Dockerfile.amd64` | linux/amd64 | **Fully public** (no ECR/S3): `python:3.12-slim` base, uv from PyPI, WebShop source + small (1000-product) data from the public `princeton-nlp/webshop` repo. **Dual venv** — `aux` (Python 3.9) runs the WebShop gym (pinned ML stack: pydantic v1, numpy<2, Flask 2.1, pyserini/Lucene), `main` (3.12) runs strands + the runtime. `openjdk` backs pyserini's Lucene. Bedrock backend. |
| `Dockerfile.arm64` | linux/arm64 | Same fully-public dual-venv build for arm64 (what AgentCore runs). Differs from `.amd64` only in the `--platform` pin and the `JAVA_HOME` arch path. |

## Build prerequisites

Both Dockerfiles are **fully public** — no private ECR/S3. They:

1. **WebShop source + data** — clone `github.com/princeton-nlp/webshop`, download the
   small **1000-product** dataset from its public Google Drive mirror (the ids the
   upstream `setup.sh -d small` uses), and build the Lucene index (`indexes_1k`) with
   pyserini at build time. Needs network access and `openjdk`.
2. **`strands_harness_optimizer`** is installed from **this repo** (copied into the
   image, not PyPI), so the runtime tracks the current working code.

The default is the small 1000-product set (fast, public). To use a larger set, override
`WEBSHOP_NUM_PRODUCTS` / `WEBSHOP_DATA_FILE` and build the matching index (see WebShop's
`search_engine/run_indexing.sh`).

## Build & deploy (sketch)

```bash
# From this repo root. All variants are fully public. AgentCore runs arm64:
docker build -f examples/webshop/webshop_runtime/Dockerfile.arm64 -t webshop-runtime .

# Push to ECR and register as an AgentCore runtime, then grab the runtime ARN.
# (Use your standard AgentCore deployment flow.)

# Optimize its system prompt from this repo (multi-agent / swarm optimizer):
export WEBSHOP_AGENT_ARN="arn:aws:bedrock-agentcore:us-west-2:<acct>:runtime/<id>"
export WEBSHOP_TASK_IDS="0,1,2,3"       # optional subset; else the whole split range
python examples/webshop/webshop_agentcore_optimization.py
```

## Note on imports

`app.py` runs as a top-level module (`python -m app` from this directory, per the
Dockerfiles' `WORKDIR /app`), so it imports its siblings directly: `from _local import
create_strands_agent` and `from dataset import WebShopDataset`. If you relocate these
files, keep them importable from the working directory (or turn the folder into a
package and switch to relative imports).
