#!/usr/bin/env bash
#
# Build the AppWorld runtime container, start it, wait until healthy, and run the
# system-prompt optimizer against it (the local-container / HTTP-engine path).
#
# Usage:
#   ./run_example.sh [-f DOCKERFILE] [-s SPLIT] [-t TASK_IDS] [-p PORT] [--skills] [--build-only] [--no-build]
#
#   -f  Dockerfile to build (default: appworld_runtime/Dockerfile.amd64 — fully public,
#       Bedrock, dual venv). Use Dockerfile.arm64 for the arm64 build AgentCore runs.
#   -s  Split to train on: train / dev / test_normal / test_challenge (default: train).
#       The full split is used — extracted as a CSV from the runtime container.
#   -t  Comma-separated AppWorld task ids to train on a SUBSET instead of the full split.
#   -p  Host port to publish the runtime on (default: 8080).
#   --skills       Optimize the SKILL LIBRARY instead of the system prompt
#                  (runs *_skill_library_optimization.py; skills ship inline in the payload).
#   --build-only   Build the image and exit.
#   --no-build     Skip building; assume the image already exists.
#
# This is the single setup+run script: it installs the client-side dependency
# (strands_harness_optimizer, editable from this repo) if missing, builds the runtime
# image, starts the container, waits for health, extracts the task split, runs the
# optimizer, and cleans up.
#
# Requires: docker, python3, and AWS credentials in the environment/instance role (for
# Bedrock at runtime, and for S3 at build time with the .amd64/.arm64 Dockerfiles).
#
# Run from the repository root (build context = repo root, per the Dockerfile COPY paths).

set -euo pipefail

# --- defaults ---
DOCKERFILE="examples/appworld/appworld_runtime/Dockerfile.amd64"
TASK_IDS=""          # empty => train on the FULL split (extracted from the container)
SPLIT="train"
PORT="8080"
IMAGE="appworld-runtime"
CONTAINER="appworld-runtime"
BUILD=1
RUN_OPTIMIZER=1
SURFACE="system_prompt"   # --skills switches to the skill-library client
AWS_REGION="${AWS_REGION:-us-west-2}"

# --- args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f) DOCKERFILE="$2"; shift 2 ;;
    -t) TASK_IDS="$2"; shift 2 ;;      # subset override (comma-separated); else full split
    -s) SPLIT="$2"; shift 2 ;;         # which split to train on (train/dev/test_normal/...)
    -p) PORT="$2"; shift 2 ;;
    --skills) SURFACE="skills"; shift ;;
    --build-only) RUN_OPTIMIZER=0; shift ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- must run from repo root (Dockerfile COPYs are repo-root relative) ---
if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: $DOCKERFILE not found. Run this script from the repository root." >&2
  exit 1
fi

# --- preflight: required tools + client-side Python dependency ---
for tool in docker python3; do
  command -v "$tool" >/dev/null || { echo "ERROR: '$tool' is required but not found." >&2; exit 1; }
done
if [[ "$RUN_OPTIMIZER" == "1" ]]; then
  # The optimizer client needs strands_harness_optimizer. Install THIS repo (editable)
  # if it isn't importable, so the client matches the code the container runs.
  if ! python3 -c "import strands_harness_optimizer" >/dev/null 2>&1; then
    echo ">> Installing strands_harness_optimizer (editable) from this repo ..."
    python3 -m pip install -e . >/dev/null || {
      echo "ERROR: failed to install strands_harness_optimizer. Install it manually:" >&2
      echo "       python3 -m pip install -e ." >&2
      exit 1
    }
  fi
fi

# --- resolve AWS credentials (needed for Bedrock at runtime; S3 at build for .amd64/.arm64) ---
# Prefer explicit env vars; otherwise resolve the active chain via boto3 (works with
# instance roles / profiles and both AWS CLI v1 and v2).
AK="${AWS_ACCESS_KEY_ID:-}"; SK="${AWS_SECRET_ACCESS_KEY:-}"; TK="${AWS_SESSION_TOKEN:-}"
if [[ -z "$AK" || -z "$SK" ]]; then
  creds=$(python3 - <<'PY' 2>/dev/null || true
import boto3
c = boto3.Session().get_credentials()
if c:
    f = c.get_frozen_credentials()
    print(f"{f.access_key}\t{f.secret_key}\t{f.token or ''}")
PY
)
  if [[ -n "$creds" ]]; then
    AK=$(printf '%s' "$creds" | cut -f1)
    SK=$(printf '%s' "$creds" | cut -f2)
    TK=$(printf '%s' "$creds" | cut -f3)
  fi
fi
if [[ -z "$AK" || -z "$SK" ]]; then
  echo "ERROR: no AWS credentials found (need them for Bedrock + S3-based builds)." >&2
  echo "       Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or configure a profile/role." >&2
  exit 1
fi

# --- build ---
if [[ "$BUILD" == "1" ]]; then
  echo ">> Building $IMAGE from $DOCKERFILE ..."
  DOCKER_BUILDKIT=1 docker build -f "$DOCKERFILE" \
    --build-arg AWS_ACCESS_KEY_ID="$AK" \
    --build-arg AWS_SECRET_ACCESS_KEY="$SK" \
    --build-arg AWS_SESSION_TOKEN="$TK" \
    -t "$IMAGE" .
fi
[[ "$RUN_OPTIMIZER" == "0" ]] && { echo ">> Built $IMAGE. Exiting (--build-only)."; exit 0; }

# --- run the container ---
echo ">> Starting container $CONTAINER on port $PORT ..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p "$PORT:8080" \
  -e AWS_ACCESS_KEY_ID="$AK" -e AWS_SECRET_ACCESS_KEY="$SK" -e AWS_SESSION_TOKEN="$TK" \
  -e AWS_REGION="$AWS_REGION" \
  -e APPWORLD_DATASET_CSV=/app/datasets/appworld \
  -e APPWORLD_DATASET_SPLIT="$SPLIT" \
  -e BYPASS_TOOL_CONSENT=true \
  "$IMAGE" >/dev/null

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# --- wait for health ---
echo -n ">> Waiting for /ping "
for _ in $(seq 1 60); do
  code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://localhost:$PORT/ping" 2>/dev/null || echo 000)
  [[ "$code" == "200" ]] && { echo " healthy."; break; }
  echo -n "."; sleep 2
done
if [[ "${code:-000}" != "200" ]]; then
  echo " runtime did not become healthy; last container logs:" >&2
  docker logs --tail 30 "$CONTAINER" >&2
  exit 1
fi

# --- run the optimizer against the container ---
echo ">> Running the optimizer against http://localhost:$PORT ..."
export APPWORLD_BASE_URL="http://localhost:$PORT"

if [[ -n "$TASK_IDS" ]]; then
  # explicit subset
  export APPWORLD_TASK_IDS="$TASK_IDS"
  echo ">> Training on task subset: $TASK_IDS"
else
  # full split: extract the split CSV the runtime generated at build time
  mkdir -p datasets/appworld
  csv_out="datasets/appworld/${SPLIT}.csv"
  docker cp "$CONTAINER:/app/datasets/appworld/${SPLIT}.csv" "$csv_out"
  n=$(( $(wc -l < "$csv_out") - 1 ))
  export APPWORLD_TASK_CSV="$csv_out"
  unset APPWORLD_TASK_IDS
  echo ">> Training on the full '$SPLIT' split ($n tasks) from $csv_out"
fi

if [[ "$SURFACE" == "skills" ]]; then
  echo ">> Optimizing the SKILL LIBRARY (AppWorld)"
  python3 examples/appworld/appworld_skill_library_optimization.py
else
  echo ">> Optimizing the SYSTEM PROMPT (AppWorld)"
  python3 examples/appworld/appworld_agentcore_optimization.py
fi

echo ">> Done."
