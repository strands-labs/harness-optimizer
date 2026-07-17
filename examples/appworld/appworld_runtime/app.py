import os
import sys

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ============================================================================
# Configuration Section - All values configurable via environment variables
# ============================================================================

from bedrock_agentcore import BedrockAgentCoreApp
from strands_harness_optimizer.adapters import StrandsAdapter
from strands_harness_optimizer.formulas import SystemPromptFormula


logger = logging.getLogger(__name__)

# Default system prompt (can be overridden by YAML config)
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a coding agent that helps to solve AppWorld task."
)

# Dataset configuration
# Task IDs: comma-separated string or None for all tasks
TASK_IDS_STR = os.getenv("APPWORLD_TASK_IDS", None)
TASK_IDS = TASK_IDS_STR.split(",") if TASK_IDS_STR else None

# Remote APIs port for AppWorld server
REMOTE_APIS_PORT = int(os.getenv("APPWORLD_REMOTE_APIS_PORT", "9000"))

# Dataset split (e.g., "train", "test_normal", "test_challenge")
DATASET_SPLIT = os.getenv("APPWORLD_DATASET_SPLIT", "test_normal")

# Dataset split (e.g., "train", "test_normal", "test_challenge")
DATASET_CSV_PATH = os.getenv("APPWORLD_DATASET_CSV", None)

# Whether to shuffle the dataset
SHUFFLE_DATASET = os.getenv("APPWORLD_SHUFFLE", "false").lower() == "true"

# Model configuration
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0"
)

# Model backend chosen explicitly via MODEL_PROVIDER ("bedrock" | "openai").
# openai targets an OpenAI-compatible server (local vLLM); MODEL_ID then names
# the served model (e.g. "qwen3-coder-30b"). No inference/fallback.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "bedrock")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# System prompt YAML configuration path (required)
SYSTEM_PROMPT_YAML_PATH = os.getenv("SYSTEM_PROMPT_YAML_PATH", "")

# Tool consent bypass
BYPASS_TOOL_CONSENT = os.getenv("BYPASS_TOOL_CONSENT", "true")

# ============================================================================

logger.info("Agent starting")
logger.info(f"Configuration: MODEL_ID={MODEL_ID}, SPLIT={DATASET_SPLIT}, PORT={REMOTE_APIS_PORT}")
print("Agent Starting")

os.environ["BYPASS_TOOL_CONSENT"] = BYPASS_TOOL_CONSENT

from _local import create_strands_agent
from dataset import AppWorldDataset, AppWorldEnvironment

logger.info(f"Creating AppWorld dataset with tasks: {TASK_IDS}")
print(f"AppWorld Dataset: {AppWorldDataset}")
print(f"Task IDs: {TASK_IDS}")

dataset = AppWorldDataset(
    task_ids=TASK_IDS,
    remote_apis_port=REMOTE_APIS_PORT,
    split=DATASET_SPLIT,
    shuffle=SHUFFLE_DATASET,
    csv_path=DATASET_CSV_PATH
)

# The environment owns task execution/evaluation (server spawn, execute tool),
# kept separate from the dataset which is just the task rows.
environment = AppWorldEnvironment()

from omegaconf import OmegaConf

# Load system prompt from YAML if path is provided, otherwise use default
if SYSTEM_PROMPT_YAML_PATH:
    logger.info(f"Loading system prompt from: {SYSTEM_PROMPT_YAML_PATH}")
    cfg = OmegaConf.load(SYSTEM_PROMPT_YAML_PATH)
    system_prompt = cfg["system_prompt"]
else:
    logger.info("Using default system prompt")
    system_prompt = DEFAULT_SYSTEM_PROMPT

logger.info(f"Loaded system prompt: {system_prompt[:100]}...")
print("Loaded system_prompt")
print(system_prompt)

_model_cfg = {
    "model_id": MODEL_ID,
    "initial_prompt": system_prompt,
    "provider": MODEL_PROVIDER,
}
if MODEL_PROVIDER == "openai":
    # Talk to a local vLLM (OpenAI-compatible) server instead of Bedrock.
    # Required fields validated in create_strands_agent (raises if missing).
    _model_cfg.update({
        "openai_base_url": OPENAI_BASE_URL,
        "openai_api_key": OPENAI_API_KEY,
    })
    logger.info(f"Model backend: openai @ {OPENAI_BASE_URL} (model={MODEL_ID})")
elif MODEL_PROVIDER == "bedrock":
    logger.info(f"Model backend: bedrock (model={MODEL_ID})")
else:
    raise ValueError(
        f"Unknown MODEL_PROVIDER '{MODEL_PROVIDER}'. Set it to 'bedrock' or 'openai'."
    )

agent = create_strands_agent(
    _model_cfg,
    mcp_client=None,
    additional_tools=environment.get_tools(),
)

app = BedrockAgentCoreApp()

# Apply the system-prompt formula to the agent. Each invocation can override
# the prompt (see invoke() below) — this is what the optimizer tunes.
processor = SystemPromptFormula(system_prompt=system_prompt)
adapter = StrandsAdapter()
adapter.apply_to_agent([processor], agent)
app.state.currently_busy = False

logger.info("Agent started successfully")
logger.info(f"Model: {MODEL_ID}")
logger.info(f"Dataset split: {DATASET_SPLIT} with {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} tasks")
print("Agent Started")



@app.entrypoint
def invoke(payload):
    """AI agent function for AppWorld task execution"""
    logger.info("=" * 60)
    logger.info("Agent invocation start")
    print("Agent invocation start")

    processor_system_prompt = payload.get("system_prompt", system_prompt)
    processor.update_params({
        "system_prompt": processor_system_prompt
    })

    # Extract payload parameters
    task_id = payload.get("task_id")
    exp_id = payload.get("exp_id", "default")

    logger.info(f"Task ID: {task_id}")
    logger.info(f"Experiment ID: {exp_id}")

    # Get trace context for OpenTelemetry
    # span_context = trace.get_current_span().get_span_context()
    # trace_id = span_context.trace_id
    # logger.info(f"Trace ID: {trace_id}")

    # Look up the task (dataset) and run it (environment).
    data_sample = dataset.get_data_by_id(task_id)
    execute_fn = environment.get_execute_fn()

    logger.info(f"Executing task: {data_sample.instruction[:100] if hasattr(data_sample, 'instruction') else 'N/A'}...")
    response = execute_fn(agent, data_sample)

    # Evaluate the result
    eval_fn = environment.get_evaluate_fn()
    eval_result = eval_fn(data_sample, agent.messages, response)

    logger.info(f"Evaluation complete. Reward: {eval_result.get('reward', 'N/A') if hasattr(eval_result, 'get') else 'N/A'}")
    logger.info("Agent invocation complete")
    logger.info("=" * 60)

    return {
        "result": response,
        "messages": adapter.extract_context(agent)["messages"],
        "stop_reason": None,
        "eval_result": eval_result
    }

if __name__ == "__main__":
    app.run()
