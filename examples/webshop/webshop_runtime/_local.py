"""Self-contained helpers for the AppWorld runtime example.

These are the few small utilities the runtime needs that would otherwise come
from the internal ``agent_customizer`` package. They are reproduced here (with
no ``agent_customizer`` dependency) so this example depends only on
``strands_harness_optimizer`` + ``strands`` + the ``appworld`` package:

- ``make_eval_result`` — the plain-dict eval-result shape the reward reads.
- ``ManagedProcess``   — start/stop a subprocess (the AppWorld env server),
  waiting for a readiness line and draining stdout so its pipe never fills.
- ``create_strands_agent`` — build a Strands ``Agent`` (Bedrock or an
  OpenAI-compatible endpoint) with MCP + extra tools and a bounded window.
"""

import logging
import os
import select
import signal
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --- eval result --------------------------------------------------------------

def make_eval_result(
    reward: float,
    turn_rewards=None,
    metrics=None,
    messages=None,
    system_prompt: str = "",
    elapsed_time: int = 0,
    session_id: str = None,
) -> dict:
    """Build a rollout-evaluation result as a plain dict with all keys present."""
    return {
        "reward": reward,
        "turn_rewards": turn_rewards if turn_rewards is not None else {},
        "metrics": metrics if metrics is not None else {},
        "messages": messages if messages is not None else [],
        "system_prompt": system_prompt,
        "elapsed_time": elapsed_time,
        "session_id": session_id,
    }


# --- managed subprocess (AppWorld environment server) -------------------------

class ManagedProcess:
    """Start a subprocess, wait for a readiness line, drain its stdout."""

    def __init__(self, cmd, ready_pattern="SERVER READY", startup_timeout=10.0):
        self.cmd = list(cmd)
        self.ready_pattern = ready_pattern
        self.startup_timeout = startup_timeout
        self.proc: Optional[subprocess.Popen] = None
        self._drain_thread: Optional[threading.Thread] = None

    def start(self):
        if self.is_running():
            return
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            preexec_fn=os.setsid,
        )
        if not self._wait_until_ready():
            self.stop()
            raise RuntimeError("ManagedProcess failed to start correctly")
        # After readiness we stop reading stdout, but the child keeps writing to
        # it (the AppWorld env server logs every request). A PIPE has a bounded
        # OS buffer (~64KB); once full the child blocks on write() and stops
        # serving. Drain continuously in a daemon thread to keep it unblocked.
        self._drain_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._drain_thread.start()

    def _drain_stdout(self):
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        try:
            for _ in proc.stdout:
                pass  # discard; only need to keep the buffer drained
        except (ValueError, OSError):
            pass  # stdout closed on stop() — expected during shutdown

    def _wait_until_ready(self) -> bool:
        assert self.proc is not None
        start_time = time.time()
        stdout = self.proc.stdout
        while True:
            if time.time() - start_time > self.startup_timeout:
                logger.warning("Startup timeout waiting for ready signal")
                return False
            if self.proc.poll() is not None:
                logger.warning("Process exited early with code %s", self.proc.returncode)
                if stdout:
                    leftover = stdout.read()
                    if leftover:
                        logger.warning("Process output before exit:\n%s", leftover)
                return False
            if stdout is None:
                return True
            rlist, _, _ = select.select([stdout], [], [], 0.1)
            if not rlist:
                continue
            line = stdout.readline()
            if not line:
                continue
            if self.ready_pattern in line:
                return True

    def stop(self):
        if not self.is_running():
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        finally:
            self.proc = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


# --- Strands agent builder ----------------------------------------------------

def create_strands_agent(
    model_config: Dict[str, Any],
    mcp_client: Optional[Any] = None,
    additional_tools: Optional[List] = None,
) -> Any:
    """Create a Strands Agent (bedrock | openai) with MCP + extra tools.

    Provider is chosen explicitly by ``model_config['provider']`` — no inference
    and no fallback between providers; missing required fields raise.
    """
    from strands import Agent
    from strands.agent.conversation_manager import SlidingWindowConversationManager

    provider = model_config.get("provider", "bedrock")

    if provider == "openai":
        if not model_config.get("openai_base_url"):
            raise ValueError("provider='openai' requires 'openai_base_url' in model_config")
        if not model_config.get("model_id"):
            raise ValueError("provider='openai' requires 'model_id' in model_config")
        if not model_config.get("openai_api_key"):
            raise ValueError(
                "provider='openai' requires 'openai_api_key' in model_config "
                "(use a placeholder like 'EMPTY' for servers that ignore it)"
            )
        from strands.models.openai import OpenAIModel
        model = OpenAIModel(
            client_args={
                "base_url": model_config["openai_base_url"],
                "api_key": model_config["openai_api_key"],
            },
            model_id=model_config["model_id"],
            params={
                "temperature": model_config.get("temperature", 0.0),
                "max_tokens": model_config.get("max_tokens", 4096),
            },
        )
        logger.info("Using OpenAI-compatible model '%s' at %s",
                    model_config["model_id"], model_config["openai_base_url"])
    elif provider == "bedrock":
        if not model_config.get("model_id"):
            raise ValueError("provider='bedrock' requires 'model_id' in model_config")
        from botocore.config import Config as BotocoreConfig
        from strands.models import BedrockModel

        additional_request_fields = {}
        if model_config.get("enable_thinking", True):
            budget = model_config.get("thinking_budget_tokens", 2048)
            additional_request_fields["thinking"] = {"type": "enabled", "budget_tokens": budget}

        model = BedrockModel(
            model_id=model_config["model_id"],
            region_name=model_config.get("region", "us-west-2"),
            boto_client_config=BotocoreConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
            streaming=model_config.get("streaming", False),
            additional_request_fields=additional_request_fields,
        )
    else:
        raise ValueError(
            f"Unknown model provider '{provider}'. Set model_config['provider'] "
            "to 'bedrock' or 'openai'."
        )

    tools = []
    if mcp_client:
        try:
            with mcp_client:
                mcp_tools = mcp_client.list_tools_sync()
                tools.extend(mcp_tools)
                logger.info("Added %d MCP tools", len(mcp_tools))
        except Exception as e:
            logger.warning("Failed to get MCP tools: %s", e)
    if additional_tools:
        tools.extend(additional_tools)
        logger.info("Added %d additional tools", len(additional_tools))

    conversation_manager = SlidingWindowConversationManager(
        window_size=model_config.get("conversation_window_size", 40),
        should_truncate_results=True,
    )
    agent = Agent(
        model=model,
        system_prompt=model_config.get("initial_prompt", ""),
        tools=tools,
        conversation_manager=conversation_manager,
    )
    logger.info("Created Strands agent with %d total tools", len(tools))
    return agent
