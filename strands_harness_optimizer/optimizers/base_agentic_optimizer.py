"""
Base class for optimizers that use a strands Agent with shell tools.

Provides infrastructure for writing rollouts to temp folders, creating
strands Agents with shell tool access, output guardrails, and a
submit_optimized_params tool for reliable parameter extraction.

Formula-agnostic: it handles the parts every agentic optimizer needs (which traces
to look at, how to build and invoke the agent, how to checkpoint) and leaves the
optimization algorithm to ``step()``. Subclasses live beside it in
``optimizers/system_prompt/`` and ``optimizers/skills/``.
"""

import json
import logging
import os
import random
import shutil
import tempfile
import time
from typing import Optional

# Must precede the strands_tools import below: without it shell initializes in
# interactive-consent mode and the model refuses to run any command.
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

from botocore.config import Config as BotocoreConfig
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import shell

from ..formulas import Formula
from ..utils.guardrails import ToolOutputGuardrail
from .optimizer import FormulaOptimizer

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_SUFFIX = """

## Submitting the Optimized Parameters

You have access to a `submit_optimized_params` tool. When you have finished
analyzing the traces and crafted the optimized parameters:

1. Write each optimized parameter to a separate file using the shell tool
2. Call `submit_optimized_params(file_path_dict={"param_name": "/path/to/file", ...})` to submit

Each file must contain ONLY the parameter value, nothing else.
Do NOT include the parameters in your response text — use the tool to submit them.

Example for system prompt optimization:
```
# Write the optimized prompt to a file
echo '...' > /tmp/system_prompt.txt
# Submit it
submit_optimized_params(file_path_dict={"system_prompt": "/tmp/system_prompt.txt"})
```"""


class BaseAgenticOptimizer(FormulaOptimizer):
    """
    Base class for agent-based Formula optimizers.

    Provides:
    - Writing rollouts to a temp folder as JSON files
    - Creating a strands Agent with shell tool and output guardrail
    - A submit_optimized_params tool for reliable parameter extraction
    - Stratified sampling of traces by reward
    - Configurable boto, model, and guardrail settings

    Subclasses implement step() to define the optimization algorithm,
    using the inherited helper methods. None of the above is specific to a
    particular Formula: ContrastiveReflectionOptimizer submits parameter VALUES,
    while SkillLibraryOptimizer has its agent write a decision tree, and both
    reuse everything here. Override ``_get_extra_tools`` to add tools, and
    ``_get_tools`` to replace the default set.
    """

    _DEFAULT_MODEL_CONFIG = {
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "temperature": 1.0,
        "streaming": True,
    }

    _DEFAULT_BOTO_CONFIG = BotocoreConfig(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=256,
        read_timeout=256,
    )

    def __init__(
        self,
        formula: Formula,
        model_config: dict | None = None,
        region_name: str = "us-west-2",
        boto_config: dict | BotocoreConfig | None = None,
        n_sample_traces: int = -1,
        stratified_sampling: bool = True,
        success_threshold: float = 0.5,
        max_output_chars: int = 100000,
        system_prompt_suffix: str | None = None,
    ):
        """
        Initialize the base agentic optimizer.

        Args:
            formula: The Formula whose parameters will be optimized.
            model_config: Bedrock model configuration dict. Supported fields
                include model_id, temperature, max_tokens, top_p, streaming,
                stop_sequences, and any other BedrockModel config fields.
                Partial overrides are merged with defaults (model_id=claude-sonnet,
                temperature=1.0, streaming=True).
            region_name: AWS region for the Bedrock service.
            boto_config: Botocore config to merge with defaults. Accepts a
                BotocoreConfig instance or a dict (e.g., {"read_timeout": 300}).
                Partial overrides are supported. Uses default settings if None.
            n_sample_traces: Number of traces to sample per step (-1 for all).
            stratified_sampling: Balance successful/failed traces in sampling.
            success_threshold: Reward threshold for classifying success.
            max_output_chars: Maximum characters for tool output before truncation.
            system_prompt_suffix: Suffix appended to the agent's system prompt
                with instructions for using the submit_optimized_prompt tool.
                Uses default instructions if None.
        """
        super().__init__(formula)

        self.model_config = {**self._DEFAULT_MODEL_CONFIG, **(model_config or {})}
        self.region_name = region_name
        self.boto_config = boto_config
        self.n_sample_traces = n_sample_traces
        self.stratified_sampling = stratified_sampling
        self.success_threshold = success_threshold
        self.max_output_chars = max_output_chars
        self.system_prompt_suffix = system_prompt_suffix or DEFAULT_SYSTEM_PROMPT_SUFFIX

        self._temp_dir: Optional[str] = None
        self._submitted_params: Optional[dict] = None
        self._step_count: int = 0
        self._prompt_history: list[str] = []
        self._sampled_indices: list[int] = []

        self.last_metrics: dict = {}
        self.last_wall_clock_s: float = 0.0

    def _create_agent(self, system_prompt: str) -> Agent:
        """Create a strands Agent with shell tool, submit tool, and output guardrail."""
        if self.boto_config:
            override = (
                self.boto_config
                if isinstance(self.boto_config, BotocoreConfig)
                else BotocoreConfig(**self.boto_config)
            )
            boto_config = self._DEFAULT_BOTO_CONFIG.merge(override)
        else:
            boto_config = self._DEFAULT_BOTO_CONFIG

        model = BedrockModel(
            region_name=self.region_name,
            boto_client_config=boto_config,
            **self.model_config,
        )

        # Create submit tool that captures the params
        optimizer_ref = self

        @tool
        def submit_optimized_params(file_path_dict: dict) -> str:
            """Submit optimized parameters by providing a mapping of param names to file paths.

            Each file should contain ONLY the parameter value as plain text.

            Args:
                file_path_dict: Dict mapping parameter names to file paths.
                    e.g., {"system_prompt": "/tmp/system_prompt.txt"}
            """
            params = {}
            errors = []
            for key, path in file_path_dict.items():
                try:
                    with open(path) as f:
                        params[key] = f.read().strip()
                except FileNotFoundError:
                    errors.append(f"{key}: file not found: {path}")
                except Exception as e:
                    errors.append(f"{key}: error reading {path}: {e}")

            if errors:
                return "Errors:\n" + "\n".join(errors)

            optimizer_ref._submitted_params = params
            summary = ", ".join(f"{k} ({len(v)} chars)" for k, v in params.items())
            return f"Parameters submitted successfully: {summary}"

        # Append suffix with tool usage instructions
        full_system_prompt = system_prompt + self.system_prompt_suffix

        agent = Agent(
            system_prompt=full_system_prompt,
            model=model,
            tools=self._get_tools(submit_optimized_params) + self._get_extra_tools(),
        )

        # Register output truncation guardrail
        guardrail = ToolOutputGuardrail(max_chars=self.max_output_chars)
        guardrail.register(agent)

        return agent

    def _get_tools(self, submit_optimized_params) -> list:
        """Return the agent's BASE toolset.

        Default is ``[shell, submit_optimized_params]``. Override to replace the
        set — a subclass whose agent submits its result some other way (writing a
        folder of files, say) should drop the submit tool rather than be handed one
        that does nothing for it, since an unused tool in the schema is an
        invitation to call it.

        To ADD tools while keeping these, override ``_get_extra_tools`` instead.
        """
        return [shell, submit_optimized_params]

    def _get_extra_tools(self) -> list:
        """Return additional tools for the agent. Override in subclasses."""
        return []

    def _invoke_agent(self, agent: Agent, message: str):
        """Invoke the agent and record wall-clock + Strands metrics on self."""
        t0 = time.time()
        response = agent(message)
        self.last_wall_clock_s = time.time() - t0

        m = response.metrics
        usage = m.accumulated_usage
        self.last_metrics = {
            "cycle_count": m.cycle_count,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
            "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
            "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
            "latency_ms": m.accumulated_metrics.get("latencyMs", 0),
            "tool_calls": {k: v.call_count for k, v in m.tool_metrics.items()},
        }
        return response

    def _get_submitted_params(self) -> Optional[dict]:
        """Get the params submitted via the tool, then reset."""
        params = self._submitted_params
        self._submitted_params = None
        return params

    def _sample_traces(self) -> list[int]:
        """Sample trace indices from accumulated rollouts and rewards.

        Returns:
            List of indices into self._rollouts / self._rewards.
        """
        n_traces = len(self._rollouts)
        all_indices = list(range(n_traces))

        if self.n_sample_traces < 0 or self.n_sample_traces >= n_traces:
            self._sampled_indices = all_indices
            return all_indices

        if self.stratified_sampling:
            if not self._rewards:
                raise ValueError(
                    "Stratified sampling requires rewards but none were provided. "
                    "Call add_rewards() before step(), or set stratified_sampling=False."
                )
            sampled = self._stratified_sample(all_indices)
        else:
            sampled = random.sample(all_indices, self.n_sample_traces)

        self._sampled_indices = sampled
        logger.info(f"Sampled {len(sampled)} of {n_traces} traces")
        return sampled

    def _stratified_sample(self, all_indices: list[int]) -> list[int]:
        """Balance successful and failed traces in sampling."""
        successful, failed = [], []

        for i in all_indices:
            reward = self._rewards[i].reward
            if reward >= self.success_threshold:
                successful.append(i)
            else:
                failed.append(i)

        # Distribute budget across successful and failed
        n_each = self.n_sample_traces // 2
        n_success = min(n_each, len(successful))
        n_fail = min(self.n_sample_traces - n_success, len(failed))

        if n_success < n_each and len(failed) > n_fail:
            n_fail = min(self.n_sample_traces - n_success, len(failed))
        if n_fail < (self.n_sample_traces - n_success) and len(successful) > n_success:
            n_success = min(self.n_sample_traces - n_fail, len(successful))

        sampled = []
        if successful:
            sampled.extend(random.sample(successful, n_success))
        if failed:
            sampled.extend(random.sample(failed, n_fail))

        logger.info(f"Stratified sample: {n_success} successful + {n_fail} failed")
        return sampled

    def _write_traces_to_temp(self, indices: list[int]) -> str:
        """Write selected rollouts with rewards to a temp folder as JSON files.

        Each file carries the same episode under TWO shapes, because prompts in the
        wild read one or the other and a prompt that reads the wrong one silently
        sees nothing:

          ``data.{messages,reward,metrics,task}``
              The original nesting. The built-in contrastive_reflection templates
              read this.
          ``{reward, response.messages, eval_result, task_id}``
              The shape a deployed AgentCore runtime returns and that rollout
              traces are conventionally stored in. Prompts written against real
              trace folders read ``d["reward"]`` and
              ``d["response"]["messages"]`` -- with only the nested shape, such a
              prompt reports `reward=None` and finds zero tool calls for EVERY
              trace, so its census, its success/failure split and its per-tool
              statistics are all empty while it reports having analyzed the
              folder.

        Duplication costs a little disk in a temp folder that is deleted at the end
        of the step; a prompt silently analyzing nothing costs a whole optimizer
        run.

        Args:
            indices: List of indices into self._rollouts / self._rewards to write.

        Returns:
            Path to the temp folder containing the trace files.
        """
        self._temp_dir = tempfile.mkdtemp(prefix="contrastive_traces_")

        for i in indices:
            rollout = self._rollouts[i]
            reward_obj = self._rewards[i] if i < len(self._rewards) else None
            reward_value = reward_obj.reward if reward_obj else 0
            metrics = (
                {"reward": reward_value, "metadata": reward_obj.metadata} if reward_obj else {}
            )
            data = {
                # Legacy nesting -- the built-in contrastive_reflection templates.
                "data": {
                    "messages": rollout.messages,
                    "reward": reward_value,
                    "metrics": metrics,
                    "task": rollout.data_sample,
                },
                # invoke.py / AgentCore-runtime convention.
                "reward": reward_value,
                "response": {"messages": rollout.messages},
                "eval_result": {"reward": reward_value},
                "task_id": rollout.data_sample.get("task_id", f"rollout_{i:04d}"),
                "metrics": metrics,
            }

            path = os.path.join(self._temp_dir, f"{self._trace_filename(rollout, i)}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        logger.info(f"Wrote {len(indices)} traces to {self._temp_dir}")
        return self._temp_dir

    @staticmethod
    def _trace_filename(rollout, index: int) -> str:
        """Filesystem-safe name for one trace, preserving a task_id if present."""
        raw = str(rollout.data_sample.get("task_id") or "").strip()
        if not raw:
            return f"rollout_{index:04d}"
        safe = raw.replace("/", "_").replace("\\", "_").replace("\0", "")
        # Keep it short enough for any filesystem, and unique regardless of the id.
        return f"{safe[:180]}_{index:04d}"

    def _cleanup_temp(self) -> None:
        """Clean up temporary folders."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up {self._temp_dir}: {e}")
            finally:
                self._temp_dir = None

    def get_state(self) -> dict:
        """Get optimizer state for checkpointing.

        Returns:
            Dictionary containing optimizer state. Subclasses should call
            super().get_state() and merge their own state.

        Note:
            boto_config is not saved if it was provided as a BotocoreConfig
            instance (not JSON-serializable). Pass boto_config as a dict
            for full checkpoint support.
        """
        # Convert boto_config to dict if possible
        boto_config_dict = None
        if isinstance(self.boto_config, dict):
            boto_config_dict = self.boto_config
        elif self.boto_config is not None:
            logger.warning(
                "boto_config is a BotocoreConfig instance and cannot be serialized. "
                "Pass boto_config as a dict for full checkpoint support."
            )

        return {
            # Config
            "model_config": dict(self.model_config),
            "region_name": self.region_name,
            "boto_config": boto_config_dict,
            "n_sample_traces": self.n_sample_traces,
            "stratified_sampling": self.stratified_sampling,
            "success_threshold": self.success_threshold,
            "max_output_chars": self.max_output_chars,
            "system_prompt_suffix": self.system_prompt_suffix,
            # State
            "step_count": self._step_count,
            "prompt_history": list(self._prompt_history),
            "sampled_indices": list(self._sampled_indices),
        }

    def load_state(self, state: dict) -> None:
        """Load optimizer state from checkpoint.

        Args:
            state: Dictionary from get_state(). Subclasses should call
                super().load_state(state) and restore their own state.
        """
        # Config
        if "model_config" in state:
            self.model_config = dict(state["model_config"])
        if "region_name" in state:
            self.region_name = state["region_name"]
        if "boto_config" in state and state["boto_config"] is not None:
            self.boto_config = state["boto_config"]
        if "n_sample_traces" in state:
            self.n_sample_traces = state["n_sample_traces"]
        if "stratified_sampling" in state:
            self.stratified_sampling = state["stratified_sampling"]
        if "success_threshold" in state:
            self.success_threshold = state["success_threshold"]
        if "max_output_chars" in state:
            self.max_output_chars = state["max_output_chars"]
        if "system_prompt_suffix" in state:
            self.system_prompt_suffix = state["system_prompt_suffix"]
        # State
        if "step_count" in state:
            self._step_count = state["step_count"]
        if "prompt_history" in state:
            self._prompt_history = list(state["prompt_history"])
        if "sampled_indices" in state:
            self._sampled_indices = list(state["sampled_indices"])
