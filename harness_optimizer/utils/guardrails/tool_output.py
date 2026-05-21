"""Guardrail that truncates excessive tool output.

Registers as a hook on AfterToolCallEvent. When tool output exceeds
the configured limit, it truncates and appends a warning message
guiding the agent to use targeted commands instead.
"""

import logging

from strands import Agent
from strands.hooks.events import AfterToolCallEvent

logger = logging.getLogger(__name__)


class ToolOutputGuardrail:
    """Guardrail that truncates excessive tool output and warns the agent.

    Args:
        max_chars: Maximum characters before truncation.

    Example:
        guardrail = ToolOutputGuardrail(max_chars=50000)
        guardrail.register(agent)
    """

    def __init__(self, max_chars: int = 100000):
        self.max_chars = max_chars

    def __call__(self, event: AfterToolCallEvent) -> None:
        result = event.result
        if not result or "content" not in result:
            return
        for item in result["content"]:
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
                if len(text) > self.max_chars:
                    preview = text[:self.max_chars]
                    item["text"] = (
                        f"{preview}\n\n"
                        f"[TRUNCATED: Output exceeded {self.max_chars} characters. "
                        f"Original length: {len(text)} characters. "
                        f"Use targeted commands (grep, head, jq with selectors) "
                        f"to extract specific content.]"
                    )
                    logger.warning(
                        f"Truncated tool output from {len(text)} to {self.max_chars} chars"
                    )

    def register(self, agent: Agent) -> None:
        """Register this guardrail on a strands Agent."""
        agent.add_hook(self, AfterToolCallEvent)
