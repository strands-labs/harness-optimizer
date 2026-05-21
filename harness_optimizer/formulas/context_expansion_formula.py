"""
Built-in formula for adding instructions to the first user message.

ContextExpansionFormula appends additional context to the first user message
in the agent's message history.
"""

import logging
from strands.hooks.events import BeforeModelCallEvent

from .formula import Formula

logger = logging.getLogger(__name__)


class ContextExpansionFormula(Formula):
    """
    Formula that adds instructions to the first user message (turn 0).

    Example:
        formula = ContextExpansionFormula(
            instruction="Consider security best practices."
        )
    """

    def __init__(self, instruction: str = None):
        """
        Initialize the add instruction formula.

        Args:
            instruction: The instruction text to append to the first user message.
        """
        super().__init__("context_expansion_formula", [BeforeModelCallEvent])
        self.instruction = instruction

    def process(self, context: dict) -> dict:
        """
        Add instruction to the first user message.

        Only applies on the first BeforeModelCallEvent (when there's exactly one user message).

        Args:
            context: Agent context dict with "messages" key.

        Returns:
            Updated context with modified first user message.
        """
        if not self.instruction:
            logger.warning("No instruction text set, skipping processing")
            return context

        messages = context.get("messages", [])
        if not messages:
            logger.warning("No messages in context, skipping processing")
            return context

        # Count user messages - only apply on first turn
        user_message_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "user"]

        if len(user_message_indices) != 1:
            logger.debug(f"Not first turn ({len(user_message_indices)} user messages), skipping")
            return context

        user_message_idx = user_message_indices[0]

        modified_messages = list(messages)
        original_msg = modified_messages[user_message_idx]
        original_content = original_msg.get("content", "")

        # Handle both string and list content formats
        if isinstance(original_content, str):
            new_content = f"{original_content}\n\n{self.instruction}"
        elif isinstance(original_content, list):
            # Append to the last text block
            new_content = list(original_content)
            for i in range(len(new_content) - 1, -1, -1):
                if isinstance(new_content[i], dict) and new_content[i].get("type") == "text":
                    new_content[i] = {
                        **new_content[i],
                        "text": f"{new_content[i].get('text', '')}\n\n{self.instruction}"
                    }
                    break
                elif isinstance(new_content[i], dict) and "text" in new_content[i]:
                    new_content[i] = {
                        **new_content[i],
                        "text": f"{new_content[i].get('text', '')}\n\n{self.instruction}"
                    }
                    break
        else:
            logger.warning(f"Unknown content format, skipping")
            return context

        logger.info(f"Adding instruction to first user message: {self.instruction[:50]}...")

        modified_messages[user_message_idx] = {
            **modified_messages[user_message_idx],
            "content": new_content
        }

        return {"messages": modified_messages}

    def get_tunable_params(self) -> dict:
        """Return the current parameters as tunable."""
        return {"instruction": self.instruction}

    def update_params(self, params: dict) -> None:
        """Update parameters from params dict."""
        if "instruction" in params:
            self.instruction = params["instruction"]

