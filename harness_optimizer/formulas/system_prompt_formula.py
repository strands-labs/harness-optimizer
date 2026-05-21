"""
Built-in formula for system prompt tuning.

SystemPromptFormula stores and manages a system prompt string as its
tunable parameter.
"""

import logging

from strands.hooks.events import BeforeInvocationEvent

from .formula import Formula

logger = logging.getLogger(__name__)


class SystemPromptFormula(Formula):
    """
    Formula that manages a system prompt as a tunable parameter.

    Example:
        formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")
        params = formula.get_tunable_params()
        # {'system_prompt': 'You are a helpful assistant.'}

        formula.update_params({'system_prompt': 'You are an expert coder.'})
    """

    def __init__(self, system_prompt: str = None, **kwargs):
        """
        Initialize the system prompt formula.

        Args:
            system_prompt: The system prompt string to manage.
        """
        super().__init__("system_prompt_formula", [BeforeInvocationEvent])
        self.system_prompt = system_prompt

    def process(self, context: dict, **kwargs) -> dict:
        """
        Update the system prompt in the context.

        Args:
            context: Agent context dict.

        Returns:
            Updated context with the new system prompt.
        """
        if not self.system_prompt:
            logger.warning("No system prompt set, skipping processing")
            return context

        logger.info(
            f"Updating system prompt: {self.system_prompt[:50]}..."
        )
        return {"system_prompt": self.system_prompt}

    def get_tunable_params(self) -> dict:
        """Return the current system prompt as a tunable parameter."""
        return {"system_prompt": self.system_prompt}

    def update_params(self, params: dict) -> None:
        """Update the system prompt from params dict."""
        if "system_prompt" in params:
            self.system_prompt = params["system_prompt"]
