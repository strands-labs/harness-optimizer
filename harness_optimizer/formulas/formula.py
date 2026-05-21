"""
Base class for Formulas.

A Formula is the core optimizable unit in Harness Optimizer. It processes agent context
(e.g., system prompts) and exposes tunable parameters that can be optimized
through rollout-based feedback.
"""

from abc import ABC, abstractmethod
from typing import Any
from strands.hooks import HookEvent


class Formula(ABC):
    """
    Base class for Formulas.

    Subclass this to define custom context processing logic and tunable parameters.

    Example:
        class MyFormula(Formula):
            def __init__(self, prompt: str):
                super().__init__("my_formula", [BeforeInvocationEvent])
                self.prompt = prompt

            def process(self, context: dict, **kwargs) -> dict:
                return {"system_prompt": self.prompt}

            def get_tunable_params(self) -> dict:
                return {"prompt": self.prompt}

            def update_params(self, params: dict) -> None:
                self.prompt = params.get("prompt", self.prompt)
    """

    def __init__(self, name: str, trigger_timings: list[str | HookEvent]):
        """
        Initialize the formula.

        Args:
            name: Identifier for this formula.
            trigger_timings: List of timing hooks when this formula should run.
                Supports strands event types (e.g., BeforeInvocationEvent) as
                first-class, or strings for other frameworks.
        """
        if not trigger_timings:
            raise ValueError("trigger_timings must not be empty — a Formula must have at least one trigger timing.")
        self.name = name
        self.trigger_timings = trigger_timings

    @abstractmethod
    def process(self, context: dict, **kwargs) -> dict:
        """
        Process the agent context and return updated context.

        Args:
            context: Agent context dict (e.g., {"system_prompt": ..., "messages": [...]}).
            **kwargs: Additional processing parameters.

        Returns:
            Updated context dict.
        """
        pass

    @abstractmethod
    def get_tunable_params(self) -> dict:
        """
        Return the current tunable parameters.

        Returns:
            Dict of parameter names to their current values.
        """
        pass

    @abstractmethod
    def update_params(self, params: dict) -> None:
        """
        Update tunable parameters.

        Args:
            params: Dict of parameter names to new values.
        """
        pass

    def can_process(self, context: dict) -> bool:
        """
        Check whether this formula should run for the given context.

        Default implementation always returns True. Override to add
        conditional logic.

        Args:
            context: Agent context dict.

        Returns:
            True if this formula should process the context.
        """
        return True
