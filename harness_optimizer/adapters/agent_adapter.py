"""
Base class for agent adapters.

An adapter bridges Formulas to a specific agent framework by translating
between the framework's native context representation and the unified
dict format that Formulas operate on.
"""

from abc import ABC, abstractmethod

from ..formulas import Formula


class AgentAdapter(ABC):
    """
    Base class for agent framework adapters.

    The adapter's core responsibility is context translation:
    - extract_context: agent's native format → unified dict
    - update_context: unified dict → agent's native format
    - apply_to_agent: attach formulas to the agent's lifecycle

    Example:
        class MyFrameworkAdapter(AgentAdapter):
            def extract_context(self, agent) -> dict:
                return {"system_prompt": agent.config["prompt"]}

            def update_context(self, agent, context: dict) -> None:
                agent.config["prompt"] = context["system_prompt"]

            def apply_to_agent(self, formulas, agent):
                for formula in formulas:
                    agent.register_middleware(formula.process)
                return agent
    """

    @abstractmethod
    def extract_context(self, agent) -> dict:
        """Extract agent context into a unified dict format.

        Args:
            agent: The agent instance (framework-specific type).

        Returns:
            Dict with standardized keys (e.g., "system_prompt", "messages").
        """
        pass

    @abstractmethod
    def update_context(self, agent, context: dict) -> None:
        """Apply a unified context dict back to the agent.

        Args:
            agent: The agent instance (framework-specific type).
            context: Dict with standardized keys to apply.
        """
        pass

    @abstractmethod
    def apply_to_agent(self, formulas: list[Formula], agent):
        """Attach formulas to the agent and return it.

        How formulas are attached depends on the framework (hooks,
        middleware, decorators, etc.).

        Args:
            formulas: List of Formula instances to attach.
            agent: The agent instance (framework-specific type).

        Returns:
            The agent with formulas applied.
        """
        pass
