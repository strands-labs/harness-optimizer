"""Strands agent adapter — connect formulas to strands-agents.

Example:
    from strands import Agent
    from strands_harness_optimizer.formulas import SystemPromptFormula
    from strands_harness_optimizer.adapters.strands_adapter import StrandsAdapter

    adapter = StrandsAdapter()
    formula = SystemPromptFormula(system_prompt="You are an expert coder.")
    agent = Agent(model=model)
    adapter.apply_to_agent([formula], agent)

    # Or use the convenience function:
    from strands_harness_optimizer.adapters import apply_formulas_on_strands_agent
    apply_formulas_on_strands_agent(agent, [formula])
"""

import logging

from strands import Agent
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from strands.hooks.registry import HookEvent

from ..formulas import Formula
from .agent_adapter import AgentAdapter

logger = logging.getLogger(__name__)

# Mapping from string trigger timings to strands event types.
TRIGGER_TIMING_MAP = {
    "agent_initialized": AgentInitializedEvent,
    "before_invocation": BeforeInvocationEvent,
    "after_invocation": AfterInvocationEvent,
    "before_model_call": BeforeModelCallEvent,
    "after_model_call": AfterModelCallEvent,
    "before_tool_call": BeforeToolCallEvent,
    "after_tool_call": AfterToolCallEvent,
    "message_added": MessageAddedEvent,
}


class StrandsAdapter(AgentAdapter):
    """Adapter that bridges Formulas to strands-agents via hooks."""

    # Context keys written back by plain ``setattr`` on the agent.
    tunable_params: set[str] = {"system_prompt", "messages"}

    # Name the AgentSkills plugin registers itself under in strands.
    _SKILLS_PLUGIN_NAME = "agent_skills"

    def extract_context(self, agent: Agent) -> dict:
        """Extract context from a strands Agent.

        Args:
            agent: The strands Agent instance.

        Returns:
            Dict with "system_prompt" and "messages" keys, plus "skills"
            (the AgentSkills plugin's available skills) when the plugin is
            attached to the agent.
        """
        context = {
            "system_prompt": agent.system_prompt,
            "messages": list(agent.messages) if hasattr(agent, "messages") else [],
        }
        skills_plugin = self._get_skills_plugin(agent)
        if skills_plugin is not None:
            context["skills"] = skills_plugin.get_available_skills()
        return context

    def update_context(self, agent: Agent, context: dict) -> None:
        """Apply updated context back to a strands Agent.

        ``system_prompt``/``messages`` are set directly on the agent. ``skills``
        is pushed into the AgentSkills plugin via ``set_available_skills()`` so
        a Formula can drive skill definitions without touching plugin internals
        or relying on a shared Skill reference.

        Args:
            agent: The strands Agent instance.
            context: Dict with keys to update (e.g., "system_prompt",
                "messages", "skills").
        """
        for key in self.tunable_params:
            if key in context:
                setattr(agent, key, context[key])

        if "skills" in context:
            skills_plugin = self._get_skills_plugin(agent)
            if skills_plugin is None:
                logger.warning(
                    "Formula produced 'skills' but no AgentSkills plugin is "
                    "attached to the agent; skipping skills update."
                )
            else:
                skills_plugin.set_available_skills(context["skills"])

    def _get_skills_plugin(self, agent: Agent):
        """Return the agent's AgentSkills plugin, or None if not attached.

        Located by plugin name (``AgentSkills.name == "agent_skills"``) via the
        agent's plugin registry. Duck-typed on ``set_available_skills`` so the
        adapter does not hard-depend on the AgentSkills class being importable.
        """
        registry = getattr(agent, "_plugin_registry", None)
        plugins = getattr(registry, "_plugins", None)
        if not plugins:
            return None
        plugin = plugins.get(self._SKILLS_PLUGIN_NAME)
        if plugin is not None and hasattr(plugin, "set_available_skills"):
            return plugin
        return None

    def apply_to_agent(self, formulas: list[Formula], agent: Agent) -> Agent:
        """Register formulas as hook callbacks on a strands agent.

        For each formula, registers a hook callback for every event type in
        the formula's trigger_timings. When the event fires, the formula's
        process() method is called with the agent's context.

        Args:
            formulas: List of Formula instances to attach.
            agent: The strands Agent instance.

        Returns:
            The agent with formulas applied.
        """
        for formula in formulas:
            event_types = formula.trigger_timings or [BeforeInvocationEvent]

            for timing in event_types:
                if isinstance(timing, str):
                    event_type = TRIGGER_TIMING_MAP.get(timing)
                    if event_type is None:
                        raise ValueError(
                            f"Unknown trigger timing '{timing}' for formula "
                            f"'{formula.name}'. Valid values: {list(TRIGGER_TIMING_MAP.keys())}"
                        )
                else:
                    event_type = timing

                def _make_callback(f: Formula):
                    def callback(event: HookEvent) -> None:
                        context = self.extract_context(event.agent)
                        if f.can_process(context):
                            updated = f.process(context)
                            self.update_context(event.agent, updated)

                    return callback

                agent.add_hook(_make_callback(formula), event_type)
                logger.info(f"Registered formula '{formula.name}' on {event_type.__name__}")

        return agent


class StrandsAgentWithFormulas(Agent):
    """Strands Agent with Formulas applied at initialization.

    Extends strands Agent to accept a list of Formulas that are
    automatically attached as hook callbacks during construction.

    Example:
        formula = SystemPromptFormula(system_prompt="You are an expert coder.")
        agent = StrandsAgentWithFormulas(
            model=model,
            system_prompt="base prompt",
            formulas=[formula],
        )
    """

    def __init__(self, *args, formulas: list[Formula] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if formulas:
            adapter = StrandsAdapter()
            adapter.apply_to_agent(formulas, self)


def apply_formulas_on_strands_agent(agent: Agent, formulas: list[Formula]) -> "Agent":
    """Convenience function to apply formulas to a strands agent.

    Creates a StrandsAdapter, applies formulas, and returns the agent.

    Args:
        agent: The strands Agent instance.
        formulas: List of Formula instances to attach.

    Returns:
        The agent with formulas applied.
    """
    adapter = StrandsAdapter()
    return adapter.apply_to_agent(formulas, agent)
