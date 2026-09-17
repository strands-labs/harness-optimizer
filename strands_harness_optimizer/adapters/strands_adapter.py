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

    # Context key carrying ``{tool_name: description}`` overrides.
    _TOOL_DESCRIPTIONS_KEY = "tool_descriptions"

    def extract_context(self, agent: Agent) -> dict:
        """Extract context from a strands Agent.

        Args:
            agent: The strands Agent instance.

        Returns:
            Dict with "system_prompt" and "messages" keys, plus "skills"
            (the AgentSkills plugin's available skills) when the plugin is
            attached to the agent, plus "tool_descriptions" (the current
            ``{tool_name: description}`` of every registered tool) when the
            agent exposes a tool registry.
        """
        context = {
            "system_prompt": agent.system_prompt,
            "messages": list(agent.messages) if hasattr(agent, "messages") else [],
        }
        skills_plugin = self._get_skills_plugin(agent)
        if skills_plugin is not None:
            context["skills"] = skills_plugin.get_available_skills()
        registry = self._get_tool_registry(agent)
        if registry is not None:
            context[self._TOOL_DESCRIPTIONS_KEY] = {
                name: self._read_tool_description(tool) for name, tool in registry.items()
            }
        return context

    def update_context(self, agent: Agent, context: dict) -> None:
        """Apply updated context back to a strands Agent.

        ``system_prompt``/``messages`` are set directly on the agent. ``skills``
        is pushed into the AgentSkills plugin via ``set_available_skills()`` so
        a Formula can drive skill definitions without touching plugin internals
        or relying on a shared Skill reference. ``tool_descriptions`` patches the
        named tools' specs in place; tools not named keep their description.

        Args:
            agent: The strands Agent instance.
            context: Dict with keys to update (e.g., "system_prompt",
                "messages", "skills", "tool_descriptions").
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

        if self._TOOL_DESCRIPTIONS_KEY in context:
            self.apply_tool_descriptions(agent, context[self._TOOL_DESCRIPTIONS_KEY] or {})

    # ── tool descriptions ────────────────────────────────────────────────────
    def _get_tool_registry(self, agent: Agent):
        """The agent's ``{name: tool}`` registry, or None if this agent has none.

        Duck-typed on a dict-valued ``agent.tool_registry.registry`` so a mock or a
        non-strands agent simply reports no tools rather than raising.
        """
        registry = getattr(getattr(agent, "tool_registry", None), "registry", None)
        return registry if isinstance(registry, dict) else None

    @staticmethod
    def _read_tool_description(tool) -> str:
        spec = getattr(tool, "tool_spec", None)
        if isinstance(spec, dict):
            return str(spec.get("description") or "")
        return ""

    @staticmethod
    def _write_tool_description(tool, text: str) -> None:
        """Patch one tool's description at the seam its ``tool_spec`` reads from.

        Two kinds of tool exist. An MCP tool builds ``tool_spec`` on every access
        from ``mcp_tool.description``, so the text must be written THERE -- a dict
        returned by the property would be a fresh copy. A Python tool
        (``@tool``-decorated or module-based) returns its own spec dict, which is
        mutable in place. Setting the MCP field first and the dict second covers
        both without knowing the class.
        """
        mcp_tool = getattr(tool, "mcp_tool", None)
        if mcp_tool is not None and hasattr(mcp_tool, "description"):
            mcp_tool.description = text
            return
        spec = getattr(tool, "tool_spec", None)
        if isinstance(spec, dict):
            spec["description"] = text
            return
        raise TypeError(
            f"tool {type(tool).__name__} exposes neither 'mcp_tool.description' nor a "
            "dict-typed 'tool_spec'; cannot patch its description"
        )

    def apply_tool_descriptions(self, agent: Agent, overrides: dict) -> list[str]:
        """Patch the named tools; return the names actually applied.

        Unknown names are logged, not raised: a stale name from an earlier toolset
        should not abort an invocation, but it must not pass silently either, since
        a run that quietly applied nothing looks exactly like "the edit did not help".
        """
        registry = self._get_tool_registry(agent)
        if registry is None:
            if overrides:
                logger.warning(
                    "Formula produced 'tool_descriptions' but the agent has no tool "
                    "registry; skipping."
                )
            return []
        applied: list[str] = []
        for name, text in overrides.items():
            tool = registry.get(name)
            if tool is None:
                logger.warning(
                    "tool_descriptions: no registered tool named %r (have: %s) -- skipped",
                    name,
                    sorted(registry),
                )
                continue
            if not str(text or "").strip():
                logger.warning("tool_descriptions: %r is empty -- keeping the original", name)
                continue
            self._write_tool_description(tool, str(text))
            applied.append(name)
        if applied:
            logger.info("tool_descriptions: applied %s", ", ".join(sorted(applied)))
        return applied

    def _check_tool_description_seam(self, agent: Agent, formula: Formula) -> None:
        """Fail at attach time, not at the first invocation, if the seam is absent.

        Tested against strands-agents 1.33 and 1.47. The package's version floor is
        lower than either, so a user on an older strands gets one clear error here
        instead of a TypeError from inside a hook callback.
        """
        registry = self._get_tool_registry(agent)
        if registry is None:
            raise TypeError(
                f"Formula '{formula.name}' needs 'agent.tool_registry.registry' (a dict of "
                "tools) to patch tool descriptions, and this agent does not expose one. "
                "Tested with strands-agents 1.33 and 1.47."
            )
        for name, tool in registry.items():
            mcp_tool = getattr(tool, "mcp_tool", None)
            if mcp_tool is not None and hasattr(mcp_tool, "description"):
                continue
            if isinstance(getattr(tool, "tool_spec", None), dict):
                continue
            raise TypeError(
                f"Formula '{formula.name}': tool {name!r} ({type(tool).__name__}) exposes "
                "neither 'mcp_tool.description' nor a dict-typed 'tool_spec', so its "
                "description cannot be patched. Tested with strands-agents 1.33 and 1.47."
            )

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
            if self._TOOL_DESCRIPTIONS_KEY in (formula.get_tunable_params() or {}):
                self._check_tool_description_seam(agent, formula)

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
