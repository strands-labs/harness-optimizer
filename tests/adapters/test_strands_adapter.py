"""Unit tests for the strands agent adapter."""

from unittest.mock import MagicMock

import pytest
from strands.hooks.events import BeforeInvocationEvent

from strands_harness_optimizer.adapters import StrandsAdapter, apply_formulas_on_strands_agent
from strands_harness_optimizer.formulas import SystemPromptFormula


@pytest.fixture
def adapter():
    return StrandsAdapter()


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.system_prompt = "original prompt"
    agent.messages = [{"role": "user", "content": "hello"}]
    return agent


def test_extract_and_update_context(adapter, mock_agent):
    """Extract context, verify fields, update system_prompt, verify change."""
    context = adapter.extract_context(mock_agent)
    assert context["system_prompt"] == "original prompt"
    assert context["messages"] == [{"role": "user", "content": "hello"}]

    adapter.update_context(mock_agent, {"system_prompt": "new"})
    assert mock_agent.system_prompt == "new"


def test_apply_to_agent_and_callback(adapter, mock_agent):
    """String timing maps to event type, callback updates prompt, invalid raises."""
    # String trigger timing maps to BeforeInvocationEvent
    formula = SystemPromptFormula(system_prompt="optimized")
    formula.trigger_timings = ["before_invocation"]
    result = adapter.apply_to_agent([formula], mock_agent)

    mock_agent.add_hook.assert_called_once()
    assert mock_agent.add_hook.call_args[0][1] == BeforeInvocationEvent
    assert result is mock_agent

    # Callback fires and updates system prompt
    callback = mock_agent.add_hook.call_args[0][0]
    event = MagicMock(spec=BeforeInvocationEvent)
    event.agent = mock_agent
    callback(event)
    assert mock_agent.system_prompt == "optimized"

    # Invalid string raises ValueError
    formula.trigger_timings = ["invalid_timing"]
    with pytest.raises(ValueError):
        adapter.apply_to_agent([formula], mock_agent)


def test_convenience_function(mock_agent):
    """Convenience function delegates to adapter and returns agent."""
    formula = SystemPromptFormula(system_prompt="new")
    result = apply_formulas_on_strands_agent(mock_agent, [formula])
    mock_agent.add_hook.assert_called_once()
    assert result is mock_agent


# --- Skills syncing (AgentSkills plugin) ---


class _FakeSkillsPlugin:
    """Minimal stand-in for the AgentSkills plugin's public surface."""

    def __init__(self, skills):
        self._skills = list(skills)

    def get_available_skills(self):
        return list(self._skills)

    def set_available_skills(self, skills):
        self._skills = list(skills)


def _agent_with_plugin(plugin):
    agent = MagicMock()
    agent.system_prompt = "p"
    agent.messages = []
    # Mirror strands' agent._plugin_registry._plugins dict.
    agent._plugin_registry._plugins = {"agent_skills": plugin} if plugin is not None else {}
    return agent


def test_extract_context_includes_skills_when_plugin_present(adapter):
    plugin = _FakeSkillsPlugin(["skillA"])
    agent = _agent_with_plugin(plugin)
    context = adapter.extract_context(agent)
    assert context["skills"] == ["skillA"]


def test_extract_context_omits_skills_without_plugin(adapter):
    agent = _agent_with_plugin(None)
    context = adapter.extract_context(agent)
    assert "skills" not in context


def test_update_context_pushes_skills_into_plugin(adapter):
    plugin = _FakeSkillsPlugin(["old"])
    agent = _agent_with_plugin(plugin)
    adapter.update_context(agent, {"skills": ["new1", "new2"]})
    assert plugin.get_available_skills() == ["new1", "new2"]


def test_update_context_skills_without_plugin_is_noop(adapter):
    agent = _agent_with_plugin(None)
    # Should warn and skip, not raise.
    adapter.update_context(agent, {"skills": ["x"]})


# --- Tool descriptions (tool registry patching) ---


class _Registry:
    """Stand-in for ``agent.tool_registry`` with a real ``{name: tool}`` dict."""

    def __init__(self, tools):
        self.registry = {t.tool_name: t for t in tools}


def _decorated_tools():
    from strands import tool

    @tool
    def search(query: str) -> str:
        """Search the catalog."""
        return query

    @tool
    def click(element: str) -> str:
        """Click an element on the page."""
        return element

    return [search, click]


class _FakeMcpTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class _FakeMcpAgentTool:
    """Mirrors MCPAgentTool: ``tool_spec`` is derived from ``mcp_tool`` on every access."""

    def __init__(self, name, description):
        self.tool_name = name
        self.mcp_tool = _FakeMcpTool(name, description)

    @property
    def tool_spec(self):
        return {"name": self.tool_name, "description": self.mcp_tool.description}


def _registry_agent(tools):
    agent = MagicMock()
    agent.system_prompt = "p"
    agent.messages = []
    agent.tool_registry = _Registry(tools)
    return agent


def test_extract_context_includes_tool_descriptions(adapter):
    agent = _registry_agent(_decorated_tools())
    ctx = adapter.extract_context(agent)
    assert ctx["tool_descriptions"] == {
        "search": "Search the catalog.",
        "click": "Click an element on the page.",
    }


def test_extract_context_without_registry_omits_key(adapter, mock_agent):
    """A bare MagicMock has no dict-typed registry, so the key must be absent."""
    assert "tool_descriptions" not in adapter.extract_context(mock_agent)


def test_update_context_patches_decorated_tool_spec_in_place(adapter):
    """The seam for @tool functions: the spec dict the registry hands out is mutated."""
    tools = _decorated_tools()
    agent = _registry_agent(tools)
    adapter.update_context(agent, {"tool_descriptions": {"search": "Search by keyword."}})
    assert tools[0].tool_spec["description"] == "Search by keyword."
    assert tools[1].tool_spec["description"] == "Click an element on the page."  # untouched


def test_update_context_patches_mcp_tool_description(adapter):
    """The seam for MCP tools: ``mcp_tool.description``, which tool_spec re-derives from."""
    mcp = _FakeMcpAgentTool("search", "orig")
    agent = _registry_agent([mcp])
    adapter.update_context(agent, {"tool_descriptions": {"search": "patched"}})
    assert mcp.mcp_tool.description == "patched"
    assert mcp.tool_spec["description"] == "patched"


def test_update_context_unknown_and_empty_are_skipped(adapter, caplog):
    import logging

    tools = _decorated_tools()
    agent = _registry_agent(tools)
    with caplog.at_level(logging.WARNING):
        applied = adapter.apply_tool_descriptions(
            agent, {"ghost": "x", "search": "", "click": "Click it."}
        )
    assert applied == ["click"]
    assert "ghost" in caplog.text
    assert tools[0].tool_spec["description"] == "Search the catalog."


def test_update_context_without_registry_warns_and_skips(adapter, mock_agent, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        adapter.update_context(mock_agent, {"tool_descriptions": {"search": "x"}})
    assert "no tool registry" in caplog.text


def test_tool_description_formula_end_to_end_via_hook(adapter):
    """apply_to_agent -> callback -> the decorated tool's spec carries the override."""
    from strands_harness_optimizer.formulas import ToolDescriptionFormula

    tools = _decorated_tools()
    agent = _registry_agent(tools)
    formula = ToolDescriptionFormula(
        base={"search": "Search the catalog.", "click": "Click an element on the page."},
        overrides={"search": "Search by keyword."},
    )
    adapter.apply_to_agent([formula], agent)
    callback = agent.add_hook.call_args[0][0]
    event = MagicMock(spec=BeforeInvocationEvent)
    event.agent = agent
    callback(event)
    assert tools[0].tool_spec["description"] == "Search by keyword."
    assert tools[1].tool_spec["description"] == "Click an element on the page."


def test_seam_check_raises_at_attach_time_without_registry(adapter, mock_agent):
    from strands_harness_optimizer.formulas import ToolDescriptionFormula

    formula = ToolDescriptionFormula(base={"search": "x"})
    with pytest.raises(TypeError, match="tool_registry"):
        adapter.apply_to_agent([formula], mock_agent)


def test_seam_check_raises_for_unpatchable_tool(adapter):
    from strands_harness_optimizer.formulas import ToolDescriptionFormula

    class _Opaque:
        tool_name = "opaque"
        tool_spec = "not a dict"

    agent = _registry_agent([_Opaque()])
    formula = ToolDescriptionFormula(base={"opaque": "x"})
    with pytest.raises(TypeError, match="opaque"):
        adapter.apply_to_agent([formula], agent)


def test_seam_check_not_run_for_other_formulas(adapter, mock_agent):
    """A prompt-only formula on an agent without a registry must still attach."""
    formula = SystemPromptFormula(system_prompt="p")
    adapter.apply_to_agent([formula], mock_agent)
    mock_agent.add_hook.assert_called_once()
