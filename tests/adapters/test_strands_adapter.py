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
