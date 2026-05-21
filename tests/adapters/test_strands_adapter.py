"""Unit tests for the strands agent adapter."""

import pytest
from unittest.mock import MagicMock

from strands.hooks.events import BeforeInvocationEvent

from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import StrandsAdapter, apply_formulas_on_strands_agent


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
