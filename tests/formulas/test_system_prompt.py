"""Tests for SystemPromptFormula."""

import pytest
from strands.hooks.events import BeforeInvocationEvent

from harness_optimizer.formulas import SystemPromptFormula


class TestSystemPromptFormulaInit:
    def test_default_system_prompt_is_none(self):
        formula = SystemPromptFormula()
        assert formula.system_prompt is None

    def test_custom_system_prompt(self):
        formula = SystemPromptFormula(system_prompt="Be concise.")
        assert formula.system_prompt == "Be concise."

    def test_name(self):
        formula = SystemPromptFormula()
        assert formula.name == "system_prompt_formula"

    def test_trigger_timings(self):
        formula = SystemPromptFormula()
        assert formula.trigger_timings == [BeforeInvocationEvent]


class TestSystemPromptFormulaProcess:
    def test_updates_system_prompt(self, sample_context):
        formula = SystemPromptFormula(system_prompt="New prompt.")
        result = formula.process(sample_context)
        assert result["system_prompt"] == "New prompt."

    def test_returns_context_when_no_prompt(self, sample_context):
        formula = SystemPromptFormula()
        result = formula.process(sample_context)
        assert result == sample_context

    def test_returns_context_when_empty_prompt(self, sample_context):
        formula = SystemPromptFormula(system_prompt="")
        result = formula.process(sample_context)
        assert result == sample_context


class TestSystemPromptFormulaTunableParams:
    def test_get_tunable_params(self):
        formula = SystemPromptFormula(system_prompt="Hello")
        assert formula.get_tunable_params() == {"system_prompt": "Hello"}

    def test_get_tunable_params_none(self):
        formula = SystemPromptFormula()
        assert formula.get_tunable_params() == {"system_prompt": None}

    def test_update_params(self):
        formula = SystemPromptFormula(system_prompt="old")
        formula.update_params({"system_prompt": "new"})
        assert formula.system_prompt == "new"
        assert formula.get_tunable_params() == {"system_prompt": "new"}

    def test_update_params_ignores_unknown(self):
        formula = SystemPromptFormula(system_prompt="keep")
        formula.update_params({"other_key": "ignored"})
        assert formula.system_prompt == "keep"

    def test_roundtrip(self):
        """Get params, modify, update — verify consistency."""
        formula = SystemPromptFormula(system_prompt="v1")
        params = formula.get_tunable_params()
        params["system_prompt"] = "v2"
        formula.update_params(params)
        assert formula.get_tunable_params() == {"system_prompt": "v2"}
