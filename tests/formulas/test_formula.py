"""Tests for the Formula base class."""

import pytest
from strands.hooks.events import BeforeInvocationEvent, AfterInvocationEvent

from harness_optimizer.formulas import Formula


class ConcreteFormula(Formula):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, value: str = "test"):
        super().__init__("test_formula", [BeforeInvocationEvent])
        self.value = value

    def process(self, context: dict, **kwargs) -> dict:
        return {"system_prompt": self.value}

    def get_tunable_params(self) -> dict:
        return {"value": self.value}

    def update_params(self, params: dict) -> None:
        if "value" in params:
            self.value = params["value"]


class TestFormulaInit:
    def test_name(self):
        formula = ConcreteFormula()
        assert formula.name == "test_formula"

    def test_trigger_timings(self):
        formula = ConcreteFormula()
        assert formula.trigger_timings == [BeforeInvocationEvent]

    def test_multiple_trigger_timings(self):
        class MultiTiming(Formula):
            def __init__(self):
                super().__init__("multi", [BeforeInvocationEvent, AfterInvocationEvent])
            def process(self, context, **kwargs):
                return context
            def get_tunable_params(self):
                return {}
            def update_params(self, params):
                pass

        formula = MultiTiming()
        assert len(formula.trigger_timings) == 2


class TestFormulaProcess:
    def test_process_returns_updated_context(self, sample_context):
        formula = ConcreteFormula(value="new prompt")
        result = formula.process(sample_context)
        assert result["system_prompt"] == "new prompt"

    def test_process_with_kwargs(self, sample_context):
        formula = ConcreteFormula()
        result = formula.process(sample_context, extra="ignored")
        assert result["system_prompt"] == "test"


class TestFormulaTunableParams:
    def test_get_tunable_params(self):
        formula = ConcreteFormula(value="hello")
        assert formula.get_tunable_params() == {"value": "hello"}

    def test_update_params(self):
        formula = ConcreteFormula(value="old")
        formula.update_params({"value": "new"})
        assert formula.get_tunable_params() == {"value": "new"}

    def test_update_params_ignores_unknown(self):
        formula = ConcreteFormula(value="keep")
        formula.update_params({"unknown_key": "ignored"})
        assert formula.get_tunable_params() == {"value": "keep"}


class TestFormulaCanProcess:
    def test_default_returns_true(self, sample_context):
        formula = ConcreteFormula()
        assert formula.can_process(sample_context) is True

    def test_default_returns_true_empty_context(self):
        formula = ConcreteFormula()
        assert formula.can_process({}) is True


class TestFormulaABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Formula("test", [BeforeInvocationEvent])
