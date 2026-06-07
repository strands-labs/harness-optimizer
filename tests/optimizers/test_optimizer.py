"""Tests for the FormulaOptimizer base class."""

import pytest

from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.optimizers import FormulaOptimizer


class PickBestOptimizer(FormulaOptimizer):
    """Picks the prompt hint from the highest-reward rollout."""

    def step(self):
        best_idx = max(range(len(self._rewards)), key=lambda i: self._rewards[i]["reward_value"])
        hint = self._rollouts[best_idx].get("data_sample", {}).get("hint", "")
        params = self.formula.get_tunable_params()
        self.formula.update_params({"system_prompt": f"{params['system_prompt']} {hint}".strip()})


class StatefulOptimizer(FormulaOptimizer):
    def __init__(self, formula):
        super().__init__(formula)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.formula.update_params(self.formula.get_tunable_params())

    def get_state(self):
        return {"step_count": self.step_count}

    def load_state(self, state):
        self.step_count = state["step_count"]


def test_optimizer_accumulate_step_and_checkpoint():
    """ABC enforcement, add_rollouts + add_rewards + step + zero, checkpoint roundtrip."""
    with pytest.raises(TypeError):
        FormulaOptimizer(SystemPromptFormula(system_prompt="test"))

    # Accumulate, step, verify formula updated (rollout contains data_sample)
    formula = SystemPromptFormula(system_prompt="original")
    optimizer = PickBestOptimizer(formula)
    optimizer.add_rollouts(
        [
            {"messages": ["t1"], "data_sample": {"hint": "be brief"}},
            {"messages": ["t2"], "data_sample": {"hint": "use examples"}},
        ]
    )
    optimizer.add_rewards(
        [{"reward_value": 0.2}, {"reward_value": 0.8}],
    )
    optimizer.step()
    assert "use examples" in formula.get_tunable_params()["system_prompt"]

    # Zero resets accumulated data
    optimizer.zero()
    assert optimizer._rollouts == []
    assert optimizer._rewards == []

    # get_state raises without implementation
    with pytest.raises(NotImplementedError):
        optimizer.get_state()

    # Stateful checkpoint roundtrip
    stateful = StatefulOptimizer(formula)
    stateful.step()
    stateful.step()
    new = StatefulOptimizer(formula)
    new.load_state(stateful.get_state())
    assert new.step_count == 2
