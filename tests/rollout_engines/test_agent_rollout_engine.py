"""Tests for AgentRolloutEngine and parallel utilities."""

import pytest

from harness_optimizer.datamodels import Rollout
from harness_optimizer.formulas import Formula
from harness_optimizer.rollout_engines import AgentRolloutEngine, LocalRolloutEngine


class MockFormula(Formula):
    def __init__(self, prompt="initial"):
        super().__init__("mock", ["before_invocation"])
        self.prompt = prompt

    def process(self, context, **kwargs):
        return {"system_prompt": self.prompt}

    def get_tunable_params(self):
        return {"system_prompt": self.prompt}

    def update_params(self, params):
        self.prompt = params.get("system_prompt", self.prompt)


def test_engine_abc_and_local_engine():
    """ABC enforcement, LocalRolloutEngine dispatch, generate convenience, num_rollouts."""
    formula = MockFormula()

    with pytest.raises(TypeError):
        AgentRolloutEngine(formula)

    engine = LocalRolloutEngine(
        formula=formula,
        agent_creator=lambda: None,
        agent_invoker=lambda agent, sample: Rollout(
            data_sample=sample,
            messages=[{"role": "assistant", "content": sample["prompt"]}],
            metadata={"response_text": sample["prompt"]},
        ),
    )

    # generate_batch yields individual Rollout objects
    rollouts = list(engine.generate_batch([{"prompt": "hello"}, {"prompt": "world"}]))
    assert len(rollouts) == 2
    assert rollouts[0].metadata["response_text"] == "hello"

    # generate() convenience
    rollout = engine.generate({"prompt": "test"})
    assert rollout.metadata["response_text"] == "test"

    # num_rollouts
    engine3 = LocalRolloutEngine(
        formula=formula,
        agent_creator=lambda: None,
        agent_invoker=lambda agent, sample: Rollout(
            data_sample=sample,
            messages=[{"role": "assistant", "content": sample["prompt"]}],
            metadata={"response_text": sample["prompt"]},
        ),
        num_rollouts=3,
    )
    rollouts3 = list(engine3.generate_batch([{"prompt": "a"}]))
    assert len(rollouts3) == 3

    # ensure_sync_params called (LocalRolloutEngine uses default no-op)
    formula.update_params({"system_prompt": "updated"})
    list(engine.generate_batch([{"prompt": "x"}]))
    assert formula.get_tunable_params() == {"system_prompt": "updated"}
