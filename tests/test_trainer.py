"""Tests for the Trainer — full loop with mock components."""

from harness_optimizer.data import Dataset, DataLoader
from harness_optimizer.datamodels import Reward, Rollout
from harness_optimizer.formulas import Formula
from harness_optimizer.rewards import RewardFunction
from harness_optimizer.optimizers import FormulaOptimizer
from harness_optimizer.rollout_engines import AgentRolloutEngine
from harness_optimizer.trainer import Trainer


class MockFormula(Formula):
    def __init__(self):
        super().__init__("mock", ["before_invocation"])
        self.prompt = "initial"

    def process(self, context, **kwargs):
        return {"system_prompt": self.prompt}

    def get_tunable_params(self):
        return {"system_prompt": self.prompt}

    def update_params(self, params):
        self.prompt = params.get("system_prompt", self.prompt)

    def can_process(self, context):
        return True


class MockReward(RewardFunction):
    def __call__(self, rollout: Rollout) -> Reward:
        response_text = rollout.metadata.get("response_text", "")
        return Reward(reward=1 if "good" in response_text else 0)


class MockEngine(AgentRolloutEngine):
    def generate_batch(self, data_samples):
        for sample in data_samples:
            yield Rollout(
                data_sample=sample,
                messages=[{"role": "assistant", "content": "good answer"}],
                metadata={"response_text": "good answer"},
            )


class MockOptimizer(FormulaOptimizer):
    def step(self):
        avg = sum(r.reward for r in self._rewards) / len(self._rewards) if self._rewards else 0
        self.formula.update_params({"system_prompt": f"optimized (avg_reward={avg})"})


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


def test_trainer_full_loop():
    """Trainer runs epochs, connects all components, formula is updated."""
    formula = MockFormula()
    optimizer = MockOptimizer(formula)
    reward_fn = MockReward()
    engine = MockEngine(formula)
    dataset = ListDataset([{"prompt": "q1"}, {"prompt": "q2"}, {"prompt": "q3"}])
    dataloader = DataLoader(dataset, batch_size=2)

    trainer = Trainer(
        formula=formula,
        optimizer=optimizer,
        reward_fn=reward_fn,
        engine=engine,
        dataloader=dataloader,
        n_epochs=2,
    )

    stats = trainer.fit()

    # Verify stats
    assert len(stats) == 2
    assert stats[0]["epoch"] == 1
    assert stats[0]["avg_reward"] == 1.0  # MockEngine returns "good answer"

    # Verify formula was updated
    assert "optimized" in formula.get_tunable_params()["system_prompt"]
