"""
Minimal Trainer for the Harness Optimizer optimization loop.

Connects DataLoader, AgentRolloutEngine, RewardFunction, and FormulaOptimizer
into an automated training loop. Users can easily write their own loop
if they need more control.
"""

import logging

from .data import DataLoader
from .formulas import Formula
from .optimizers import FormulaOptimizer
from .rewards import RewardFunction
from .rollout_engines import AgentRolloutEngine

logger = logging.getLogger(__name__)


class Trainer:
    """
    Minimal training loop tying all components together.

    The loop structure:
        for epoch in range(n_epochs):
            for batch in dataloader:
                rollouts = list(engine.generate_batch(batch))
                rewards = [reward_fn(rollout) for rollout in rollouts]
                optimizer.add_rollouts(rollouts)
                optimizer.add_rewards(rewards)
            optimizer.step()
            optimizer.zero()

    Args:
        formula: The Formula being optimized.
        optimizer: The FormulaOptimizer (must be initialized with the same formula).
        reward_fn: RewardFunction to score rollouts.
        engine: AgentRolloutEngine to generate rollouts (must be initialized
            with the same formula).
        dataloader: DataLoader providing batches of data samples.
        n_epochs: Number of training epochs.

    Example:
        trainer = Trainer(
            formula=formula,
            optimizer=optimizer,
            reward_fn=reward_fn,
            engine=engine,
            dataloader=dataloader,
            n_epochs=3,
        )
        trainer.fit()
    """

    def __init__(
        self,
        formula: Formula,
        optimizer: FormulaOptimizer,
        reward_fn: RewardFunction,
        engine: AgentRolloutEngine,
        dataloader: DataLoader,
        n_epochs: int = 1,
    ):
        self.formula = formula
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.engine = engine
        self.dataloader = dataloader
        self.n_epochs = n_epochs

    def fit(self) -> list[dict]:
        """
        Run the training loop.

        Returns:
            List of per-epoch stats: [{"epoch": int, "avg_reward": float}, ...]
        """
        epoch_stats = []

        for epoch in range(self.n_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.n_epochs}")

            epoch_rewards = []

            for batch in self.dataloader:
                # Wait for all rollouts to arrive (generate_batch can be async)
                rollouts = list(self.engine.generate_batch(batch))
                rewards = [self.reward_fn(rollout) for rollout in rollouts]
                epoch_rewards.extend(r.reward for r in rewards)

                self.optimizer.add_rollouts(rollouts)
                self.optimizer.add_rewards(rewards)

            # Step and reset
            self.optimizer.step()
            self.optimizer.zero()

            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            stats = {"epoch": epoch + 1, "avg_reward": avg_reward}
            epoch_stats.append(stats)
            logger.info(f"  Avg reward: {avg_reward:.4f}")

        return epoch_stats
