"""
Base class for Formula optimizers.

A FormulaOptimizer proposes new tunable parameters for a Formula based on
collected rollouts and their rewards. Follows PyTorch's optimizer pattern:
accumulate data, then step to update parameters.
"""

from abc import ABC, abstractmethod

from ..datamodels import Reward, Rollout
from ..formulas import Formula


class FormulaOptimizer(ABC):
    """
    Base class for Formula parameter optimizers.

    Follows PyTorch's optimizer pattern:
    1. Initialize with a Formula (like PyTorch optimizer takes model.parameters())
    2. Accumulate rollouts and rewards (like loss.backward() accumulates gradients)
    3. Call step() to update Formula parameters (like optimizer.step())
    4. Call zero() to reset accumulated data (like optimizer.zero_grad())

    Rollouts are expected to contain the data sample information (e.g., task input,
    expected answer) alongside the conversation trace, so there is no separate
    add_data_samples() method. The number of rollouts and rewards must be consistent
    (aligned by index) when step() is called.

    Example:
        class MyOptimizer(FormulaOptimizer):
            def step(self):
                best = max(
                    zip(self._rollouts, self._rewards),
                    key=lambda x: x[1].reward,
                )
                self.formula.update_params({"system_prompt": "Learned: ..."})

        optimizer = MyOptimizer(formula)
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)
        optimizer.step()
        optimizer.zero()
    """

    def __init__(self, formula: Formula):
        """
        Initialize the optimizer with a Formula.

        Args:
            formula: The Formula whose parameters will be optimized.
        """
        self.formula = formula
        self._rollouts: list[Rollout] = []
        self._rewards: list[Reward] = []

    def add_rollouts(self, rollouts: list[Rollout]) -> None:
        """
        Accumulate rollouts.

        Args:
            rollouts: List of Rollout objects containing data_sample, messages,
                and optional metrics and metadata.
        """
        self._rollouts.extend(rollouts)

    def add_rewards(self, rewards: list[Reward]) -> None:
        """
        Accumulate Reward objects from RewardFunction, aligned with rollouts.

        Args:
            rewards: List of Reward objects containing reward value and metadata.
        """
        self._rewards.extend(rewards)

    @abstractmethod
    def step(self) -> None:
        """
        Propose new parameters and update the Formula.

        Uses accumulated rollouts and rewards to compute new parameters,
        then calls self.formula.update_params() internally.
        """
        pass

    def zero(self) -> None:
        """Reset accumulated rollouts and rewards."""
        self._rollouts = []
        self._rewards = []

    def get_state(self) -> dict:
        """
        Get the current optimizer state for checkpointing.

        Override this method in subclasses that maintain state across
        step() calls (e.g., accumulated playbook, statistics).

        Returns:
            Dictionary containing optimizer state that can be serialized to JSON.

        Raises:
            NotImplementedError: If the subclass does not implement checkpointing.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_state(). "
            "Override get_state() and load_state() to support checkpointing."
        )

    def load_state(self, state: dict) -> None:
        """
        Load optimizer state from a previously saved checkpoint.

        Override this method in subclasses to restore optimizer-specific state.

        Args:
            state: Dictionary containing optimizer state (from get_state()).

        Raises:
            NotImplementedError: If the subclass does not implement checkpointing.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement load_state(). "
            "Override get_state() and load_state() to support checkpointing."
        )
