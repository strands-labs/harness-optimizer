"""
Base reward function interface for reward computation.

Reward functions compute rewards and return a Reward object.
"""

from abc import ABC, abstractmethod

from harness_optimizer.datamodels import Reward, Rollout


class RewardFunction(ABC):
    """
    Base class for reward functions.

    Users implement __call__() and return a Reward object.

    Example:
        class MyReward(RewardFunction):
            def __call__(self, rollout: Rollout) -> Reward:
                response = rollout.messages[-1].get("content", "") if rollout.messages else ""
                expected = rollout.metadata.get("expected_answer", "")
                match = response.strip() == expected.strip()
                return Reward(reward=1 if match else 0)
    """

    @abstractmethod
    def __call__(self, rollout: Rollout) -> Reward:
        """
        Compute reward for the rollout.

        Args:
            rollout: Rollout object containing data_sample, messages, and optional metrics.

        Returns:
            Reward: A Reward object containing the reward value.
        """
        pass


class NaiveRewardFunction(RewardFunction):
    """
    Naive reward function that extracts existing reward from Rollout metrics.

    If the rollout metrics contain a reward, returns it. Otherwise returns Reward(0).
    Useful when rewards are computed during rollout generation.
    """

    def __call__(self, rollout: Rollout) -> Reward:
        """
        Extract reward from rollout metrics if it exists, otherwise return 0.

        Args:
            rollout: Rollout object that may contain metrics with a reward.

        Returns:
            Reward: The existing reward or Reward(reward=0) if none exists.
        """
        if rollout.metrics and "reward" in rollout.metrics:
            return Reward(reward=rollout.metrics["reward"])
        return Reward(reward=0)
