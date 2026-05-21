"""
Base class for agent rollout engines.

AgentRolloutEngine generates rollouts from data samples. Infrastructure
concerns (parallelism, distribution, agent pooling) are handled internally.

The generate_batch() method returns an Iterator[Rollout] to support both
synchronous and asynchronous generation patterns:
- Sync engines yield once with the full batch of rollouts.
- Async engines yield multiple times as partial results arrive.
The Trainer consumes the iterator the same way in both cases.
"""

from abc import ABC, abstractmethod
from typing import Iterator

from ..datamodels import Rollout
from ..formulas import Formula


class AgentRolloutEngine(ABC):
    """
    Base class for generating agent rollouts.

    Subclass this and implement generate_batch() to define how rollouts
    are generated from data samples.
    """

    def __init__(self, formula: Formula, num_rollouts: int = 1):
        """
        Args:
            formula: The Formula being optimized. Engine reads params
                via formula.get_tunable_params() for sync with agent.
            num_rollouts: Default number of rollouts per data sample.
        """
        self.formula = formula
        self.num_rollouts = num_rollouts

    def ensure_sync_params(self) -> None:
        """Ensure formula params are synced with the agent before generation.

        Default is no-op. Override for engines that need explicit sync
        (e.g., AgentCore pushing params to S3 or payload).

        For Strands engines, params sync happens automatically via hooks
        when the agent is invoked, so this is a no-op.
        """
        pass

    @abstractmethod
    def generate_batch(self, data_samples: list[dict]) -> Iterator[Rollout]:
        """
        Generate rollouts for a batch of data samples.

        Yields Rollout objects one at a time. Sync engines yield all rollouts
        immediately. Async engines yield each rollout as it completes.

        Args:
            data_samples: List of input task dicts.

        Yields:
            Rollout object containing data_sample, messages, and optional metrics.
        """
        pass

    def generate(self, data_sample: dict) -> Rollout:
        """Generate a single rollout. Convenience wrapper."""
        return next(iter(self.generate_batch([data_sample])))
