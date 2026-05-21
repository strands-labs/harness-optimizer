"""
Local rollout engine — generate rollouts by invoking agents locally.

Framework-agnostic: works with any agent type. Pass agent_creator and
agent_invoker callables to define how agents are created and invoked.
"""

import logging
import queue
from typing import Callable, Iterator

from ..datamodels import AgentCreator, AgentInvoker, Rollout, SupportedAgent
from ..formulas import Formula
from .agent_rollout_engine import AgentRolloutEngine
from ..utils.parallel_rollout import expand_for_num_rollouts, run_parallel

logger = logging.getLogger(__name__)


class LocalRolloutEngine(AgentRolloutEngine):
    """
    Rollout engine that invokes agents locally with thread-safe agent pooling.

    Example:
        engine = LocalRolloutEngine(
            formula=formula,
            agent_creator=lambda: Agent(model=model),
            agent_invoker=lambda agent, sample: Rollout(
                data_sample=sample,
                messages=list(agent.messages),
            ),
        )

    Args:
        formula: The Formula being optimized.
        agent_creator: Callable that returns a fresh agent instance.
        agent_invoker: Callable(agent, data_sample) -> Rollout.
        num_rollouts: Default number of rollouts per data sample.
        num_workers: Number of parallel workers. Each gets its own agent.
    """

    def __init__(
        self,
        formula: Formula,
        agent_creator: AgentCreator,
        agent_invoker: AgentInvoker,
        num_rollouts: int = 1,
        num_workers: int = 1,
    ):
        super().__init__(formula, num_rollouts)
        self.num_workers = num_workers
        self._agent_create_fn = agent_creator
        self._agent_invoke_fn = agent_invoker

        # Create agent pool — one per worker for thread safety
        pool_size = max(num_workers, 1)
        self._agent_pool = queue.Queue()
        for _ in range(pool_size):
            self._agent_pool.put(self._agent_create_fn())
        logger.info(f"Created {pool_size} agent(s) for LocalRolloutEngine")

    def generate_batch(self, data_samples: list[dict]) -> Iterator[Rollout]:
        """Generate rollouts by invoking agents from the pool."""
        self.ensure_sync_params()
        tasks = expand_for_num_rollouts(data_samples, self.num_rollouts)
        results = run_parallel(self._pool_invoke, tasks, self.num_workers)
        yield from results

    def _pool_invoke(self, data_sample: dict) -> Rollout:
        """Borrow an agent from the pool, invoke, then return it."""
        agent = self._agent_pool.get()
        try:
            return self._agent_invoke_fn(agent, data_sample)
        finally:
            self._agent_pool.put(agent)
