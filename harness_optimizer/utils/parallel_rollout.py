"""
Parallel execution utilities for rollout generation.

Standalone functions for num_rollouts expansion, parallel dispatch via
ThreadPoolExecutor, and error handling. Used by LocalRolloutEngine and
AgentCoreRolloutEngine.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from harness_optimizer.datamodels import Rollout

logger = logging.getLogger(__name__)


def expand_for_num_rollouts(data_samples: list[dict], num_rollouts: int) -> list[dict]:
    """Duplicate each data sample num_rollouts times.

    Assigns a unique UUID string to each data sample in the 'uuid' key before expansion.

    Args:
        data_samples: List of input task dicts.
        num_rollouts: Number of rollouts per sample.

    Returns:
        Expanded list with length len(data_samples) * num_rollouts.
    """
    # Ensure all samples have task_id
    for sample in data_samples:
        sample["uuid"] = str(uuid.uuid4())

    if num_rollouts <= 1:
        return data_samples
    return [s for s in data_samples for _ in range(num_rollouts)]


def run_parallel(
    fn: Callable[[dict], Rollout],
    tasks: list[dict],
    num_workers: int = 1,
) -> list[Rollout]:
    """Run fn(task) across tasks, sequentially or in parallel.

    Uses ThreadPoolExecutor when num_workers > 1, otherwise sequential.
    Order of results matches order of tasks.

    Args:
        fn: Callable that takes a data_sample dict and returns a Rollout.
        tasks: List of data_sample dicts to process.
        num_workers: Number of parallel workers. 1 = sequential.

    Returns:
        List of Rollout objects, preserving task order.
    """
    if num_workers <= 1:
        return [safe_invoke(fn, t) for t in tasks]

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(safe_invoke, fn, s): i for i, s in enumerate(tasks)}
        for future in as_completed(future_to_idx, timeout=3600):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=900)
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                results[idx] = Rollout(
                    data_sample=tasks[idx], messages=[], metadata={"error": str(e)}
                )
    return results


def safe_invoke(fn: Callable[[dict], Rollout], data_sample: dict) -> Rollout:
    """Call fn(data_sample) with error handling.

    Args:
        fn: Callable that takes a data_sample dict and returns a Rollout.
        data_sample: Input task dict.

    Returns:
        Rollout object, or error Rollout if fn raises.
    """
    try:
        return fn(data_sample)
    except Exception as e:
        logger.error(f"Invocation failed for {data_sample}: {e}")
        return Rollout(data_sample=data_sample, messages=[], metadata={"error": str(e)})
