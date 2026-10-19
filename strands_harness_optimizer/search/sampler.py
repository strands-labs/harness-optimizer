"""SearchSampler: let a search's item-sampling policy choose the Trainer's batch.

Under the Trainer, which items a step runs on is the DataLoader's decision, not the optimizer's. A
policy such as `stratified_sampler` needs the search state to make it — the current parent and its
per-item scores — so this Sampler asks the optimizer for that state each time the DataLoader starts
an epoch, which is once per search step.
"""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

from ..data import Dataset, Sampler


class SearchSampler(Sampler[int]):
    """Yield the dataset indices of the items `policy` picks for the optimizer's current step.

    Args:
        data_source: The feedback dataset the DataLoader reads.
        optimizer: The `CandidateSearchOptimizer`; read for `iteration`, `parent`, `store` and
            `item_scores`.
        policy: (iteration, parent, store, item_ids, scores=) -> item_ids, e.g.
            `stratified_sampler(...)`. `scores` is the optimizer's running per-item score map.
        id_key: The sample field holding the item id. Default: the optimizer's.
    """

    def __init__(
        self,
        data_source: Dataset,
        optimizer,
        policy: Callable[..., Sequence[str]],
        id_key: str | None = None,
    ) -> None:
        self.data_source = data_source
        self.optimizer = optimizer
        self.policy = policy
        key = id_key or optimizer.id_key
        self._ids = [str(data_source[i][key]) for i in range(len(data_source))]
        self._index = {item: i for i, item in enumerate(self._ids)}
        if len(self._index) != len(self._ids):
            raise ValueError("item ids in the feedback dataset are not unique")

    def __iter__(self) -> Iterator[int]:
        o = self.optimizer
        chosen = self.policy(o.iteration, o.parent, o.store, list(self._ids), scores=o.item_scores)
        missing = [i for i in chosen if str(i) not in self._index]
        if missing:
            raise KeyError(f"policy chose items not in the dataset: {missing[:5]}")
        return iter([self._index[str(i)] for i in chosen])
