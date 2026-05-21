"""
Simplified DataLoader for Harness Optimizer.

Composes Dataset + Sampler + BatchSampler for batched iteration.
Single-process only — parallelism happens at the AgentRolloutEngine level.
"""

from .dataset import Dataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler


class DataLoader:
    """Simplified DataLoader that iterates over a Dataset in batches.

    Args:
        dataset: Dataset to load data from.
        batch_size: Number of samples per batch.
        shuffle: If True, data is shuffled at every iteration.
        drop_last: If True, drop the last incomplete batch.
        sampler: Custom sampler. Mutually exclusive with shuffle.

    Example:
        dataset = MyDataset(data)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        for batch in loader:
            # batch is a list of 4 samples
            process(batch)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Sampler | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle are mutually exclusive")

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield [self.dataset[i] for i in batch_indices]

    def __len__(self):
        return len(self.batch_sampler)
