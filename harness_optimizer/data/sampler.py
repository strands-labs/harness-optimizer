"""
Sampler classes for Harness Optimizer.

Adapted from PyTorch (torch.utils.data.sampler) with torch dependencies
replaced by stdlib. Original source is BSD licensed.

See: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py
"""

import itertools
import random
from typing import Generic, Iterable, Iterator, List, Optional, Sized, TypeVar, Union


_T_co = TypeVar("_T_co", covariant=True)


class Sampler(Generic[_T_co]):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a __len__ method that returns the length of the returned iterators.
    """

    def __iter__(self) -> Iterator[_T_co]:
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.

    Args:
        data_source: Dataset to sample from.
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """Samples elements randomly.

    Args:
        data_source: Dataset to sample from.
        replacement: If True, samples are drawn with replacement.
        num_samples: Number of samples to draw. Defaults to len(dataset).
        generator: Random generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[random.Random] = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        gen = self.generator or random.Random()

        if self.replacement:
            for _ in range(self.num_samples):
                yield gen.randint(0, n - 1)
        else:
            # Generate full permutation(s)
            for _ in range(self.num_samples // n):
                indices = list(range(n))
                gen.shuffle(indices)
                yield from indices
            # Remaining samples
            indices = list(range(n))
            gen.shuffle(indices)
            yield from indices[: self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[List[int]]):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler: Base sampler. Can be any iterable object.
        batch_size: Size of mini-batch.
        drop_last: If True, the sampler will drop the last batch if
            its size would be less than batch_size.
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            args = [sampler_iter] * self.batch_size
            for batch_droplast in zip(*args):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
