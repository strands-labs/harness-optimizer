"""
Dataset classes for Harness Optimizer.

Adapted from PyTorch (torch.utils.data.dataset) with torch dependencies
replaced by stdlib. Original source is BSD licensed.

See: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataset.py
"""

from typing import Generic, Iterable, Sequence, TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class Dataset(Generic[_T_co]):
    """An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite __getitem__, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    __len__, which is expected to return the size of the dataset.
    """

    def __getitem__(self, index) -> _T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __add__(self, other: "Dataset[_T_co]") -> "Dataset[_T_co]":
        raise NotImplementedError(
            "Use ChainDataset for IterableDatasets or Subset for map-style Datasets."
        )


class IterableDataset(Dataset[_T_co], Iterable[_T_co]):
    """An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite __iter__, which would return an iterator
    of samples in this dataset.
    """

    def __add__(self, other: Dataset[_T_co]):
        return ChainDataset([self, other])


class ChainDataset(IterableDataset):
    """Dataset for chaining multiple IterableDatasets.

    Args:
        datasets: Iterable of IterableDatasets to be chained together.
    """

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            yield from d

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)
        return total


class Subset(Dataset[_T_co]):
    """Subset of a dataset at specified indices.

    Args:
        dataset: The whole Dataset.
        indices: Indices in the whole set selected for subset.
    """

    dataset: Dataset[_T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
