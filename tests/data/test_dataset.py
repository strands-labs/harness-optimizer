"""Tests for Dataset classes."""

from harness_optimizer.data import ChainDataset, Dataset, IterableDataset, Subset


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class RangeIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start


def test_subset_selects_correct_indices():
    """Subset returns items at the specified indices."""
    ds = ListDataset(list(range(10)))
    subset = Subset(ds, [2, 5, 7])
    assert len(subset) == 3
    assert subset[0] == 2
    assert subset[1] == 5
    assert subset[2] == 7


def test_chain_dataset_iterates_all():
    """ChainDataset yields items from all chained IterableDatasets."""
    ds1 = RangeIterableDataset(0, 3)
    ds2 = RangeIterableDataset(10, 13)
    chained = ChainDataset([ds1, ds2])
    assert list(chained) == [0, 1, 2, 10, 11, 12]
    assert len(chained) == 6
