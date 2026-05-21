"""Tests for DataLoader — verifying Dataset + Sampler integration."""

from harness_optimizer.data import Dataset, DataLoader


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test_dataloader_shuffle_with_dict_samples():
    """DataLoader batches, shuffles, and works with dict samples."""
    ds = ListDataset([
        {"prompt": "What is 2+3?", "answer": "5"},
        {"prompt": "What is 10-4?", "answer": "6"},
        {"prompt": "What is 7*8?", "answer": "56"},
        {"prompt": "What is 100/4?", "answer": "25"},
        {"prompt": "What is 15+27?", "answer": "42"},
    ])
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batches = list(loader)
    assert len(batches) == 3  # 2 + 2 + 1
    all_prompts = [s["prompt"] for b in batches for s in b]
    assert len(all_prompts) == 5
    assert set(all_prompts) == {s["prompt"] for s in ds.data}
