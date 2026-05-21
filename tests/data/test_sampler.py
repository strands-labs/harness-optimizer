"""Tests for Sampler classes — verifying stdlib random replacement works."""

import random

from harness_optimizer.data import RandomSampler


class FakeDataset:
    def __len__(self):
        return 10


def test_random_sampler_reproducible():
    """RandomSampler with stdlib random.Random is reproducible and covers all indices."""
    ds = FakeDataset()
    s1 = RandomSampler(ds, generator=random.Random(42))
    s2 = RandomSampler(ds, generator=random.Random(42))
    indices = list(s1)
    assert indices == list(s2)
    assert sorted(indices) == list(range(10))
