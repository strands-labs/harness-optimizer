"""
WebShop dataset — the task rows only.

A minimal map-style dataset built directly on strands_harness_optimizer's Dataset
(so it's DataLoader/Sampler-compatible). WebShop tasks are identified by an integer
index into the benchmark's goal list; a split just selects a contiguous range of
those indices. Running/evaluating a task lives in WebShopEnvironment
(dataset/environment.py) — the dataset is just the data.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from strands_harness_optimizer.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class WebShopDataSample:
    """A single WebShop task — just a task id (index into the goal list)."""

    id: str = ""
    input: str = ""
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.input:
            self.input = self.task_id

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __getitem__(self, item):
        return getattr(self, item)


class WebShopDataset(Dataset[WebShopDataSample]):
    """WebShop benchmark dataset (map-style, lazily loaded).

    WebShop ships one instruction per goal; tasks are addressed by the goal's
    integer index. A split is a contiguous range of indices:

        train : indices [0, 100)
        eval  : indices [0, 200)  (offset by +100 at run time, i.e. the held-out set)

    The environment (WebShopEnvironment) resets the WebShop gym to ``task_id`` to
    fetch the task instruction and score the final purchase.
    """

    name = "webshop"
    description = "WebShop benchmark for evaluating agents on e-commerce shopping tasks"

    # Default index ranges per split (mirrors the internal WebShop dataset).
    _SPLIT_RANGES = {
        "train": (0, 100),
        "eval": (0, 200),
    }

    def __init__(
        self,
        split: str = "train",
        start_idx: int = 0,
        end_idx: int = 0,
        task_ids: List[str] | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = None,
    ):
        self.split = split
        self.task_ids = task_ids
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        if end_idx > 0:
            self.start_idx, self.end_idx = start_idx, end_idx
        elif split in self._SPLIT_RANGES:
            self.start_idx, self.end_idx = self._SPLIT_RANGES[split]
        else:
            raise ValueError(
                f"Unknown split '{split}'. Use one of {list(self._SPLIT_RANGES)}, "
                "or pass explicit start_idx/end_idx."
            )
        # eval indices are the held-out goals after the train range.
        self._offset = 0 if split == "train" else 100

        self.data: List[WebShopDataSample] = []
        self.data_index: Dict[str, WebShopDataSample] = {}
        self._loaded = False

    # --- map-style Dataset surface (DataLoader-compatible) ---

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def __len__(self) -> int:
        self.ensure_loaded()
        return len(self.data)

    def __getitem__(self, idx: int) -> WebShopDataSample:
        self.ensure_loaded()
        return self.data[idx]

    def get_data_by_id(self, task_id: str) -> WebShopDataSample:
        self.ensure_loaded()
        return self.data_index[str(task_id)]

    # --- loading ---

    def load(self) -> None:
        """Create one sample per goal index in the split's range."""
        logger.info(
            f"Creating WebShop dataset: split={self.split}, "
            f"range=[{self.start_idx}, {self.end_idx}), offset={self._offset}"
        )
        for idx in range(self.start_idx, self.end_idx):
            task_id = str(idx)
            self.data.append(
                WebShopDataSample(id=task_id, task_id=task_id, input=str(idx + self._offset))
            )

        # An explicit task_ids subset trims the split (used for quick runs).
        if self.task_ids:
            wanted = {str(t) for t in self.task_ids}
            self.data = [s for s in self.data if s.task_id in wanted]

        self._apply_shuffle()
        self.data_index = {s.task_id: s for s in self.data}
        self._loaded = True
        logger.info(f"Loaded {len(self.data)} WebShop tasks")

    def _apply_shuffle(self) -> None:
        if not self.shuffle:
            return
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(self.data)
