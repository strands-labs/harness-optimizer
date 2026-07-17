"""
AppWorld dataset — the task rows only.

A minimal map-style dataset built directly on strands_harness_optimizer's Dataset
(so it's DataLoader/Sampler-compatible). It loads tasks from CSVs (fast) or the
appworld package. Running/evaluating a task lives in AppWorldEnvironment
(dataset/environment.py) — the dataset is just the data.
"""

import csv
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from strands_harness_optimizer.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class AppWorldDataSample:
    """AppWorld task data sample."""

    id: str = ""
    input: str = ""
    task_id: str = ""
    instruction: str = ""
    supervisor: str = ""
    app_descriptions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.input:
            self.input = self.instruction

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppWorldDataSample":
        return cls(
            id=data.get("id", ""),
            input=data.get("input", ""),
            task_id=data.get("task_id", ""),
            instruction=data.get("instruction", ""),
            supervisor=data.get("supervisor", ""),
            app_descriptions=data.get("app_descriptions", {}),
            metadata=data.get("metadata", {}),
        )

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __getitem__(self, item):
        return getattr(self, item)


class AppWorldDataset(Dataset[AppWorldDataSample]):
    """AppWorld benchmark dataset (map-style, lazily loaded).

    Loads on first access. ``get_execute_fn`` runs one task against a fresh AppWorld
    environment server; ``get_evaluate_fn`` turns the run result into an eval result.
    """

    name = "appworld"
    description = "AppWorld benchmark for evaluating agents on app-based tasks"

    def __init__(
        self,
        task_ids: List[str] | None = None,
        remote_apis_port: int = 9000,
        split: str = "train",
        csv_path: str | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = None,
    ):
        self.task_ids = task_ids
        self.remote_apis_port = remote_apis_port
        self.split = split
        self.csv_path = csv_path
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        self.data: List[AppWorldDataSample] = []
        self.data_index: Dict[str, AppWorldDataSample] = {}
        self._loaded = False

    # --- map-style Dataset surface (DataLoader-compatible) ---

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def __len__(self) -> int:
        self.ensure_loaded()
        return len(self.data)

    def __getitem__(self, idx: int) -> AppWorldDataSample:
        self.ensure_loaded()
        return self.data[idx]

    def get_data_by_id(self, task_id: str) -> AppWorldDataSample:
        self.ensure_loaded()
        return self.data_index[task_id]

    # --- loading ---

    def load(self) -> None:
        """Populate self.data from CSVs (if csv_path set) or the appworld package."""
        if self.csv_path:
            splits = os.listdir(self.csv_path) if self.split == "all" else [f"{self.split}.csv"]
            for csv_file in splits:
                self._load_csv(os.path.join(self.csv_path, csv_file))
            if self.task_ids:
                self.data = [s for s in self.data if s.task_id in self.task_ids]
        else:
            self._load_from_appworld()

        self._apply_shuffle()
        self.data_index = {s.task_id: s for s in self.data}
        self._loaded = True
        logger.info(f"Loaded {len(self.data)} AppWorld tasks")

    def _load_csv(self, csv_file_path: str) -> None:
        """Read a split CSV; JSON-decode embedded cells (supervisor, app_descriptions, ...)."""
        logger.info(f"Loading AppWorld tasks from CSV: {csv_file_path}")

        def deserialize(value):
            if not value:
                return value
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        with open(csv_file_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.data.append(
                    AppWorldDataSample.from_dict({k: deserialize(v) for k, v in row.items()})
                )

    def _load_from_appworld(self) -> None:
        try:
            from appworld import AppWorld, load_task_ids
        except ImportError:
            raise ImportError(
                "The 'appworld' package is required to load tasks from the environment. "
                "Install it, or pass csv_path to load from CSV."
            )
        task_ids = self.task_ids or load_task_ids(self.split)
        logger.info(f"Loading {len(task_ids)} AppWorld tasks from environment")
        for task_id in task_ids:
            world = AppWorld(task_id=task_id, experiment_name=f"load_{task_id}")
            task = world.task
            self.data.append(
                AppWorldDataSample(
                    id=task_id,
                    task_id=task_id,
                    instruction=task.instruction,
                    supervisor=task.supervisor,
                    app_descriptions=getattr(task, "app_descriptions", {}),
                    metadata={"remote_apis_port": self.remote_apis_port},
                )
            )
            world.__exit__(None, None, None)

    def _apply_shuffle(self) -> None:
        if not self.shuffle:
            return
        if self.shuffle_seed is not None:
            random.seed(self.shuffle_seed)
        random.shuffle(self.data)
