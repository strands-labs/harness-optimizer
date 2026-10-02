"""Synthetic doubles for the search control plane.

Nothing here reads a dataset, a cluster, or a model. A "task" is a table of scores, which is
enough to exercise every decision the search makes and keeps these tests runnable by anyone.
"""

from __future__ import annotations

from typing import Iterator, Mapping, Sequence

from strands_harness_optimizer.data import Dataset
from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import Formula
from strands_harness_optimizer.optimizers import FormulaOptimizer
from strands_harness_optimizer.rollout_engines import AgentRolloutEngine


class DictFormula(Formula):
    """The smallest Formula that holds a parameter dict."""

    def __init__(self, params: Mapping[str, str] | None = None):
        super().__init__("dict_formula", ["before_invocation"])
        self.params = dict(params or {})

    def process(self, context: dict, **kwargs) -> dict:
        return {"system_prompt": self.params.get("document", "")}

    def get_tunable_params(self) -> dict:
        return dict(self.params)

    def update_params(self, params: dict) -> None:
        self.params.update(params)


class ListDataset(Dataset):
    """Samples carrying an item id and a prompt, so a test can tell full samples from bare ids."""

    def __init__(self, item_ids: Sequence[str]):
        self.samples = [{"item_id": i, "prompt": f"question {i}"} for i in item_ids]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class TableEngine(AgentRolloutEngine):
    """Serve scores from a table: (candidate_id, item_id) -> sequence of scores.

    A missing entry falls back to the candidate's default, so a test only has to state the cells
    it cares about. Statuses can be attached per cell to exercise the exclusion policy.
    """

    def __init__(
        self,
        table: Mapping[tuple[str, str], Sequence[float]],
        default: Mapping[str, Sequence[float]] | None = None,
        statuses: Mapping[tuple[str, str], Sequence[str]] | None = None,
    ):
        self.formula = None
        self.num_rollouts = 1
        self.table = dict(table)
        self.default = dict(default or {})
        self.statuses = dict(statuses or {})
        self.calls: list[tuple[str, str, int]] = []
        self.samples_seen: list[list[dict]] = []

    def ensure_sync_params(self) -> None:
        return None

    def generate_batch(self, data_samples: Sequence[Mapping]) -> Iterator[Rollout]:
        if data_samples:
            s0 = data_samples[0]
            self.calls.append((str(s0["candidate_id"]), str(s0["role"]), len(data_samples)))
            self.samples_seen.append([dict(s) for s in data_samples])
        for s in data_samples:
            cid, item = str(s["candidate_id"]), str(s["item_id"])
            scores = self.table.get((cid, item), self.default.get(cid, [0.0]))
            sts = self.statuses.get((cid, item), ["valid"] * len(scores))
            for score, st in zip(scores, sts):
                yield Rollout(
                    data_sample=dict(s),
                    messages=[],
                    metrics={"score": float(score), "status": st},
                    metadata={},
                )


class ScoreReward:
    def __call__(self, rollout: Rollout) -> Reward:
        m = rollout.metrics or {}
        return Reward(
            reward=float(m.get("score") or 0.0), metadata={"status": m.get("status") or "valid"}
        )


class ScriptedOptimizer(FormulaOptimizer):
    """Propose a scripted sequence of parameter dicts by editing the formula, like any optimizer.

    Records the parameters it started from (the parent the search materialised) and the rollouts it
    was fed, so a test can check both.
    """

    def __init__(self, formula: Formula, params: Sequence[Mapping[str, str]]):
        super().__init__(formula)
        self.params = list(params)
        self.seen_params: list[dict] = []
        self.seen_rollouts: list[list[Rollout]] = []

    def step(self) -> None:
        self.seen_params.append(dict(self.formula.get_tunable_params()))
        self.seen_rollouts.append(list(self._rollouts))
        self.formula.update_params(dict(self.params[len(self.seen_params) - 1]))
