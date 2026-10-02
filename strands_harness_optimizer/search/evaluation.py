"""Per-item evaluation records and their store.

The upstream Trainer keeps one scalar per epoch (`avg_reward`). Three things this design needs
are impossible from that: selection policies richer than the mean, difficulty buckets for
sampling, and paired comparison between two candidates on the same items. So evaluation is
retained at (candidate, role, item, replicate) granularity and every aggregate is computed
from those records.

`RolloutStatus` lives here rather than in `datamodels` deliberately: an absent reward must not
be silently equivalent to a valid run that answered incorrectly. Aggregation takes an explicit
`exclude` policy, because "which failures count against the candidate" is a decision, not a
detail. The default excludes EXECUTION_ERROR only — infrastructure failures measure the
cluster, not the harness — which reproduces the behaviour of the loop this replaces.
"""

from __future__ import annotations

import collections
import dataclasses
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class RolloutStatus(str, Enum):
    VALID = "valid"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    MISSING_OUTPUT = "missing_output"


DEFAULT_EXCLUDE: tuple[RolloutStatus, ...] = (RolloutStatus.EXECUTION_ERROR,)


@dataclass(frozen=True)
class EvaluationRecord:
    candidate_id: str
    role: str
    item_id: str
    replicate: int
    score: float
    status: RolloutStatus = RolloutStatus.VALID
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["metrics"] = dict(self.metrics)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "EvaluationRecord":
        return cls(
            candidate_id=d["candidate_id"],
            role=d["role"],
            item_id=d["item_id"],
            replicate=int(d["replicate"]),
            score=float(d["score"]),
            status=RolloutStatus(d.get("status", "valid")),
            metrics=dict(d.get("metrics") or {}),
        )


class EvaluationStore:
    """In-memory records with optional JSONL persistence.

    Deliberately not a database: the whole point of this layer is to make aggregation explicit
    and reusable, and a dict of lists does that. Persistence exists so a search can be inspected
    or resumed, not so it can be queried.
    """

    def __init__(self, path: str | None = None):
        self._records: list[EvaluationRecord] = []
        self._by: dict[tuple[str, str], list[EvaluationRecord]] = collections.defaultdict(list)
        self.path = path

    # ---- writing -----------------------------------------------------------------------
    def add(self, record: EvaluationRecord) -> None:
        self._records.append(record)
        self._by[(record.candidate_id, record.role)].append(record)
        if self.path:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(record.to_json()) + "\n")

    def extend(self, records: Iterable[EvaluationRecord]) -> None:
        for r in records:
            self.add(r)

    def relabel(self, candidate_id: str, role: str, new_candidate_id: str) -> int:
        """Move one (candidate, role)'s records under another candidate id. Returns how many moved.

        Candidate ids follow the pool size, so a rejected child carries the id the next admitted
        child will get. Left in place, its feedback records would be merged into that later
        candidate's history by `latest_per_item` — the scores of one document read as another's.
        The optimizer moves them aside once the gate declines, and the file is rewritten so a
        reloaded store agrees with the live one.
        """
        moved = [
            dataclasses.replace(r, candidate_id=new_candidate_id)
            for r in self._by.pop((candidate_id, role), [])
        ]
        if not moved:
            return 0
        self._records = [
            r for r in self._records if (r.candidate_id, r.role) != (candidate_id, role)
        ] + moved
        self._by[(new_candidate_id, role)].extend(moved)
        if self.path:
            self.dump(self.path)
        return len(moved)

    # ---- reading -----------------------------------------------------------------------
    def records(self, candidate_id: str, role: str) -> list[EvaluationRecord]:
        return list(self._by.get((candidate_id, role), ()))

    def has(self, candidate_id: str, role: str) -> bool:
        """Whether this candidate was already evaluated on this role.

        Selection sweeps were about a third of the rollout spend in the runs this design comes
        from, so the search asks before paying again.
        """
        return bool(self._by.get((candidate_id, role)))

    def per_item(
        self, candidate_id: str, role: str, exclude: Sequence[RolloutStatus] = DEFAULT_EXCLUDE
    ) -> dict[str, float]:
        """item -> mean score over that item's kept replicates. Items with none are absent."""
        ex = set(exclude)
        g: dict[str, list[float]] = collections.defaultdict(list)
        for r in self._by.get((candidate_id, role), ()):
            if r.status in ex:
                continue
            g[r.item_id].append(r.score)
        return {k: sum(v) / len(v) for k, v in g.items() if v}

    def mean(
        self,
        candidate_id: str,
        role: str,
        exclude: Sequence[RolloutStatus] = DEFAULT_EXCLUDE,
        items: Iterable[str] | None = None,
    ) -> float:
        """Mean over items of the per-item mean.

        Note the order: per item first, then over items. Averaging over episodes instead would
        weight items by how many replicates survived, which is exactly what differs between two
        arms whose tail episodes landed differently.
        """
        pi = self.per_item(candidate_id, role, exclude)
        keys = [k for k in (items if items is not None else pi) if k in pi]
        return (sum(pi[k] for k in keys) / len(keys)) if keys else 0.0

    def pass_at_k(
        self, candidate_id: str, role: str, exclude: Sequence[RolloutStatus] = DEFAULT_EXCLUDE
    ) -> tuple[float, int]:
        """Fraction of items solved at least once, and the k it was measured at.

        Reported with its k because k varies across sub-arms in real corpora, and comparing
        pass@k across different k is meaningless. A document that lifts the mean without lifting
        this has traded commitment for discrimination, which is a different claim.
        """
        ex = set(exclude)
        g: dict[str, list[float]] = collections.defaultdict(list)
        for r in self._by.get((candidate_id, role), ()):
            if r.status in ex:
                continue
            g[r.item_id].append(r.score)
        if not g:
            return 0.0, 0
        solved = sum(1 for v in g.values() if any(s > 0 for s in v))
        ks = {len(v) for v in g.values()}
        return solved / len(g), (max(ks) if ks else 0)

    def latest_per_item(
        self,
        candidate_id: str,
        role_prefix: str,
        exclude: Sequence[RolloutStatus] = DEFAULT_EXCLUDE,
    ) -> dict[str, float]:
        """Merge every role starting with `role_prefix`, most recent value winning per item.

        Feedback is recorded per iteration ("feedback:1", "feedback:2", ...) because a gate must
        compare the two candidates on that iteration's episodes alone — pooling a candidate's
        feedback across iterations averages over different batches and silently changes the
        comparison. Difficulty bucketing, on the other hand, wants the cumulative picture: the
        most recent measurement of each item under whatever candidate is current. That is what
        this returns, mirroring the recorded loop's running score map.
        """
        ex = set(exclude)
        latest: dict[str, list[float]] = {}
        for (cid, role), recs in self._by.items():
            if cid != candidate_id or not role.startswith(role_prefix):
                continue
            g: dict[str, list[float]] = collections.defaultdict(list)
            for r in recs:
                if r.status in ex:
                    continue
                g[r.item_id].append(r.score)
            for item, vals in g.items():
                if vals:
                    latest[item] = vals
        return {k: sum(v) / len(v) for k, v in latest.items() if v}

    def status_counts(self, candidate_id: str, role: str) -> dict[str, int]:
        c = collections.Counter(r.status.value for r in self._by.get((candidate_id, role), ()))
        return dict(c)

    def replicates(self, candidate_id: str, role: str) -> list[int]:
        c = collections.Counter(r.item_id for r in self._by.get((candidate_id, role), ()))
        return sorted(set(c.values()))

    def __len__(self) -> int:
        return len(self._records)

    # ---- persistence -------------------------------------------------------------------
    def dump(self, path: str) -> None:
        with open(path, "w") as fh:
            for r in self._records:
                fh.write(json.dumps(r.to_json()) + "\n")

    @classmethod
    def load(cls, path: str) -> "EvaluationStore":
        s = cls()
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    s.add(EvaluationRecord.from_json(json.loads(line)))
        return s
