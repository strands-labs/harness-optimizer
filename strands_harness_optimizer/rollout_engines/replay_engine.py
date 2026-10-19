"""ReplayRolloutEngine: serve rollouts that were recorded earlier.

Developing an optimizer against live rollouts is slow and expensive, and it makes acceptance
tests non-deterministic. If a previous run's rollouts are on disk, the search loop can be driven
from them at no inference cost and with exact reproducibility — which is what makes a
zero-tolerance test of the loop's decisions possible at all.

The engine is indexed by whatever the caller puts in the data sample; the convention used here is
(candidate_id, role, item_id), which the search populates. A missing key raises: a replay that
quietly substitutes a different rollout is worse than no replay.
"""

from __future__ import annotations

import collections
from typing import Any, Iterator, Mapping, Sequence

from ..datamodels import Rollout
from ..formulas import Formula
from .agent_rollout_engine import AgentRolloutEngine

ReplayKey = tuple[str, str, str]


class ReplayRolloutEngine(AgentRolloutEngine):
    """Serve recorded rollouts from an in-memory index.

    Args:
        formula: kept for interface compatibility; replay ignores its parameters, since the
            recorded rollouts were produced by whatever parameters were live at the time.
        index: (candidate_id, role, item_id) -> sequence of recorded rollouts, each a mapping
            carrying at least a score and optionally a status, messages and metrics.
        strict: raise on a missing key (default). With strict=False a missing key yields nothing,
            which lets a caller probe coverage before committing to a run.
    """

    def __init__(
        self,
        formula: Formula | None,
        index: Mapping[ReplayKey, Sequence[Mapping[str, Any]]],
        num_rollouts: int = 1,
        strict: bool = True,
    ):
        # AgentRolloutEngine.__init__ wants a formula; replay does not use it.
        self.formula = formula
        self.num_rollouts = num_rollouts
        self.index = {tuple(k): list(v) for k, v in index.items()}
        self.strict = strict
        self.served = collections.Counter()

    def ensure_sync_params(self) -> None:
        return None

    def generate_batch(self, data_samples: Sequence[Mapping[str, Any]]) -> Iterator[Rollout]:
        for sample in data_samples:
            key = (
                str(sample.get("candidate_id")),
                str(sample.get("role")),
                str(sample.get("item_id")),
            )
            recorded = self.index.get(key)
            if recorded is None:
                if self.strict:
                    raise KeyError(
                        f"no recorded rollouts for {key}; refusing to substitute. "
                        f"index holds {len(self.index)} keys"
                    )
                continue
            self.served[key] += 1
            for rec in recorded:
                yield Rollout(
                    data_sample={
                        **dict(sample),
                        **{
                            k: v
                            for k, v in rec.items()
                            if k not in ("score", "status", "messages", "metrics")
                        },
                    },
                    messages=list(rec.get("messages") or []),
                    metrics={
                        **dict(rec.get("metrics") or {}),
                        "score": rec.get("score"),
                        "status": rec.get("status"),
                    },
                    metadata={"replayed": True},
                )

    def coverage(self, data_samples: Sequence[Mapping[str, Any]]) -> dict:
        """How much of a requested batch the index can serve. Cheap pre-flight for a fixture set."""
        missing = [
            s
            for s in data_samples
            if (str(s.get("candidate_id")), str(s.get("role")), str(s.get("item_id")))
            not in self.index
        ]
        return {
            "requested": len(data_samples),
            "missing": len(missing),
            "missing_sample": missing[:5],
        }
