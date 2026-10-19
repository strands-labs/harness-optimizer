"""Candidate: an immutable snapshot of a Formula's tunable parameters, with lineage.

The upstream loop mutates one Formula in place, so a proposal cannot be declined and an
earlier version cannot be revisited. A Candidate is the smallest thing that fixes that: the
parameter dict, frozen, plus who it came from.

Immutability is enforced here (the params mapping is copied and exposed read-only), but the
Formula that executes a candidate is still a mutable object shared with the rollout engine.
The search therefore materialises exactly one candidate at a time and asserts the formula's
params match before requesting any rollout; see CandidateSearchOptimizer._materialise.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Candidate:
    """A frozen parameter snapshot.

    Attributes:
        candidate_id: Stable identity, unique within one search.
        params: The tunable parameters, exactly as a Formula would accept them.
        parent_id: The candidate this was derived from, or None for the seed.
        origin: How it came about — "seed", "proposal", or a caller-defined label.
        metadata: Free-form provenance (iteration, generator, template, timings).
    """

    candidate_id: str
    params: Mapping[str, Any]
    parent_id: str | None = None
    origin: str = "proposal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", types.MappingProxyType(dict(self.params)))
        object.__setattr__(self, "metadata", types.MappingProxyType(dict(self.metadata)))

    @classmethod
    def seed(
        cls, params: Mapping[str, Any], candidate_id: str = "c0", **metadata: Any
    ) -> "Candidate":
        """The starting point, kept in the population for the whole search.

        Retaining it is not a formality: on two of the eight tasks this design was measured
        against, the empty document was the best candidate, and on a third the selector reached
        back to the seed after two accepted candidates and the rewrite became the best document.
        """
        return cls(
            candidate_id=candidate_id,
            params=params,
            parent_id=None,
            origin="seed",
            metadata=metadata,
        )

    def child(self, candidate_id: str, params: Mapping[str, Any], **metadata: Any) -> "Candidate":
        return Candidate(
            candidate_id=candidate_id,
            params=params,
            parent_id=self.candidate_id,
            origin="proposal",
            metadata=metadata,
        )

    def size(self) -> dict:
        """Character counts per parameter — the cheapest signal that a proposal grew or shrank."""
        return {k: len(v) if isinstance(v, str) else None for k, v in self.params.items()}
