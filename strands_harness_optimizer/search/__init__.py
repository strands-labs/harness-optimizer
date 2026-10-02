"""Candidate search — keep several formula versions alive and choose between them on held-out data.

A plain `FormulaOptimizer.step()` mutates the formula in place and returns nothing, so nothing can
decline an edit, compare two edits, or keep the original around. `CandidateSearchOptimizer` is a
`FormulaOptimizer` that does: run under the ordinary `Trainer`, each epoch is one search step. A
`Candidate` is an immutable snapshot with lineage, an `EvaluationStore` keeps results per item
rather than one scalar per epoch, the child is run on the same items as its parent and gated on
that paired comparison, and only admitted candidates are scored on a split the proposer never saw.

Parent selection, item sampling and the gate are plain callables, so a different search strategy is
a different function rather than a subclass. `SearchSampler` hands item sampling to the DataLoader.
"""

from .candidate import Candidate
from .evaluation import EvaluationRecord, EvaluationStore, RolloutStatus
from .guards import GuardVerdict, all_of, forbid_patterns, forbid_values
from .optimizer import (
    CandidateSearchOptimizer,
    EvaluationPass,
    IterationRecord,
    SearchResult,
)
from .policies import (
    bucket_items,
    force_sequence,
    pareto_frontier,
    replay_sampler,
    select_by_mean,
    select_from_pareto_frontier,
    stratified_sampler,
    strict_improvement,
)
from .sampler import SearchSampler
from .views import minimal_view

__all__ = [
    "Candidate",
    "CandidateSearchOptimizer",
    "EvaluationPass",
    "EvaluationRecord",
    "EvaluationStore",
    "GuardVerdict",
    "IterationRecord",
    "RolloutStatus",
    "SearchResult",
    "SearchSampler",
    "all_of",
    "bucket_items",
    "forbid_patterns",
    "forbid_values",
    "force_sequence",
    "minimal_view",
    "pareto_frontier",
    "replay_sampler",
    "select_by_mean",
    "select_from_pareto_frontier",
    "stratified_sampler",
    "strict_improvement",
]
