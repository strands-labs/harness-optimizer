"""Callable policies for the search loop.

Kept as plain functions and small closures rather than a class hierarchy: there are three
decision points — parent selection, item sampling and the gate — each is a few lines, and every one
of them is something a caller may want to replace. A factory returns a closure when a policy needs
configuration. There is no stopping policy: the number of steps is the Trainer's `n_epochs`.

The defaults reproduce the loop this design replaces, so that a replay of recorded runs can be
compared decision by decision. Alternatives (a gate with a margin, a per-item Pareto selector)
are variants of the same signatures.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, Mapping, Sequence

from .candidate import Candidate
from .evaluation import DEFAULT_EXCLUDE, EvaluationStore, RolloutStatus

SELECTION_ROLE = "selection"
FEEDBACK_ROLE = "feedback"


# --------------------------------------------------------------------- parent selection
def select_by_mean(
    role: str = SELECTION_ROLE,
) -> Callable[[Sequence[Candidate], EvaluationStore], Candidate]:
    """Pick the candidate with the highest mean on `role`; first maximum wins ties.

    Candidates with no evaluation on that role are skipped rather than treated as zero, so a
    freshly admitted candidate cannot be selected before it has been measured.
    """

    def policy(pool: Sequence[Candidate], store: EvaluationStore) -> Candidate:
        scored = [
            (store.mean(c.candidate_id, role), i, c)
            for i, c in enumerate(pool)
            if store.has(c.candidate_id, role)
        ]
        if not scored:
            return pool[0]
        best = max(scored, key=lambda t: (t[0], -t[1]))
        return best[2]

    return policy


def force_sequence(
    ids: Sequence[str],
) -> Callable[[Sequence[Candidate], EvaluationStore], Candidate]:
    """Replay a recorded sequence of parents, one per call.

    Used to separate two questions when validating against a recorded run: does the gate and the
    bookkeeping behave, and separately, does the selection policy pick what the recorded run
    picked. Forcing answers the first without the second confounding it.
    """
    remaining = list(ids)

    def policy(pool: Sequence[Candidate], store: EvaluationStore) -> Candidate:
        want = remaining.pop(0)
        by_id = {c.candidate_id: c for c in pool}
        if want not in by_id:
            raise KeyError(f"forced parent {want!r} is not in the pool {sorted(by_id)}")
        return by_id[want]

    return policy


# ------------------------------------------------------------------- Pareto selection
def pareto_frontier(
    pool: Sequence[Candidate],
    store: EvaluationStore,
    items: Sequence[str],
    role: str = SELECTION_ROLE,
) -> dict:
    """The deterministic part of a per-item Pareto selection.

    A candidate that is best on even one item stays eligible. Selecting the best *average*
    instead is what lets the search settle into a local optimum: on the corpus this was measured
    against, the candidate with the best mean was frequently the seed, and following the mean
    alone would have stopped exploring after the first rejected proposal.

    Returns the win sets, the frontier after dropping dominated candidates, and the weight of
    each survivor (how many items it wins). Everything here is a function of the records, so it
    can be compared exactly against a recorded run even when the draw that follows cannot.

    A candidate is dominated when its win set is a strict subset of another's.
    """
    ids = [c.candidate_id for c in pool]
    per_item_scores = {cid: store.per_item(cid, role) for cid in ids}
    best = {i: max(per_item_scores[cid].get(i, 0.0) for cid in ids) for i in items}
    winners = {
        i: {cid for cid in ids if per_item_scores[cid].get(i, 0.0) >= best[i]} for i in items
    }
    union = set().union(*winners.values()) if winners else set(ids[:1])
    kept = set(union)
    for x in sorted(union):
        wx = {i for i in items if x in winners[i]}
        for y in sorted(union):
            if y == x:
                continue
            wy = {i for i in items if y in winners[i]}
            if wx < wy:
                kept.discard(x)
                break
    if not kept:
        kept = set(union)
    weights = {cid: sum(1 for i in items if cid in winners[i]) for cid in sorted(kept)}
    return {
        "winners": winners,
        "union": sorted(union),
        "frontier": sorted(kept),
        "weights": weights,
        "best_per_item": best,
    }


def select_from_pareto_frontier(
    items: Sequence[str],
    role: str = SELECTION_ROLE,
    rng: random.Random | None = None,
    on_frontier: Callable[[dict], None] | None = None,
):
    """Sample a parent from the Pareto frontier, weighted by items won.

    The draw is stochastic by design, so a recorded run's *choice* is not reproducible while its
    *frontier and weights* are. `on_frontier` receives the deterministic part on every call, which
    is what an acceptance test should compare.
    """
    r = rng or random.Random(0)

    def policy(pool: Sequence[Candidate], store: EvaluationStore) -> Candidate:
        if len(pool) == 1:
            return pool[0]
        info = pareto_frontier(pool, store, items, role)
        if on_frontier:
            on_frontier(info)
        by_id = {c.candidate_id: c for c in pool}
        total = sum(info["weights"].values()) or 1
        x = r.random() * total
        acc = 0
        for cid, w in sorted(info["weights"].items()):
            acc += w
            if x <= acc:
                return by_id[cid]
        return by_id[info["frontier"][0]]

    return policy


# ------------------------------------------------------------------------------ buckets
def bucket_items(per_item: Mapping[str, float]) -> dict[str, list[str]]:
    """never / sometimes / always, from a candidate's per-item scores.

    Items with no score are absent rather than bucketed as never: not having measured an item is
    different from having measured it and failed.
    """
    b: dict[str, list[str]] = {"never": [], "sometimes": [], "always": []}
    for item, v in per_item.items():
        b["never" if v <= 0.0 else ("always" if v >= 1.0 else "sometimes")].append(item)
    return b


def stratified_sampler(
    quota: Mapping[str, int], seed: int | None = None, role: str = FEEDBACK_ROLE
) -> Callable[..., list[str]]:
    """Draw a batch to a per-bucket quota, topping up from other buckets when one is short.

    Why stratify at all: tasks arrive with most items already solved — 45% to 87% in the corpus
    this was measured on — so a uniform draw spends its rollouts where there is no signal to
    reflect on. This is a different decision from balancing traces after the fact: it chooses
    where to spend rollouts in the first place.

    Items are bucketed by `scores` when given — under `SearchSampler` that is the search's running
    map, `CandidateSearchOptimizer.item_scores` — and otherwise by the parent's own latest feedback
    scores. The running map is what to use: bucketing by the parent's own scores alone confines an
    admitted child to the one batch it was admitted on, so every later step draws that batch again.
    An item with no score at all is drawn as "never": not yet measured is as worth showing the
    proposer as not yet solved, and leaving it out would shrink the pool to what was measured.
    """
    rng = random.Random(seed)

    def policy(
        iteration: int,
        parent: Candidate,
        store: EvaluationStore,
        items: Sequence[str],
        scores: Mapping[str, float] | None = None,
    ) -> list[str]:
        measured = (
            scores if scores is not None else store.latest_per_item(parent.candidate_id, role)
        )
        per_item = {i: 0.0 for i in items}
        per_item.update({k: v for k, v in measured.items() if k in per_item})
        buckets = bucket_items(per_item)
        out: list[str] = []
        for name, k in quota.items():
            pool = buckets.get(name) or []
            out.extend(rng.sample(pool, min(k, len(pool))))
        short = sum(quota.values()) - len(out)
        if short > 0:
            rest = [i for b in buckets.values() for i in b if i not in out]
            out.extend(rng.sample(rest, min(short, len(rest))))
        rng.shuffle(out)
        return out

    return policy


def replay_sampler(batches: Mapping[int, Sequence[str]]) -> Callable[..., list[str]]:
    """Serve a recorded batch per iteration.

    Required for decision-level comparison against a recorded run: the run's logs record the
    bucket counts but not which items were drawn, while the recorded sub-arm results do carry the
    item set. Re-sampling instead would change the batch and with it every accept/reject.
    """

    def policy(
        iteration: int,
        parent: Candidate,
        store: EvaluationStore,
        items: Sequence[str],
        scores: Mapping[str, float] | None = None,
    ) -> list[str]:
        if iteration not in batches:
            raise KeyError(f"no recorded batch for iteration {iteration}")
        return list(batches[iteration])

    return policy


# --------------------------------------------------------------------------------- gate
def strict_improvement(
    role: str = FEEDBACK_ROLE,
    margin: float = 0.0,
    basis: str = "own",
    exclude: Sequence[RolloutStatus] = DEFAULT_EXCLUDE,
) -> Callable[..., dict]:
    """Admit the child when it beats the parent on the shared batch by `margin`.

    `basis` decides which items each mean is taken over:

    * ``"own"`` — each candidate is averaged over the items *it* has scores for. This reproduces
      the loop this design replaces, and it is the default so that a replay of recorded runs can
      be compared decision by decision.
    * ``"intersection"`` — both means are taken over the items both candidates scored, which makes
      the comparison properly paired.

    The difference is not cosmetic: an item whose every replicate failed for the child but not for
    the parent enters one mean and not the other. Measured on the recorded corpus, that is worth
    one episode in 96 — 0.0104 — which is the same order as the margins these gates decide on. The
    honest summary is that ``"own"`` is what was run and ``"intersection"`` is what should probably
    be run; both are one word apart.
    """

    def policy(
        parent: Candidate,
        child: Candidate,
        store: EvaluationStore,
        batch: Sequence[str],
        role_override: str | None = None,
    ) -> dict:
        r = role_override or role
        p = store.per_item(parent.candidate_id, r, exclude)
        c = store.per_item(child.candidate_id, r, exclude)
        in_batch = set(batch)
        pk = [i for i in batch if i in p]
        ck = [i for i in batch if i in c]
        if basis == "intersection":
            shared = [i for i in batch if i in p and i in c]
            pk = ck = shared
        pm = (sum(p[i] for i in pk) / len(pk)) if pk else 0.0
        cm = (sum(c[i] for i in ck) / len(ck)) if ck else 0.0
        return {
            "admit": cm > pm + margin,
            "parent_mean": pm,
            "child_mean": cm,
            "role": r,
            "parent_items": len(pk),
            "child_items": len(ck),
            "basis": basis,
            "items_compared": len(set(pk) & set(ck)),
            "margin": margin,
        }

    return policy
