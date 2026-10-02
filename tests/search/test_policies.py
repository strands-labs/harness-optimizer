"""The three policies, each tested on the behaviour it exists for."""

import random

import pytest

from strands_harness_optimizer.search.candidate import Candidate
from strands_harness_optimizer.search.evaluation import (
    EvaluationRecord,
    EvaluationStore,
    RolloutStatus,
)
from strands_harness_optimizer.search.policies import (
    bucket_items,
    force_sequence,
    pareto_frontier,
    replay_sampler,
    select_by_mean,
    select_from_pareto_frontier,
    stratified_sampler,
    strict_improvement,
)


def store_with(per_candidate, role="selection"):
    s = EvaluationStore()
    for cid, items in per_candidate.items():
        for item, scores in items.items():
            for r, v in enumerate(scores if isinstance(scores, (list, tuple)) else [scores]):
                s.add(
                    EvaluationRecord(
                        candidate_id=cid, role=role, item_id=item, replicate=r, score=float(v)
                    )
                )
    return s


def pool(*ids):
    out = [Candidate.seed({"document": ""}, candidate_id=ids[0])]
    for i in ids[1:]:
        out.append(out[0].child(i, {"document": i}))
    return out


# ------------------------------------------------------------------ parent selection
def test_select_by_mean_skips_candidates_with_no_measurement():
    s = store_with({"c0": {"i1": 0.5}})
    assert (
        select_by_mean()(pool("c0", "c1"), s).candidate_id == "c0"
    ), "an unmeasured candidate must not be selectable — it would look like a zero"


def test_select_by_mean_breaks_ties_on_the_earlier_candidate():
    s = store_with({"c0": {"i1": 1.0}, "c1": {"i1": 1.0}})
    assert select_by_mean()(pool("c0", "c1"), s).candidate_id == "c0"


def test_pareto_frontier_keeps_a_candidate_that_wins_only_one_item():
    """The whole point of the frontier: a specialist stays eligible where the mean would drop it."""
    s = store_with(
        {"c0": {"i1": 1.0, "i2": 1.0, "i3": 0.0}, "c1": {"i1": 0.0, "i2": 0.0, "i3": 1.0}}
    )
    info = pareto_frontier(pool("c0", "c1"), s, ["i1", "i2", "i3"])
    assert info["frontier"] == ["c0", "c1"]
    assert info["weights"] == {"c0": 2, "c1": 1}
    assert (
        select_by_mean()(pool("c0", "c1"), s).candidate_id == "c0"
    ), "the mean prefers c0, which is exactly why following it alone stops exploring"


def test_pareto_frontier_drops_a_dominated_candidate():
    s = store_with({"c0": {"i1": 1.0, "i2": 1.0}, "c1": {"i1": 1.0, "i2": 0.0}})
    info = pareto_frontier(pool("c0", "c1"), s, ["i1", "i2"])
    assert info["frontier"] == ["c0"], "c1 wins a strict subset of what c0 wins"


def test_pareto_draw_is_weighted_by_items_won():
    s = store_with(
        {
            "c0": {"i1": 1.0, "i2": 1.0, "i3": 1.0, "i4": 0.0},
            "c1": {"i1": 0.0, "i2": 0.0, "i3": 0.0, "i4": 1.0},
        }
    )
    items = ["i1", "i2", "i3", "i4"]
    picks = [
        select_from_pareto_frontier(items, rng=random.Random(seed))(
            pool("c0", "c1"), s
        ).candidate_id
        for seed in range(60)
    ]
    assert picks.count("c0") > picks.count("c1") > 0, "both eligible, the 3-item winner more often"


def test_single_member_pool_needs_no_measurement():
    assert select_from_pareto_frontier(["i1"])(pool("c0"), EvaluationStore()).candidate_id == "c0"


def test_force_sequence_raises_when_the_forced_parent_is_absent():
    with pytest.raises(KeyError):
        force_sequence(["c9"])(pool("c0"), EvaluationStore())


# ------------------------------------------------------------------------- bucketing
def test_buckets_split_on_never_sometimes_always():
    b = bucket_items({"a": 0.0, "b": 0.5, "c": 1.0})
    assert b == {"never": ["a"], "sometimes": ["b"], "always": ["c"]}


def test_stratified_sampler_fills_its_quota_and_tops_up_from_elsewhere():
    s = store_with(
        {"c0": {f"n{i}": 0.0 for i in range(3)} | {f"a{i}": 1.0 for i in range(10)}},
        role="feedback",
    )
    items = [f"n{i}" for i in range(3)] + [f"a{i}" for i in range(10)]
    batch = stratified_sampler({"never": 5, "sometimes": 0, "always": 2}, seed=0)(
        1, pool("c0")[0], s, items
    )
    assert len(batch) == 7, "a short bucket is topped up rather than shrinking the batch"
    assert len([i for i in batch if i.startswith("n")]) == 3


def test_replay_sampler_refuses_an_iteration_it_has_no_record_for():
    with pytest.raises(KeyError):
        replay_sampler({1: ["i1"]})(2, pool("c0")[0], EvaluationStore(), ["i1"])


# ------------------------------------------------------------------------------ gate
def test_gate_admits_only_a_strict_improvement():
    s = store_with({"c0": {"i1": 0.5}, "c1": {"i1": 0.5}}, role="feedback")
    g = strict_improvement()(pool("c0", "c1")[0], pool("c0", "c1")[1], s, ["i1"])
    assert g["admit"] is False, "equal is not better"


def test_gate_margin_can_reject_a_small_gain():
    s = store_with({"c0": {"i1": 0.0}, "c1": {"i1": 0.25}}, role="feedback")
    p, c = pool("c0", "c1")
    assert strict_improvement()(p, c, s, ["i1"])["admit"] is True
    assert strict_improvement(margin=0.5)(p, c, s, ["i1"])["admit"] is False


def test_gate_basis_changes_the_comparison_when_an_item_is_missing_for_one_side():
    """`own` averages each candidate over its own items; `intersection` over the shared ones.

    The child here failed infrastructurally on the item it would have lost, so under `own` it looks
    perfect while under `intersection` it does not improve at all. One word apart, opposite verdict.
    """
    s = EvaluationStore()
    for item, sc in (("i1", 1.0), ("i2", 0.0)):
        s.add(EvaluationRecord("c0", "feedback", item, 0, sc))
    s.add(EvaluationRecord("c1", "feedback", "i1", 0, 1.0))
    s.add(EvaluationRecord("c1", "feedback", "i2", 0, 0.0, RolloutStatus.EXECUTION_ERROR))
    p, c = pool("c0", "c1")
    own = strict_improvement(basis="own")(p, c, s, ["i1", "i2"])
    inter = strict_improvement(basis="intersection")(p, c, s, ["i1", "i2"])
    assert own["parent_mean"] == pytest.approx(0.5) and own["child_mean"] == pytest.approx(1.0)
    assert own["admit"] is True
    assert inter["parent_mean"] == pytest.approx(1.0) and inter["child_mean"] == pytest.approx(1.0)
    assert inter["admit"] is False


def test_stratified_sampler_buckets_by_the_running_map_when_given_one():
    s = store_with({"c1": {"i1": 1.0, "i2": 1.0}}, role="feedback:1")
    draw = stratified_sampler({"never": 2}, seed=0)
    own = draw(2, pool("c1")[0], s, ["i1", "i2", "i3", "i4"])
    assert sorted(own) == ["i3", "i4"], "unmeasured items are drawn as never"
    running = draw(2, pool("c1")[0], s, ["i1", "i2", "i3", "i4"], scores={"i3": 1.0, "i4": 1.0})
    assert sorted(running) == ["i1", "i2"], "the running map, not the parent's own scores, decides"
