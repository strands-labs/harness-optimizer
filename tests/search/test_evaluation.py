"""The store's aggregation policies are where a scalar mean would hide things."""

import pytest

from strands_harness_optimizer.search.evaluation import (
    EvaluationRecord,
    EvaluationStore,
    RolloutStatus,
)


def rec(cid, role, item, rep, score, status=RolloutStatus.VALID):
    return EvaluationRecord(
        candidate_id=cid, role=role, item_id=item, replicate=rep, score=score, status=status
    )


def test_per_item_then_over_items_not_over_episodes():
    """Order matters: averaging episodes weights items by surviving replicate count."""
    s = EvaluationStore()
    s.extend(
        [
            rec("c0", "sel", "i1", 0, 1.0),
            rec("c0", "sel", "i1", 1, 1.0),
            rec("c0", "sel", "i1", 2, 1.0),
            rec("c0", "sel", "i2", 0, 0.0),
        ]
    )
    assert s.per_item("c0", "sel") == {"i1": 1.0, "i2": 0.0}
    assert s.mean("c0", "sel") == pytest.approx(0.5)  # per item, then over items
    episodes = [r.score for r in s.records("c0", "sel")]
    assert sum(episodes) / len(episodes) == pytest.approx(0.75)  # what NOT to report


def test_execution_errors_are_excluded_by_default_and_can_be_included():
    s = EvaluationStore()
    s.extend(
        [
            rec("c0", "sel", "i1", 0, 0.0, RolloutStatus.EXECUTION_ERROR),
            rec("c0", "sel", "i1", 1, 1.0),
        ]
    )
    assert s.per_item("c0", "sel") == {"i1": 1.0}, "infrastructure failure must not count against"
    assert s.per_item("c0", "sel", exclude=()) == {"i1": 0.5}


def test_an_item_that_only_failed_infrastructurally_is_absent_not_zero():
    s = EvaluationStore()
    s.extend(
        [
            rec("c0", "sel", "i1", 0, 0.0, RolloutStatus.EXECUTION_ERROR),
            rec("c0", "sel", "i2", 0, 1.0),
        ]
    )
    assert "i1" not in s.per_item("c0", "sel")
    assert s.mean("c0", "sel") == pytest.approx(1.0), "not measured is not the same as failed"


def test_timeout_counts_against_the_candidate():
    """A run that used its whole budget and answered nothing is a real failure of the harness."""
    s = EvaluationStore()
    s.extend([rec("c0", "sel", "i1", 0, 0.0, RolloutStatus.TIMEOUT)])
    assert s.per_item("c0", "sel") == {"i1": 0.0}


def test_pass_at_k_reports_its_k_and_separates_from_the_mean():
    s = EvaluationStore()
    # two items, k=2: one solved once, one never
    s.extend(
        [
            rec("c0", "sel", "i1", 0, 1.0),
            rec("c0", "sel", "i1", 1, 0.0),
            rec("c0", "sel", "i2", 0, 0.0),
            rec("c0", "sel", "i2", 1, 0.0),
        ]
    )
    assert s.mean("c0", "sel") == pytest.approx(0.25)
    assert s.pass_at_k("c0", "sel") == (pytest.approx(0.5), 2)


def test_has_reports_whether_a_role_was_paid_for():
    s = EvaluationStore()
    assert not s.has("c1", "selection")
    s.add(rec("c1", "selection", "i1", 0, 1.0))
    assert s.has("c1", "selection")


def test_mean_can_be_restricted_to_a_given_item_set():
    s = EvaluationStore()
    s.extend([rec("c0", "sel", "i1", 0, 1.0), rec("c0", "sel", "i2", 0, 0.0)])
    assert s.mean("c0", "sel", items=["i1"]) == pytest.approx(1.0)


def test_latest_per_item_merges_iteration_scoped_feedback_roles():
    s = EvaluationStore()
    s.add(rec("c0", "feedback:1", "i1", 0, 0.0))
    s.add(rec("c0", "feedback:2", "i1", 0, 1.0))
    s.add(rec("c0", "feedback:2", "i2", 0, 1.0))
    merged = s.latest_per_item("c0", "feedback")
    assert set(merged) == {"i1", "i2"}
    assert s.per_item("c0", "feedback:1") == {"i1": 0.0}, "per-role view stays scoped"


def test_roundtrip_through_jsonl(tmp_path):
    s = EvaluationStore()
    s.extend([rec("c0", "sel", "i1", 0, 1.0, RolloutStatus.TIMEOUT)])
    p = tmp_path / "records.jsonl"
    s.dump(str(p))
    back = EvaluationStore.load(str(p))
    assert len(back) == 1
    r = back.records("c0", "sel")[0]
    assert r.status is RolloutStatus.TIMEOUT and r.score == 1.0
