"""Candidate is a snapshot: it must not be editable, and lineage must be recorded."""

import pytest

from strands_harness_optimizer.search.candidate import Candidate


def test_params_are_not_writable():
    c = Candidate.seed({"document": "a"})
    with pytest.raises(TypeError):
        c.params["document"] = "b"  # type: ignore[index]
    with pytest.raises(Exception):
        c.candidate_id = "other"  # type: ignore[misc]


def test_mutating_the_source_dict_does_not_reach_the_candidate():
    src = {"document": "a"}
    c = Candidate.seed(src)
    src["document"] = "b"
    assert c.params["document"] == "a", "the snapshot must be a copy, not a view"


def test_child_records_lineage_without_touching_the_parent():
    seed = Candidate.seed({"document": "a"})
    child = seed.child("c1", {"document": "ab"}, iteration=1)
    assert child.parent_id == "c0" and child.origin == "proposal"
    assert child.metadata["iteration"] == 1
    assert seed.params["document"] == "a", "generating a child must leave the parent alone"


def test_seed_is_labelled_as_such():
    assert Candidate.seed({"document": ""}).origin == "seed"


def test_size_reports_characters_per_parameter():
    c = Candidate.seed({"document": "abc", "other": 3})
    assert c.size() == {"document": 3, "other": None}
