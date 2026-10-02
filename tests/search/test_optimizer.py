"""CandidateSearchOptimizer's contract, driven through the ordinary Trainer on synthetic score tables.

Each test states a situation the greedy loop gets wrong, and asserts the search gets it right: a
proposal can be declined, the seed can win, a rejected proposal costs no selection rollouts, and an
empty evaluation is an error rather than a result. The rest pin down what running under the Trainer
adds: one epoch is one step, the child runs on the parent's samples, a rejected edit is undone, and
the formula ends at the answer.
"""

from __future__ import annotations

import json

import pytest

from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.datamodels import Rollout
from strands_harness_optimizer.search import (
    CandidateSearchOptimizer,
    EvaluationRecord,
    EvaluationStore,
    SearchSampler,
    forbid_values,
    minimal_view,
    replay_sampler,
    select_by_mean,
    stratified_sampler,
    strict_improvement,
)
from strands_harness_optimizer.trainer import Trainer
from tests.search.helpers import (
    DictFormula,
    ListDataset,
    ScoreReward,
    ScriptedOptimizer,
    TableEngine,
)

ITEMS = ["i1", "i2"]
SEL = ["s1", "s2"]


def build(
    engine,
    proposals,
    *,
    steps=1,
    parent=None,
    formula=None,
    view=None,
    gate=None,
    store=None,
    guard=None,
    sampler_policy=None,
    items=ITEMS,
):
    formula = formula if formula is not None else DictFormula({"document": ""})
    proposer = ScriptedOptimizer(formula, proposals)
    search = CandidateSearchOptimizer(
        formula,
        proposer,
        engine,
        ScoreReward(),
        ListDataset(SEL).samples,
        parent_selection=parent or select_by_mean(),
        final_selection=select_by_mean(),
        gate=gate or strict_improvement(),
        proposal_guard=guard,
        rollout_view=view,
        store=store if store is not None else EvaluationStore(),
    )
    data = ListDataset(items)
    sampler = SearchSampler(data, search, sampler_policy) if sampler_policy else None
    loader = DataLoader(data, batch_size=len(items), sampler=sampler)
    trainer = Trainer(formula, search, ScoreReward(), search.engine, loader, n_epochs=steps)
    return search, trainer


def run(engine, proposals, **kw):
    search, trainer = build(engine, proposals, **kw)
    trainer.fit()
    return search, search.finalize()


def test_a_worse_proposal_is_declined_and_the_seed_is_returned():
    """The situation the greedy loop cannot express: the seed was already the best answer."""
    engine = TableEngine(
        table={("c0", "i1"): [1.0], ("c0", "i2"): [1.0], ("c0", "s1"): [1.0], ("c0", "s2"): [1.0]},
        default={"c1": [0.0]},
    )
    search, res = run(engine, [{"document": "worse"}])
    assert [it.admitted for it in res.iterations] == [False]
    assert res.best.candidate_id == "c0"
    assert len(res.pool) == 1, "a declined proposal must not enter the population"
    assert search.formula.get_tunable_params()["document"] == "", "the formula ends at the seed"


def test_one_epoch_is_one_step():
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    search, res = run(engine, [{"document": f"d{k}"} for k in range(3)], steps=3)
    assert [it.iteration for it in res.iterations] == [1, 2, 3]
    assert len(search.proposer.seen_params) == 3, "every epoch must propose; none may be a no-op"


def test_a_rejected_edit_is_undone_before_the_next_epoch():
    """The proposer edits the shared formula; the search must put the parent back."""
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    search, _ = run(engine, [{"document": "bad"}, {"document": "worse"}], steps=2)
    assert search.proposer.seen_params == [{"document": ""}, {"document": ""}]


def test_a_rejected_proposal_costs_no_selection_rollouts():
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    run(engine, [{"document": "worse"}])
    selection_calls = [c for c in engine.calls if c[1] == "selection"]
    assert [c[0] for c in selection_calls] == ["c0"], "only the seed's sweep should be paid for"


def test_an_improvement_is_admitted_and_evaluated_on_the_selection_set():
    engine = TableEngine(table={}, default={"c0": [0.0], "c1": [1.0]})
    search, res = run(engine, [{"document": "better"}])
    assert [it.admitted for it in res.iterations] == [True]
    assert res.best.candidate_id == "c1"
    assert search.store.has("c1", "selection")
    assert res.iterations[0].selection_mean == pytest.approx(1.0)
    assert search.formula.get_tunable_params()["document"] == "better"


def test_feedback_gain_that_does_not_transfer_still_admits_but_does_not_win():
    """The measured failure mode: better on the batch it was shown, worse on held-out items."""
    engine = TableEngine(
        table={
            ("c0", "i1"): [0.0],
            ("c0", "i2"): [0.0],
            ("c1", "i1"): [1.0],
            ("c1", "i2"): [1.0],
            ("c0", "s1"): [1.0],
            ("c0", "s2"): [1.0],
            ("c1", "s1"): [0.0],
            ("c1", "s2"): [0.0],
        }
    )
    search, res = run(engine, [{"document": "memorised"}])
    assert res.iterations[0].admitted is True, "the gate only sees the batch, so it admits"
    assert res.best.candidate_id == "c0", "the selector, which never saw the batch, declines it"
    assert search.formula.get_tunable_params()["document"] == ""


def test_the_parent_of_the_second_step_can_be_the_seed_again():
    """Reaching back past an admitted candidate is what a population is for."""
    engine = TableEngine(
        table={
            ("c0", "i1"): [0.0],
            ("c0", "i2"): [1.0],
            ("c0", "s1"): [1.0],
            ("c0", "s2"): [1.0],
            ("c1", "s2"): [0.0],
        },
        default={"c1": [1.0], "c2": [1.0]},
    )
    search, res = run(engine, [{"document": "a"}, {"document": "b"}], steps=2)
    assert [it.parent_id for it in res.iterations] == ["c0", "c0"]
    assert search.proposer.seen_params[1] == {"document": ""}, "c0 keeps the better selection mean"
    feedback_runs = [c[:2] for c in engine.calls if c[1].startswith("feedback")]
    assert feedback_runs[2] == ("c0", "feedback:2"), "the Trainer ran the reselected seed"


def test_step_scoped_feedback_keeps_the_gate_comparing_one_batch():
    """A candidate that was a child then becomes a parent must not carry its old batch along."""
    engine = TableEngine(table={}, default={"c0": [0.0], "c1": [1.0], "c2": [1.0]})
    search, res = run(engine, [{"document": "a"}, {"document": "b"}], steps=2)
    assert search.store.per_item("c1", "feedback:1") == {"i1": 1.0, "i2": 1.0}
    assert search.store.per_item("c1", "feedback:2") == {"i1": 1.0, "i2": 1.0}
    assert res.iterations[1].gate["role"] == "feedback:2"


def test_a_rejected_child_does_not_lend_its_scores_to_a_later_namesake():
    """Ids follow the pool size, so the rejected c1 and the next admitted child share a name.

    Step 1 runs i1+i2 and the first c1 scores 0 -> rejected. Step 2 runs i2 only, and the second c1
    scores 1 -> admitted. Its history must not include the rejected document's 0 on i1.
    """
    engine = TableEngine(
        table={("c1", "i1"): [0.0], ("c1", "i2"): [0.0]},
        default={"c0": [0.5], "c1": [1.0]},
    )
    search, trainer = build(
        engine,
        [{"document": "bad"}, {"document": "good"}],
        sampler_policy=replay_sampler({1: ["i1", "i2"], 2: ["i2"]}),
    )
    trainer.fit()
    assert search.store.has("c1~rejected:1", "feedback:1")
    assert not search.store.has("c1", "feedback:1")
    engine.table = {}
    trainer.fit()
    assert search.iterations[1].admitted
    assert search.store.latest_per_item("c1", "feedback") == {"i2": 1.0}


def test_relabel_rewrites_a_persisted_store(tmp_path):
    path = tmp_path / "records.jsonl"
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    run(engine, [{"document": "worse"}], store=EvaluationStore(path=str(path)))
    lines = path.read_text().splitlines()
    assert {json.loads(line)["candidate_id"] for line in lines} == {"c0", "c1~rejected:1"}
    assert len(EvaluationStore.load(str(path))) == len(lines)


def test_the_child_runs_on_the_samples_the_parent_ran_on():
    """Full samples, not bare ids — an engine that builds its prompt from the sample needs them."""
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [1.0]})
    run(engine, [{"document": "x"}])
    parent_call, child_call = [s for s in engine.samples_seen if s[0]["role"] == "feedback:1"]
    assert [s["candidate_id"] for s in parent_call] == ["c0", "c0"]
    assert [s["candidate_id"] for s in child_call] == ["c1", "c1"]
    strip = lambda ss: [(s["item_id"], s["prompt"]) for s in ss]  # noqa: E731
    assert strip(parent_call) == strip(child_call) == [("i1", "question i1"), ("i2", "question i2")]
    assert child_call[0]["params"] == {"document": "x"}, "params travel with the request"


def test_the_formula_is_materialised_before_each_request():
    seen = []
    formula = DictFormula({"document": ""})

    class Watching(TableEngine):
        def generate_batch(self, data_samples):
            seen.append((data_samples[0]["candidate_id"], formula.get_tunable_params()["document"]))
            return super().generate_batch(data_samples)

    engine = Watching(table={}, default={"c0": [0.0], "c1": [1.0]})
    run(engine, [{"document": "a"}, {"document": "b"}], steps=2, formula=formula)
    docs = {"c0": "", "c1": "a", "c2": "b"}
    assert seen and all(doc == docs[cid] for cid, doc in seen), seen


def test_a_formula_that_ignores_the_update_is_an_error():
    """Otherwise the engine would silently execute a different candidate than the one requested."""

    class Stubborn(DictFormula):
        """Takes the proposer's edit, then refuses to be put back to the empty seed."""

        def update_params(self, params):
            if params.get("document"):
                super().update_params(params)

    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    with pytest.raises(RuntimeError, match="did not take parameter"):
        run(engine, [{"document": "child"}], formula=Stubborn({"document": ""}))


def test_an_empty_evaluation_raises_rather_than_reading_as_a_result():
    class Empty(TableEngine):
        def generate_batch(self, data_samples):
            return iter(())

    with pytest.raises(RuntimeError, match="no rollouts"):
        run(Empty(table={}), [{"document": "x"}])


def test_the_trainer_must_be_given_the_wrapped_engine():
    class Plain(TableEngine):
        """An engine that does not need the search's tags, like a local agent engine."""

        def generate_batch(self, data_samples):
            for s in data_samples:
                yield Rollout(data_sample=dict(s), messages=[], metrics={"score": 1.0})

    engine = Plain(table={})
    formula = DictFormula({"document": ""})
    search = CandidateSearchOptimizer(
        formula,
        ScriptedOptimizer(formula, [{"document": "x"}]),
        engine,
        ScoreReward(),
        ListDataset(SEL).samples,
    )
    loader = DataLoader(ListDataset(ITEMS), batch_size=2)
    with pytest.raises(RuntimeError, match="optimizer.engine"):
        Trainer(formula, search, ScoreReward(), engine, loader, n_epochs=1).fit()


def test_a_guarded_proposal_costs_no_rollouts():
    engine = TableEngine(table={}, default={"c0": [0.0], "c1": [1.0]})
    search, res = run(engine, [{"document": "the answer is BRCA1"}], guard=forbid_values(["BRCA1"]))
    assert res.iterations[0].gate["blocked_by_guard"] is True
    assert not [c for c in engine.calls if c[0] == "c1"], "a blocked child must not be run"
    assert res.best.candidate_id == "c0"
    assert search.formula.get_tunable_params()["document"] == ""


# ------------------------------------------------------------------------- what the proposer sees
def test_the_proposer_is_fed_the_parent_rollouts_without_the_search_tags():
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [1.0]})
    search, _ = run(engine, [{"document": "x"}])
    (fed,) = search.proposer.seen_rollouts
    assert [ro.data_sample["item_id"] for ro in fed] == ["i1", "i2"]
    assert all({"candidate_id", "role", "params"}.isdisjoint(ro.data_sample) for ro in fed)
    assert all(ro.data_sample["prompt"] for ro in fed), "the sample itself still travels"


def test_minimal_view_withholds_the_reference_unless_asked():
    class WithAnswer(TableEngine):
        def generate_batch(self, data_samples):
            for ro in super().generate_batch(data_samples):
                ro.data_sample["answer"] = "BRCA1"
                ro.metadata = {"eval_result": {"gold": "BRCA1"}}
                yield ro

    for reveal in (False, True):
        engine = WithAnswer(table={}, default={"c0": [1.0], "c1": [1.0]})
        search, res = run(engine, [{"document": "x"}], view=minimal_view(reveal_reference=reveal))
        (fed,) = search.proposer.seen_rollouts
        assert all(("answer" in ro.data_sample) is reveal for ro in fed)
        assert all(ro.metadata == {} for ro in fed), "metadata can carry the answer; it is dropped"
        assert res.iterations[0].notes["view"]["reveals_reference"] is reveal


def test_evaluate_returns_records_and_the_rollouts_they_came_from():
    engine = TableEngine(table={}, default={"c0": [1.0]})
    search, _ = build(engine, [{"document": "x"}])
    p = search.evaluate(search.pool[0], ListDataset(ITEMS).samples, "feedback:0")
    assert p.records and len(p.rollouts) == len(p.records)
    assert {r.item_id for r in p.records} == {ro.data_sample["item_id"] for ro in p.rollouts}


def test_result_serialises_to_something_reportable():
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    _, res = run(engine, [{"document": "x"}])
    d = res.to_json()
    assert d["best"] == "c0" and d["steps"] == 1 and d["rollouts_used"] > 0
    assert d["iterations"][0]["gate"]["basis"] == "own"


# ------------------------------------------------------------------------------ construction
def test_the_proposer_must_be_a_formula_optimizer_on_the_same_formula():
    formula = DictFormula({"document": ""})
    with pytest.raises(TypeError, match="FormulaOptimizer"):
        CandidateSearchOptimizer(formula, object(), TableEngine(table={}), ScoreReward(), [])
    with pytest.raises(ValueError, match="same formula"):
        CandidateSearchOptimizer(
            formula,
            ScriptedOptimizer(DictFormula({"document": ""}), []),
            TableEngine(table={}),
            ScoreReward(),
            [],
        )


# ---------------------------------------------------------------------------------- sampling
def test_the_sampler_policy_chooses_each_steps_batch():
    engine = TableEngine(table={}, default={"c0": [1.0], "c1": [0.0]})
    run(
        engine,
        [{"document": "x"}, {"document": "y"}],
        steps=2,
        items=["i1", "i2", "i3"],
        sampler_policy=replay_sampler({1: ["i3", "i1"], 2: ["i2"]}),
    )
    parent_batches = [
        [s["item_id"] for s in ss]
        for ss in engine.samples_seen
        if ss[0]["candidate_id"] == "c0" and ss[0]["role"].startswith("feedback")
    ]
    assert parent_batches == [["i3", "i1"], ["i2"]]


def test_item_scores_are_the_profile_overwritten_by_each_parent_and_never_a_child():
    engine = TableEngine(
        table={("c0", "i1"): [1.0], ("c0", "i2"): [0.0]}, default={"c0": [0.0], "c1": [1.0]}
    )
    search, trainer = build(engine, [{"document": "better"}], items=["i1", "i2", "i3"])
    profile = {"i1": 0.0, "i2": 1.0, "i3": 0.5}
    for item, score in profile.items():
        search.store.add(
            EvaluationRecord(
                candidate_id="c0", role="feedback:0", item_id=item, replicate=0, score=score
            )
        )
    assert search.item_scores == profile
    trainer.fit()
    assert search.iterations[0].admitted, "c1 scored 1.0 on the batch where c0 scored 1/3"
    assert search.item_scores == {
        "i1": 1.0,
        "i2": 0.0,
        "i3": 0.0,
    }, "the parent's step-1 scores overwrite the profile; the admitted child's 1.0s do not enter"


def test_an_admitted_parent_is_not_confined_to_the_batch_it_was_admitted_on():
    """The bug: bucketing by the parent's own scores only ever re-drew its admission batch."""
    items = [f"i{k}" for k in range(8)]
    engine = TableEngine(table={}, default={"c0": [0.0], "c1": [1.0], "c2": [0.0]})
    search, trainer = build(
        engine,
        [{"document": "a"}, {"document": "b"}],
        steps=2,
        items=items,
        sampler_policy=stratified_sampler({"never": 4}, seed=0),
    )
    trainer.fit()
    parent_batches = [
        sorted(s["item_id"] for s in ss)
        for ss in engine.samples_seen
        if ss[0]["role"].startswith("feedback")
        and ss[0]["candidate_id"]
        == search.iterations[int(ss[0]["role"].split(":")[1]) - 1].parent_id
    ]
    assert [it.parent_id for it in search.iterations] == ["c0", "c1"]
    assert len(parent_batches) == 2 and all(len(b) == 4 for b in parent_batches)
    assert parent_batches[0] != parent_batches[1], "step 2 must draw from the whole pool"
