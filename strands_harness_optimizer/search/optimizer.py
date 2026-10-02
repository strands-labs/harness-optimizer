"""CandidateSearchOptimizer: candidate search as one optimization method under the Trainer.

A plain `FormulaOptimizer.step()` commits whatever it proposes. This optimizer keeps a population
instead: each step proposes a child of the current parent, runs the child on the same items the
Trainer just ran the parent on, and admits it only if the gate says it beat the parent on that
paired comparison. Admitted children are scored on a selection split the proposer never sees, and
the next parent is chosen from the whole pool — which may be the seed.

One Trainer epoch is one search step:

    Trainer:  rollouts = engine(batch)          # the current parent, tagged feedback:<step>
              optimizer.add_rollouts / add_rewards
              optimizer.step():
                  child  = proposer.step() on rollout_view(rollouts), read back from the formula
                  guard(child)                   # a blocked proposal costs no rollouts
                  run child on the same samples  # paired by construction
                  gate(parent, child)
                      admitted -> run child on the selection split; add to the pool
                  parent = parent_selection(pool) and materialise it for the next epoch

So the number of steps is the Trainer's `n_epochs`; there is no separate budget that could leave a
step with nothing to do. After `fit()`, call `finalize()`: it puts the best pool member into the
formula and returns the search record. Until then the formula holds the *next parent*, which under a
Pareto parent policy is a draw, not the answer.

The Trainer must be given `optimizer.engine`, not the raw engine. That wrapper tags each sample with
the candidate and role it belongs to, which is how engines that key on them (replay, a cluster
launcher naming its arms) know what they are running, and it records the batch so the child can be
run on exactly the same inputs. It also materialises the parent before every request, so which
candidate is executing never depends on what ran last.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, NamedTuple, Sequence

from ..datamodels import Reward, Rollout
from ..formulas import Formula
from ..optimizers.optimizer import FormulaOptimizer
from ..rewards import RewardFunction
from ..rollout_engines import AgentRolloutEngine
from .candidate import Candidate
from .evaluation import EvaluationRecord, EvaluationStore, RolloutStatus
from .policies import FEEDBACK_ROLE, SELECTION_ROLE, select_by_mean, strict_improvement

# Keys the engine wrapper adds to every sample. `params` travels with the request because an engine
# that looks a candidate up elsewhere can get it wrong: a child is not in the pool until it is
# admitted, and an engine that reached into the pool silently executed the seed for a whole step.
TAG_KEYS = ("item_id", "candidate_id", "role", "params")


class EvaluationPass(NamedTuple):
    """One evaluation of one candidate on one role: the records stored, and the rollouts they came
    from. Both are needed — the store answers "how did it score", the rollouts are what the
    proposer reads."""

    records: list[EvaluationRecord]
    rollouts: list[Rollout]


@dataclass
class IterationRecord:
    iteration: int
    parent_id: str
    child_id: str | None
    batch_size: int
    gate: dict
    admitted: bool
    selection_mean: float | None
    rollouts_used: int
    wall_seconds: float
    notes: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    best: Candidate
    pool: list[Candidate]
    iterations: list[IterationRecord]
    rollouts_used: int

    def to_json(self) -> dict:
        return {
            "best": self.best.candidate_id,
            "pool": [
                {"id": c.candidate_id, "parent": c.parent_id, "origin": c.origin, "size": c.size()}
                for c in self.pool
            ],
            "iterations": [vars(i) for i in self.iterations],
            "rollouts_used": self.rollouts_used,
            "steps": len(self.iterations),
        }


class _TaggingEngine(AgentRolloutEngine):
    """What the Trainer runs: the wrapped engine, pointed at the current parent."""

    def __init__(self, owner: "CandidateSearchOptimizer", inner: AgentRolloutEngine):
        self.owner = owner
        self.inner = inner
        self.formula = owner.formula
        self.num_rollouts = getattr(inner, "num_rollouts", 1)

    def ensure_sync_params(self) -> None:
        self.inner.ensure_sync_params()

    def generate_batch(self, data_samples: list[dict]) -> Iterator[Rollout]:
        o = self.owner
        o._feedback_samples.extend(dict(s) for s in data_samples)
        yield from o._generate(o.parent, o.feedback_role, data_samples)


class CandidateSearchOptimizer(FormulaOptimizer):
    """Gated candidate search over a Formula's parameters, run one step per Trainer epoch.

    Args:
        formula: The Formula being optimized. Its parameters at construction are the seed.
        proposer: A `FormulaOptimizer` bound to the same formula. Each step it is fed the parent's
            rollouts (through `rollout_view`), its `step()` edits the formula, and the result is
            read back as the child. A rejected child is undone by re-materialising the parent.
        engine: The rollout engine. Give the Trainer `optimizer.engine`, which wraps this one.
        reward_fn: Scores the rollouts this optimizer runs itself. Give the Trainer the same
            function: the gate compares the parent, scored by the Trainer, with the child, scored
            here.
        selection_samples: The held-out split admitted candidates are scored on.
        parent_selection: (pool, store) -> the candidate to extend next. Default: best selection mean.
        final_selection: (pool, store) -> the answer `finalize()` returns. Default: parent_selection.
        gate: (parent, child, store, batch, role_override=) -> {"admit": bool, ...}.
            Default: `strict_improvement()`.
        proposal_guard: params -> GuardVerdict, checked before the child costs any rollouts.
        rollout_view: (rollouts, rewards) -> (rollouts, rewards), applied before the proposer sees
            anything — the place to withhold reference answers (see `minimal_view`). Default: pass
            through unchanged.
        status_from: (rollout, reward) -> RolloutStatus. Default: a `status` the engine or reward
            already reported, else valid.
        id_key: The sample field that identifies an item.
        store: Where evaluation records go. Pass one with a path to persist them.
        evaluate_seed_on_selection: Score the seed on the selection split at the first step, so the
            seed competes with every admitted child.
    """

    def __init__(
        self,
        formula: Formula,
        proposer: FormulaOptimizer,
        engine: AgentRolloutEngine,
        reward_fn: RewardFunction,
        selection_samples: Sequence[Mapping[str, Any]],
        *,
        parent_selection: Callable[[Sequence[Candidate], EvaluationStore], Candidate] | None = None,
        final_selection: Callable[[Sequence[Candidate], EvaluationStore], Candidate] | None = None,
        gate: Callable[..., dict] | None = None,
        proposal_guard: Callable[[Mapping[str, Any]], Any] | None = None,
        rollout_view: (
            Callable[
                [Sequence[Rollout], Sequence[Reward]], tuple[Sequence[Rollout], Sequence[Reward]]
            ]
            | None
        ) = None,
        status_from: Callable[[Rollout, Reward], RolloutStatus] | None = None,
        id_key: str = "item_id",
        store: EvaluationStore | None = None,
        evaluate_seed_on_selection: bool = True,
    ):
        super().__init__(formula)
        if not isinstance(proposer, FormulaOptimizer):
            raise TypeError(f"proposer must be a FormulaOptimizer, got {type(proposer).__name__}")
        if proposer.formula is not formula:
            raise ValueError(
                "the proposer must be bound to the same formula as the search; its step() result "
                "is read back from that formula"
            )
        self.proposer = proposer
        self.reward_fn = reward_fn
        self.engine = _TaggingEngine(self, engine)
        self.id_key = id_key
        self.selection_samples = [dict(s) for s in selection_samples]
        self.parent_selection = parent_selection or select_by_mean()
        # Which candidate to extend and which to return are separate decisions. They coincide by
        # default, but a replay forces the parent while still wanting the real selector to pick the
        # answer, and a caller may extend greedily while returning the most robust member.
        self.final_selection = final_selection or self.parent_selection
        self.gate = gate or strict_improvement()
        self.proposal_guard = proposal_guard
        self.rollout_view = rollout_view
        self.status_from = status_from or _default_status
        self.store = store if store is not None else EvaluationStore()
        self.evaluate_seed_on_selection = evaluate_seed_on_selection

        self.pool: list[Candidate] = [Candidate.seed(formula.get_tunable_params())]
        self.iterations: list[IterationRecord] = []
        self.rollouts_used = 0
        self.iteration = 1
        self._feedback_samples: list[dict] = []
        self.parent = self.parent_selection(self.pool, self.store)
        self._materialise(self.parent)

    # ------------------------------------------------------------------------ properties
    @property
    def feedback_role(self) -> str:
        """The role the current step's feedback is recorded under.

        Scoped per step: a candidate can be a child in one step and the parent in the next, and
        pooling both batches under one role would average over different items and change the
        gate's comparison.
        """
        return f"{FEEDBACK_ROLE}:{self.iteration}"

    @property
    def item_scores(self) -> dict[str, float]:
        """The running per-item score map a sampling policy buckets items by.

        Starts from any `feedback:0` records — a profiling pass run with `evaluate()` before the
        first epoch — and is overwritten, step by step, with each step's parent scores on its batch.
        A child's scores do not enter it: the map describes the parents the search has actually
        extended. This is the map the recorded loop kept, and it is what lets a later step draw
        from the whole pool rather than from the batch its parent was admitted on.
        """
        scores: dict[str, float] = {}
        for c in self.pool:
            scores.update(self.store.per_item(c.candidate_id, f"{FEEDBACK_ROLE}:0"))
        for it in self.iterations:
            scores.update(self.store.per_item(it.parent_id, f"{FEEDBACK_ROLE}:{it.iteration}"))
        return scores

    @property
    def selection_items(self) -> list[str]:
        return [self._item_id(s) for s in self.selection_samples]

    # ------------------------------------------------------------------------- internals
    def _item_id(self, sample: Mapping[str, Any]) -> str:
        if self.id_key not in sample:
            raise KeyError(f"sample has no {self.id_key!r} field (keys {sorted(sample)[:10]})")
        return str(sample[self.id_key])

    def _materialise(self, candidate: Candidate) -> None:
        """Point the shared Formula at this candidate, and check that it took."""
        self.formula.update_params(dict(candidate.params))
        now = self.formula.get_tunable_params()
        for k, v in candidate.params.items():
            if now.get(k) != v:
                raise RuntimeError(
                    f"formula did not take parameter {k!r} for candidate {candidate.candidate_id}; "
                    "the engine would have executed a different candidate than the one requested"
                )
        self.engine.ensure_sync_params()

    def _generate(
        self, candidate: Candidate, role: str, samples: Sequence[Mapping[str, Any]]
    ) -> list[Rollout]:
        self._materialise(candidate)
        params = dict(candidate.params)
        tagged = [
            {
                **dict(s),
                "item_id": self._item_id(s),
                "candidate_id": candidate.candidate_id,
                "role": role,
                "params": params,
            }
            for s in samples
        ]
        rollouts = list(self.engine.inner.generate_batch(tagged))
        if samples and not rollouts:
            raise RuntimeError(
                f"no rollouts returned for {candidate.candidate_id} on role {role!r} "
                f"({len(samples)} samples requested) — an empty evaluation must fail loudly rather "
                "than be read as a result"
            )
        return rollouts

    def _record(
        self,
        candidate: Candidate,
        role: str,
        rollouts: Sequence[Rollout],
        rewards: Sequence[Reward],
    ) -> list[EvaluationRecord]:
        seen: dict[str, int] = {}
        records = []
        for ro, rw in zip(rollouts, rewards):
            ds = ro.data_sample or {}
            if "item_id" not in ds and self.id_key not in ds:
                raise KeyError(
                    "a rollout came back without its item id; the engine must keep the sample it "
                    "was given in rollout.data_sample"
                )
            item = str(ds.get("item_id", ds.get(self.id_key)))
            rep = seen.get(item, 0)
            seen[item] = rep + 1
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    role=role,
                    item_id=item,
                    replicate=rep,
                    score=float(rw.reward),
                    status=self.status_from(ro, rw),
                    metrics={**dict(ro.metrics or {}), **dict(rw.metadata or {})},
                )
            )
        self.store.extend(records)
        self.rollouts_used += len(records)
        return records

    def evaluate(
        self, candidate: Candidate, samples: Sequence[Mapping[str, Any]], role: str
    ) -> EvaluationPass:
        """Run, score and record one candidate on these samples under this role.

        Public so a caller can add passes the step does not make itself — a profiling pass over the
        whole feedback pool before the first epoch, for instance. The formula is left at
        `candidate`; the Trainer's next request re-materialises the parent before running.
        """
        rollouts = self._generate(candidate, role, samples)
        rewards = [self.reward_fn(ro) for ro in rollouts]
        # The rollouts travel back with the records. Returning records alone loses everything the
        # proposer reflects on — transcripts, predictions, call counts — and silently: the proposer
        # still writes a document from the rest of its prompt, so nothing errors.
        return EvaluationPass(
            records=self._record(candidate, role, rollouts, rewards), rollouts=rollouts
        )

    def _batch_samples(self) -> list[dict]:
        """The samples the Trainer ran the parent on, once per item, in the order they arrived."""
        out, seen = [], set()
        for s in self._feedback_samples:
            i = self._item_id(s)
            if i not in seen:
                seen.add(i)
                out.append(s)
        return out

    def _propose(
        self, parent: Candidate, rollouts: Sequence[Rollout], rewards: Sequence[Reward]
    ) -> dict:
        """Let the proposer edit the formula from the parent, and read the edit back."""
        # The tags are the search's plumbing, not data. Left in, `params` alone would put the whole
        # parent document into every trace a reflective proposer reads.
        rollouts = [
            dataclasses.replace(
                ro,
                data_sample={
                    k: v
                    for k, v in (ro.data_sample or {}).items()
                    if k not in TAG_KEYS or k == "item_id"
                },
            )
            for ro in rollouts
        ]
        if self.rollout_view is not None:
            rollouts, rewards = self.rollout_view(rollouts, rewards)
        if len(rollouts) != len(rewards):
            raise RuntimeError(
                f"rollout_view returned {len(rollouts)} rollouts but {len(rewards)} rewards"
            )
        self._materialise(parent)
        self.proposer.zero()
        self.proposer.add_rollouts(list(rollouts))
        self.proposer.add_rewards(list(rewards))
        self.proposer.step()
        return dict(self.formula.get_tunable_params())

    def _advance(self) -> None:
        self.iteration += 1
        self.parent = self.parent_selection(self.pool, self.store)
        self._materialise(self.parent)

    # --------------------------------------------------------------------------- driving
    def step(self) -> None:
        it, role, parent = self.iteration, self.feedback_role, self.parent
        t0 = time.time()
        batch = self._batch_samples()
        if not self._rollouts or not batch:
            raise RuntimeError(
                "no feedback rollouts for this step. Give the Trainer `optimizer.engine` (the "
                "wrapper that records the batch), not the raw engine"
            )
        if len(self._rewards) != len(self._rollouts):
            raise RuntimeError(
                f"{len(self._rollouts)} rollouts but {len(self._rewards)} rewards; they must align"
            )
        batch_ids = [self._item_id(s) for s in batch]

        seed = self.pool[0]
        if self.evaluate_seed_on_selection and not self.store.has(
            seed.candidate_id, SELECTION_ROLE
        ):
            self.evaluate(seed, self.selection_samples, SELECTION_ROLE)

        self._record(parent, role, self._rollouts, self._rewards)
        child_params = self._propose(parent, self._rollouts, self._rewards)
        view_notes = {"view": getattr(self.rollout_view, "description", None)}
        child = parent.child(f"c{len(self.pool)}", child_params, iteration=it)

        def finish(gate: dict, admitted: bool, selection_mean: float | None, notes: dict) -> None:
            self.iterations.append(
                IterationRecord(
                    iteration=it,
                    parent_id=parent.candidate_id,
                    child_id=child.candidate_id if admitted else None,
                    batch_size=len(batch),
                    gate=gate,
                    admitted=admitted,
                    selection_mean=selection_mean,
                    rollouts_used=self.rollouts_used,
                    wall_seconds=round(time.time() - t0, 3),
                    notes=notes,
                )
            )
            self._advance()

        if self.proposal_guard is not None:
            verdict = self.proposal_guard(dict(child_params))
            if not verdict.ok:
                finish(
                    {
                        "admit": False,
                        "blocked_by_guard": True,
                        "guard": verdict.guard,
                        "violations": verdict.violations[:20],
                    },
                    False,
                    None,
                    {
                        **view_notes,
                        "guard_summary": verdict.summary(),
                        "rejected_child_size": child.size(),
                    },
                )
                return

        self.evaluate(child, batch, role)
        g = self.gate(parent, child, self.store, batch_ids, role_override=role)
        if not g["admit"]:
            self.store.relabel(child.candidate_id, role, f"{child.candidate_id}~rejected:{it}")
            finish(g, False, None, {**view_notes, "rejected_child_size": child.size()})
            return

        self.pool.append(child)
        self.evaluate(child, self.selection_samples, SELECTION_ROLE)
        finish(g, True, self.store.mean(child.candidate_id, SELECTION_ROLE), view_notes)

    def zero(self) -> None:
        super().zero()
        self._feedback_samples = []

    def finalize(self) -> SearchResult:
        """Put the best pool member into the formula and return the search record."""
        best = self.final_selection(self.pool, self.store)
        self._materialise(best)
        return SearchResult(
            best=best,
            pool=list(self.pool),
            iterations=list(self.iterations),
            rollouts_used=self.rollouts_used,
        )


def _default_status(rollout: Rollout, reward: Reward) -> RolloutStatus:
    """Read a status the rollout engine already put in metadata, else assume valid.

    Engines know why a run failed; a reward function usually does not. So the engine is expected
    to report it, and this default only keeps the optimizer usable when it does not.
    """
    for src in (reward.metadata or {}, rollout.metrics or {}, rollout.metadata or {}):
        s = src.get("status")
        if s:
            return s if isinstance(s, RolloutStatus) else RolloutStatus(str(s))
    return RolloutStatus.VALID
