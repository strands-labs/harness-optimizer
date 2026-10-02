# Search

A [FormulaOptimizer](./optimizers.md) commits every edit it proposes: `step()` mutates the Formula in
place and returns nothing. That is what you want for greedy tuning against a dense reward. It is not
enough when an improvement measured on the batch you showed the proposer might not survive a fresh
draw of items — the loop has no point at which an edit can be declined, two edits compared, or the
original kept.

`CandidateSearchOptimizer` is a `FormulaOptimizer` that adds that layer. It runs under the ordinary
`Trainer`, and it uses any existing optimizer as its proposer.

## One epoch is one step

Each Trainer epoch runs the current parent on a batch, then calls `step()`, which:

1. **Proposes** a child: the proposer is fed the parent's rollouts and edits the formula.
2. **Guards** the proposal — a blocked proposal costs no rollouts.
3. **Runs the child on the same samples.** The comparison is paired by construction.
4. **Gates.** A rejected child is recorded and undone; the parent stays.
5. **Scores an admitted child** on the selection split, which the proposer never sees.
6. **Chooses the next parent** from the whole pool — possibly the seed — and puts it in the formula.

The number of steps is `n_epochs`; there is no separate budget or stopping rule.

## Minimal use

```python
from strands_harness_optimizer.trainer import Trainer
from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.optimizers import ContrastiveReflectionOptimizer
from strands_harness_optimizer.search import (
    CandidateSearchOptimizer, select_by_mean, strict_improvement,
)

proposer = ContrastiveReflectionOptimizer(formula, system_tpl, task_tpl)  # any FormulaOptimizer on this formula
search = CandidateSearchOptimizer(
    formula, proposer, engine, reward_fn,
    selection_samples=holdout,                            # list of samples, each with an "item_id"
    parent_selection=select_by_mean(),
    gate=strict_improvement(),
)

trainer = Trainer(formula, search, reward_fn, search.engine,   # note: search.engine
                  DataLoader(train, batch_size=24, shuffle=True), n_epochs=8)
trainer.fit()

result = search.finalize()   # puts result.best into the formula
result.best.params           # the winning parameters, which may be the seed's
result.iterations            # one record per step: gate means, admission, cost
```

Three things to get right:

- **Give the Trainer `search.engine`**, not the raw engine. The wrapper tags each sample with the
  candidate and role it belongs to, records the batch so the child runs on exactly the parent's
  inputs, and puts the parent into the formula before every request. Passing the raw engine fails on
  the first step.
- **Give the Trainer the same `reward_fn`.** The parent is scored by the Trainer and the child by the
  search; the gate compares the two.
- **Call `finalize()` after `fit()`.** Between steps the formula holds the next parent, which under a
  Pareto parent policy is a draw rather than the answer.

## The candidate pool

A `Candidate` is an immutable snapshot of a formula's parameters plus its lineage. The seed — the
formula's parameters when the search is constructed — stays in the pool for the whole search, so
"the original was already the best" is an outcome the search can return, not one it has to be rescued
from. A seed with empty parameters is a legitimate member of the pool.

## Choosing the batch

Which samples an epoch runs on is the DataLoader's decision. To let the search decide — for instance
to put unsolved items in front of the proposer — wrap a sampling policy in a `SearchSampler`:

```python
from strands_harness_optimizer.search import SearchSampler, stratified_sampler

policy = stratified_sampler({"never": 10, "sometimes": 10, "always": 4}, seed=0)
loader = DataLoader(train, batch_size=24, sampler=SearchSampler(train, search, policy))
```

The sampler asks the search for its current parent and records each time an epoch starts. A
stratified policy buckets items by the parent's earlier scores, so give it something to bucket: run
the seed over the pool once before training,
`search.evaluate(search.pool[0], [train[i] for i in range(len(train))], "feedback:0")`. Without that, every item starts in the
"never" bucket and the first draw is uniform.

## Per-item evaluation

`search.store` keeps one record per (candidate, role, item, replicate) rather than one number per
epoch:

```python
store.per_item(candidate_id, role="selection")   # {item_id: mean score}
store.mean(candidate_id, role="selection")
store.pass_at_k(candidate_id, role="selection")  # (value, k)
```

Feedback is recorded per step (`feedback:1`, `feedback:2`, ...) so the gate compares one batch. A
rejected child's records are moved to `cN~rejected:<step>`: candidate ids follow the pool size, so the
rejected child shares its id with the next admitted one, and its scores must not be read as that
candidate's.

Each record carries a status: `valid`, `timeout`, `execution_error`, `missing_output`. Execution
errors are excluded from scoring by default, so an infrastructure failure is not silently read as the
model getting the answer wrong.

## What the proposer sees

A reflective proposer reads trajectories, and an upstream reflection optimizer writes the whole data
sample — reference answer included — into the files it reads. A `rollout_view` decides what reaches
it:

```python
from strands_harness_optimizer.search import minimal_view

view = minimal_view(keep=("prompt", "prediction", "messages"))   # reference withheld
view = minimal_view(reveal_reference=True)                       # reference included
search = CandidateSearchOptimizer(..., rollout_view=view)
```

`minimal_view` keeps the item id, the listed sample fields and the trajectory, and drops everything
else, `rollout.metadata` included. The default is no view: rollouts pass through unchanged.

Shown the answers, a capable proposer writes them into the artifact — a rational response to the
objective, not misbehaviour — and the resulting improvement does not transfer to items it never saw.
Withholding is necessary and not sufficient: it cannot stop a proposer that reads the corpus through
a shell tool, and on a task whose answer space is a shared candidate list, a proposer denied the
answers will copy the model's own predictions instead, which are other items' answers. Pair it with an
instruction in your template and a guard:

```python
from strands_harness_optimizer.search import forbid_values

guard = forbid_values(reference_values, min_length=3)   # rejected before it costs rollouts
search = CandidateSearchOptimizer(..., proposal_guard=guard)
```

## Testing a search without paying for rollouts

`ReplayRolloutEngine` serves rollouts you have already recorded, keyed by (candidate, role, item), and
fails closed on a key it does not have. Search decisions are then deterministic, so a policy change
can be tested against recorded runs rather than by re-running an agent.
