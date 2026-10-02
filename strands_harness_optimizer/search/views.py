"""What the proposer is allowed to see.

The upstream reflection optimizer writes the whole data sample — reference answer included — into
files the proposer reads with a shell tool. Measured consequence on the corpus this design comes
from: learned documents contained verbatim dataset answers on five of eight tasks, one opening with a
section titled "Known gene -> answer table (use this first)", and correcting the resulting inflation
moved a headline result from +10.7 to +8.9 macro. The failure was silent for a day; nothing errored
and every reported number improved.

A view is a function `(rollouts, rewards) -> (rollouts, rewards)` that the search applies before the
proposer is fed, so "what the proposer sees" is an explicit choice rather than "whatever happens to
be in the rollout", and the choice is recorded in each step's notes. It is not a security boundary:
a proposer with shell access can read the corpus directly, and withholding on its own was measured
to be insufficient (see `minimal_view`). Detecting leakage and restricting tools remain application
concerns; this module only makes the decision explicit.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from ..datamodels import Reward, Rollout


def minimal_view(
    reference_key: str = "answer",
    reveal_reference: bool = False,
    keep: Sequence[str] = ("prompt", "prediction", "messages"),
):
    """Keep each rollout's item id, the `keep` fields of its data sample and its trajectory — and,
    only when asked, the reference answer.

    **The default withholds the reference.** Pass `reveal_reference=True` to include it, and expect
    the artifact to contain answer values if you do.

    Everything not listed is dropped, `rollout.metadata` included, because an engine may put an
    evaluator's output there and that can carry the answer. Rewards pass through unchanged: an
    objective that reads `Reward.metadata` still needs it.

    Withholding is necessary and, on its own, not sufficient — worth stating because the measured
    result is counter-intuitive. Denied the answer, the proposer copied the student's own predictions
    into the artifact, and on a task whose answer space is a shared candidate list those predictions
    are other instances' answers: ten answer values in the produced document against one when the
    answer was visible. Withholding a field from this view also cannot stop a proposer that reads the
    corpus through a shell tool.

    What did work was *telling* the proposer not to write answers down, plus a guard on the proposal
    before it costs any rollouts. Both belong to the application: the first is a template, the second
    a policy the search calls. This function only decides which fields travel.
    """
    fields = ("item_id", *keep) + ((reference_key,) if reveal_reference else ())

    def view(
        rollouts: Sequence[Rollout], rewards: Sequence[Reward]
    ) -> tuple[list[Rollout], list[Reward]]:
        out = []
        for ro in rollouts:
            src = ro.data_sample or {}
            out.append(
                dataclasses.replace(
                    ro, data_sample={k: src[k] for k in fields if k in src}, metadata={}
                )
            )
        return out, list(rewards)

    view.description = {
        "view": "minimal",
        "fields": list(fields),
        "reveals_reference": reveal_reference,
    }
    return view
