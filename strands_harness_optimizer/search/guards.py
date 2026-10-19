"""Guards on a proposal, applied before it costs any rollouts.

A generator that is shown the reference answers will write them down; that is a rational response
to the objective, not misbehaviour. Measured on the corpus this design comes from, learned
documents contained verbatim dataset answers on five of eight tasks, and the resulting inflation
moved a headline result from +10.7 to +8.9 macro. The failure was silent: nothing errored, and
every reported number improved.

So a guard is cheap insurance, and it runs before evaluation because a rejected proposal should
cost nothing. What a guard is *not* is a correctness test — overlap alone is weak evidence. On one
task 43% of the dataset's answers appeared in the document with no differential gain at all,
because they were ordinary domain terms any competent document would mention. Treat a violation as
a reason to look, and keep the differential analysis (per-item, named versus not-named) as the
thing that decides whether a document earned its score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


@dataclass
class GuardVerdict:
    ok: bool
    violations: list[dict] = field(default_factory=list)
    guard: str = ""

    def summary(self) -> str:
        if self.ok:
            return f"{self.guard}: clean"
        which = ", ".join(sorted({str(v.get("match")) for v in self.violations})[:8])
        return f"{self.guard}: {len(self.violations)} violation(s) [{which}]"


def forbid_patterns(
    patterns: Sequence[str], name: str = "forbidden_patterns", flags: int = re.IGNORECASE
) -> Callable[[Mapping[str, object]], GuardVerdict]:
    """Reject a proposal that mentions any of these regular expressions.

    The intended use is an ablation: when an experiment removes a resource, a proposal that names
    the resource's path silently restores it, and what the experiment then measures is whether the
    document disclosed a path rather than what the model can do without the data.
    """
    compiled = [(p, re.compile(p, flags)) for p in patterns]

    def guard(params: Mapping[str, object]) -> GuardVerdict:
        violations = []
        for key, value in params.items():
            if not isinstance(value, str):
                continue
            for src, rx in compiled:
                m = rx.search(value)
                if m:
                    i = max(0, m.start() - 60)
                    violations.append(
                        {
                            "param": key,
                            "pattern": src,
                            "match": m.group(0),
                            "context": value[i : m.end() + 60].replace("\n", " "),
                        }
                    )
        return GuardVerdict(ok=not violations, violations=violations, guard=name)

    return guard


def forbid_values(
    values: Iterable[str],
    min_length: int = 3,
    max_allowed: int = 0,
    name: str = "dataset_answers",
    tokenizer: Callable[[str], Iterable[str]] | None = None,
) -> Callable[[Mapping[str, object]], GuardVerdict]:
    """Reject a proposal that contains more than `max_allowed` of the dataset's answer values.

    Matching is on whole tokens, upper-cased, so a value is not flagged because it happens to be a
    substring of an unrelated word. `min_length` drops values too short to be evidence of anything
    (single letters, small integers). `max_allowed` exists because zero is not always the right
    threshold: for a task whose answers are common domain terms, a handful of mentions is expected,
    and the recorded corpus has exactly that case.
    """
    wanted = {str(v).strip().upper() for v in values if len(str(v).strip()) >= min_length}
    tok = tokenizer or (lambda s: re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{1,}", s))

    def guard(params: Mapping[str, object]) -> GuardVerdict:
        hits: dict[str, list[str]] = {}
        for key, value in params.items():
            if not isinstance(value, str):
                continue
            present = {t.upper() for t in tok(value)}
            for v in sorted(wanted & present):
                hits.setdefault(v, []).append(key)
        if len(hits) <= max_allowed:
            return GuardVerdict(ok=True, violations=[], guard=name)
        return GuardVerdict(
            ok=False,
            guard=name,
            violations=[
                {"param": ks[0], "match": v, "pattern": "dataset answer value"}
                for v, ks in sorted(hits.items())
            ],
        )

    return guard


def all_of(
    *guards: Callable[[Mapping[str, object]], GuardVerdict], name: str = "all"
) -> Callable[[Mapping[str, object]], GuardVerdict]:
    """Run several guards; the proposal must pass every one. Verdicts are merged."""

    def guard(params: Mapping[str, object]) -> GuardVerdict:
        violations, failed = [], []
        for g in guards:
            v = g(params)
            if not v.ok:
                failed.append(v.guard)
                violations.extend({**x, "guard": v.guard} for x in v.violations)
        return GuardVerdict(
            ok=not violations, violations=violations, guard=name if not failed else "+".join(failed)
        )

    return guard
