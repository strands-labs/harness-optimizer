"""The configured objective: which scores count, and with what weight.

A ``Reward`` is one number plus free-form ``metadata``. The multi-surface
optimizer can show its agent more than one number -- a benchmark checker AND a
judge's Conciseness score, say -- when the reward function puts the components
somewhere agreed. This module is that agreement:

    Reward(
        reward=0.5,
        metadata={
            "scores": {"TaskSuccessScore": 1.0, "Conciseness": 0.0},
            "explanations": {"Conciseness": "The agent restated the cart three times ..."},
        },
    )

Rules:

- No weights configured: the objective is the single term
  ``TaskSuccessScore = Reward.reward``. ``metadata["scores"]`` is ignored.
- Weights configured: every named term must be present for every sampled reward.
  ``TaskSuccessScore`` always reads ``Reward.reward``; any other name reads
  ``metadata["scores"][name]``. A missing term raises BEFORE the agent runs.
- Scores present in ``metadata["scores"]`` but not named in the weights are never
  written to a trace or shown to the agent. Their presence does not make them
  part of the objective.
"""

from typing import Optional

from ...datamodels import Reward

TASK_SUCCESS_TERM = "TaskSuccessScore"
SCORES_KEY = "scores"
EXPLANATIONS_KEY = "explanations"


def normalize_weights(weights: Optional[dict]) -> dict[str, float]:
    """Return the configured weights, or the single-term default."""
    if not weights:
        return {TASK_SUCCESS_TERM: 1.0}
    out = {str(k): float(v) for k, v in weights.items()}
    if any(v < 0 for v in out.values()):
        raise ValueError(f"objective weights must be non-negative: {out}")
    if sum(out.values()) <= 0:
        raise ValueError(f"objective weights sum to zero: {out}")
    return out


def objective_terms(reward: Reward, weights: dict[str, float]) -> dict[str, float]:
    """The configured terms' values for one reward. Raises on a missing term."""
    scores = (getattr(reward, "metadata", None) or {}).get(SCORES_KEY) or {}
    terms: dict[str, float] = {}
    missing: list[str] = []
    for name in weights:
        if name == TASK_SUCCESS_TERM:
            value = reward.reward
        else:
            value = scores.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            missing.append(name)
        else:
            terms[name] = float(value)
    if missing:
        raise ValueError(
            f"objective term(s) {missing} have no numeric score in "
            f"Reward.metadata['{SCORES_KEY}'] (have: {sorted(scores)}). Every configured "
            "term must be scored for every sampled rollout; score the rollouts first or "
            "drop the term from objective_weights."
        )
    return terms


def weighted_total(terms: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of the terms. Weights need not sum to one."""
    wsum = sum(weights[k] for k in terms)
    if wsum <= 0:
        return 0.0
    return sum(weights[k] * terms[k] for k in terms) / wsum


def explanations_for(reward: Reward, weights: dict[str, float]) -> dict[str, str]:
    """The judge's explanation per configured term, where the reward carries one."""
    raw = (getattr(reward, "metadata", None) or {}).get(EXPLANATIONS_KEY) or {}
    return {k: str(raw[k]) for k in weights if raw.get(k)}
