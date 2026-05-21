"""Tests for the RewardFunction base class."""

import pytest

from harness_optimizer.rewards import RewardFunction


class MatchReward(RewardFunction):
    def __call__(self, **kwargs):
        response = kwargs.get("response_text", "").strip()
        expected = kwargs.get("expected_answer", "").strip()
        match = response == expected
        return {
            "reward_value": 1.0 if match else 0.0,
            "reason": "exact match" if match else "mismatch",
        }


def test_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        RewardFunction()


def test_reward_returns_dict():
    """Reward function returns dict with reward_value and metadata."""
    reward_fn = MatchReward()
    result = reward_fn(response_text="5", expected_answer="5")
    assert result["reward_value"] == 1.0
    assert result["reason"] == "exact match"

    result = reward_fn(response_text="2 + 3 = 5", expected_answer="5")
    assert result["reward_value"] == 0.0
    assert result["reason"] == "mismatch"
