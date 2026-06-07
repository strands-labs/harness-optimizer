"""Integration test for the optimization loop with a real Bedrock model.

Tests the full flow: vague prompt → agent gives verbose answer → low reward →
LLM-based optimizer analyzes rollouts and generates improved prompt →
same agent gives concise answer → higher reward.

Requires AWS credentials; skipped otherwise.
"""

import os
import re

import pytest
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel

from harness_optimizer.adapters import StrandsAdapter
from harness_optimizer.datamodels import Reward, Rollout
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.optimizers import FormulaOptimizer
from harness_optimizer.rewards import RewardFunction

load_dotenv()

has_aws_credentials = bool(os.environ.get("AWS_ACCESS_KEY_ID"))


class ConcisenessReward(RewardFunction):
    """Returns reward 1.0 if the response exactly matches the expected answer."""

    def __call__(self, rollout: Rollout) -> Reward:
        response_text = rollout.metadata.get("response_text", "").strip()
        expected = rollout.data_sample.get("expected_answer", "").strip()
        match = response_text == expected
        return Reward(
            reward=1.0 if match else 0.0,
            metadata={
                "response_text": response_text,
                "expected_answer": expected,
            },
        )


class LLMPromptOptimizer(FormulaOptimizer):
    """Uses an LLM to analyze rollouts and generate an improved system prompt."""

    def __init__(self, formula, model=None):
        super().__init__(formula)
        if model is None:
            model = BedrockModel(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                region_name="us-west-2",
            )
        self.model = model

    def step(self):
        # Build analysis of rollouts for the LLM
        analysis = []
        for i, (rollout, reward) in enumerate(zip(self._rollouts, self._rewards)):
            sample = rollout.data_sample
            analysis.append(
                f"Task: {sample.get('prompt', '')}\n"
                f"Expected answer: {sample.get('expected_answer', '')}\n"
                f"Agent response: {reward.metadata.get('response_text', '')}\n"
                f"Reward: {reward.reward}"
            )

        current_prompt = self.formula.get_tunable_params().get("system_prompt", "")
        optimization_prompt = (
            f"You are optimizing a system prompt for an LLM agent.\n\n"
            f'Current system prompt: "{current_prompt}"\n\n'
            f"Here are the agent's rollouts and their rewards:\n\n"
            + "\n\n".join(analysis)
            + f"\n\nAnalyze what patterns lead to high vs low rewards, "
            f"and generate an improved system prompt.\n\n"
            f"Reply with ONLY the new system prompt, nothing else."
        )

        optimizer_agent = Agent(model=self.model)
        response = optimizer_agent(optimization_prompt)
        new_prompt = response.message["content"][0]["text"].strip()

        self.formula.update_params({"system_prompt": new_prompt})


@pytest.mark.integration
@pytest.mark.skipif(not has_aws_credentials, reason="AWS credentials not available")
class TestOptimizationWithModel:

    @pytest.fixture
    def model(self):
        return BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name="us-west-2",
        )

    def test_llm_optimizer_improves_prompt(self, model):
        """Multi-round optimization: vague prompt → optimize over rounds → concise answers."""
        formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")
        agent = Agent(model=model, system_prompt="You are a helpful assistant.")
        adapter = StrandsAdapter()
        adapter.apply_to_agent([formula], agent)

        reward_fn = ConcisenessReward()
        optimizer = LLMPromptOptimizer(formula)  # optimizer holds formula reference

        data_samples = [
            {"prompt": "What is 2+3?", "expected_answer": "5"},
            {"prompt": "What is 10-4?", "expected_answer": "6"},
        ]

        num_rounds = 5
        round_rewards = []

        for round_idx in range(num_rounds):
            # Collect rollouts and rewards
            rollouts = []
            rewards = []
            for sample in data_samples:
                response = agent(sample["prompt"])
                response_text = response.message["content"][0]["text"]

                rollout = Rollout(
                    data_sample=sample,
                    messages=list(agent.messages),
                    metadata={"response_text": response_text},
                )
                reward = reward_fn(rollout)

                rollouts.append(rollout)
                rewards.append(reward)

            # Accumulate and step
            optimizer.add_rollouts(rollouts)
            optimizer.add_rewards(rewards)

            avg_reward = sum(r.reward for r in rewards) / len(rewards)
            round_rewards.append(avg_reward)

            # Stop early if perfect score
            if avg_reward == 1.0:
                break

            optimizer.step()
            optimizer.zero()

        # Verify: prompt was updated and rewards improved over rounds
        assert formula.get_tunable_params()["system_prompt"] != "You are a helpful assistant."
        assert round_rewards[-1] >= round_rewards[0]
