"""Tests for the ContrastiveReflectionOptimizer."""

import os
import pytest
from dotenv import load_dotenv
from strands.models import BedrockModel

from harness_optimizer.datamodels import Reward, Rollout
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.optimizers import ContrastiveReflectionOptimizer
from harness_optimizer.utils import load_builtin_template

load_dotenv()

has_aws_credentials = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
os.environ["BYPASS_TOOL_CONSENT"] = "true"


def test_writes_rollouts_and_cleans_up():
    """Rollouts are written to temp folder and cleaned up after step fails gracefully."""
    formula = SystemPromptFormula(system_prompt="original")
    optimizer = ContrastiveReflectionOptimizer(
        formula,
        system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
        task_message_template=load_builtin_template("contrastive_reflection/task_message_system_prompt.jinja"),
        model_config={"model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    )
    optimizer.add_rollouts([
        Rollout(data_sample={"prompt": "test"}, messages=[{"role": "user", "content": "hello"}]),
    ])
    optimizer.add_rewards([Reward(reward=1)])

    # Sample and write traces to temp
    indices = optimizer._sample_traces()
    traces_folder = optimizer._write_traces_to_temp(indices)
    assert os.path.exists(traces_folder)
    assert len(os.listdir(traces_folder)) == 1

    # Cleanup
    optimizer._cleanup_temp()
    assert not os.path.exists(traces_folder)


@pytest.mark.integration
@pytest.mark.skipif(not has_aws_credentials, reason="AWS credentials not available")
def test_contrastive_reflection_step():
    """Full step with real model: write rollouts → reflect → update formula."""
    formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")
    optimizer = ContrastiveReflectionOptimizer(
        formula,
        system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
        task_message_template=load_builtin_template("contrastive_reflection/task_message_system_prompt.jinja"),
        model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
    )

    # Add simulated rollouts with contrasting rewards
    optimizer.add_rollouts([
        Rollout(
            data_sample={"prompt": "What is 2+3?", "expected_answer": "5", "uuid": "uuid1"},
            messages=[{"role": "user", "content": "What is 2+3?"}, {"role": "assistant", "content": "2 + 3 = 5"}]
        ),
        Rollout(
            data_sample={"prompt": "What is 10-4?", "expected_answer": "6", "uuid": "uuid2"},
            messages=[{"role": "user", "content": "What is 10-4?"}, {"role": "assistant", "content": "6"}]
        ),
    ])
    optimizer.add_rewards([
        Reward(reward=0.0, metadata={"reason": "response contains extra text"}),
        Reward(reward=1.0, metadata={"reason": "exact match"}),
    ])

    optimizer.step()

    # Formula should be updated with a new prompt
    updated = formula.get_tunable_params()["system_prompt"]
    assert updated != "You are a helpful assistant."
    assert len(updated) > 0
