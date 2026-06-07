"""Tests for the MultiAgentOptimizer."""

import os

from context_alchemy.datamodels import Reward, Rollout
from context_alchemy.formulas import SystemPromptFormula
from context_alchemy.optimizers import MultiAgentOptimizer
from context_alchemy.utils import load_builtin_template


def test_multi_agent_writes_rollouts_and_cleans_up():
    """Rollouts are written to temp folder and cleaned up."""
    formula = SystemPromptFormula(system_prompt="original")
    optimizer = MultiAgentOptimizer(
        formula,
        system_prompt_template=load_builtin_template("multi_agent/system_prompt.jinja"),
        task_message_template=load_builtin_template("multi_agent/task_message_system_prompt.jinja"),
        rollout_analyzer_template=load_builtin_template("multi_agent/rollout_analyzer.jinja"),
        model_config={"model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    )
    optimizer.add_rollouts(
        [
            Rollout(
                messages=[{"role": "user", "content": "hello"}], data_sample={"prompt": "test"}
            ),
        ]
    )
    optimizer.add_rewards([Reward(reward=1.0)])

    # Sample and write traces to temp
    indices = optimizer._sample_traces()
    traces_folder = optimizer._write_traces_to_temp(indices)
    assert os.path.exists(traces_folder)
    assert len(os.listdir(traces_folder)) == 1

    # Cleanup
    optimizer._cleanup_temp()
    assert not os.path.exists(traces_folder)


def test_multi_agent_has_swarm_tool():
    """MultiAgentOptimizer adds swarm tool to the agent."""
    formula = SystemPromptFormula(system_prompt="original")
    optimizer = MultiAgentOptimizer(
        formula,
        system_prompt_template=load_builtin_template("multi_agent/system_prompt.jinja"),
        task_message_template=load_builtin_template("multi_agent/task_message_system_prompt.jinja"),
        model_config={"model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    )

    from strands_tools import swarm

    assert swarm in optimizer._get_extra_tools()


def test_multi_agent_rollout_analyzer_template_in_vars():
    """rollout_analyzer_prompt is passed as template variable."""
    formula = SystemPromptFormula(system_prompt="original")
    analyzer_prompt = "You are a rollout analyzer."
    optimizer = MultiAgentOptimizer(
        formula,
        system_prompt_template="{{ rollout_analyzer_prompt }}",
        task_message_template="task",
        rollout_analyzer_template=analyzer_prompt,
        model_config={"model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    )

    assert optimizer._rollout_analyzer_prompt == analyzer_prompt
