"""
Multi-Agent Optimizer.

Extends ContrastiveReflectionOptimizer by adding the `swarm` tool from
strands_tools, allowing the orchestrator agent to dynamically create
sub-agents for detailed rollout analysis.

Based on the Strands Agents "Agents as Tools" pattern:
https://strandsagents.com/latest/user-guide/concepts/multi-agent/agents-as-tools/
"""

import logging
from typing import Optional

from jinja2 import Template
from strands_tools import swarm

from ...formulas import Formula
from ...utils.templates import create_template
from .contrastive_reflection import ContrastiveReflectionOptimizer

logger = logging.getLogger(__name__)


class MultiAgentOptimizer(ContrastiveReflectionOptimizer):
    """
    Optimizer that uses a multi-agent swarm for deep rollout analysis.

    Extends ContrastiveReflectionOptimizer by adding the `swarm` tool,
    allowing the orchestrator to dynamically create sub-agents for
    more thorough rollout analysis.

    Args:
        formula: The Formula whose parameters will be optimized.
        system_prompt_template: Jinja2 template for the orchestrator.
        task_message_template: Jinja2 template for the task message.
        rollout_analyzer_template: Optional template for the sub-agent
            system prompt. Passed as template variable `rollout_analyzer_prompt`.
        **kwargs: Additional arguments passed to BaseAgenticOptimizer.

    Example:
        optimizer = MultiAgentOptimizer(
            formula,
            system_prompt_template=load_builtin_template("..."),
            task_message_template=load_builtin_template("..."),
            rollout_analyzer_template="You are an expert at analyzing...",
        )
    """

    def __init__(
        self,
        formula: Formula,
        system_prompt_template: str | Template,
        task_message_template: str | Template,
        rollout_analyzer_template: str | Template | None = None,
        **kwargs,
    ):
        super().__init__(formula, system_prompt_template, task_message_template, **kwargs)

        if rollout_analyzer_template is None:
            self._rollout_analyzer_prompt = None
        elif isinstance(rollout_analyzer_template, str):
            self._rollout_analyzer_prompt = rollout_analyzer_template
        else:
            self._rollout_analyzer_prompt = rollout_analyzer_template.render()

        logger.info(
            f"Initialized MultiAgentOptimizer "
            f"(model={self.model_config.get('model_id', 'default')}, "
            f"n_sample_traces={self.n_sample_traces})"
        )

    def _get_extra_tools(self) -> list:
        """Add swarm tool for sub-agent creation."""
        return [swarm]

    def _run_reflection(self, traces_folder: str, params: dict) -> None:
        """Run the orchestrator with rollout_analyzer_prompt available."""
        import os

        template_vars = {
            "traces_folder": os.path.abspath(traces_folder),
            "params": params,
        }

        if self._rollout_analyzer_prompt:
            template_vars["rollout_analyzer_prompt"] = self._rollout_analyzer_prompt

        system_prompt = self._system_prompt_template.render(**template_vars)
        task_message = self._task_message_template.render(**template_vars)

        agent = self._create_agent(system_prompt)
        agent(task_message)
