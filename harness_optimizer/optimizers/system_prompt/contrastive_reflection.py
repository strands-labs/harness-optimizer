"""
Contrastive Reflection Optimizer.

Uses a strands LLM agent with shell tools to analyze rollout traces,
compare successful vs failed trajectories, and generate optimized parameters.
"""

import logging
from typing import Optional

from jinja2 import Template

from ...formulas import Formula
from ...utils.templates import create_template
from .base_agentic_optimizer import BaseAgenticOptimizer

logger = logging.getLogger(__name__)


class ContrastiveReflectionOptimizer(BaseAgenticOptimizer):
    """
    Optimizer that uses an LLM agent with shell tools for contrastive learning.

    The optimizer:
    1. Writes accumulated rollouts to a temp folder as JSON files
    2. Optionally samples a subset (with stratified sampling by reward)
    3. Creates a strands Agent with shell tool and submit_optimized_params tool
    4. The agent analyzes traces contrastively (success vs failure)
    5. The agent writes optimized params to files and submits via tool

    Example:
        optimizer = ContrastiveReflectionOptimizer(formula)
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)
        optimizer.step()
        optimizer.zero()
    """

    def __init__(
        self,
        formula: Formula,
        system_prompt_template: str | Template,
        task_message_template: str | Template,
        **kwargs,
    ):
        """
        Initialize the contrastive reflection optimizer.

        Args:
            formula: The Formula whose parameters will be optimized.
            system_prompt_template: Jinja2 template (str or Template) for the
                reflection agent's system prompt. Use load_builtin_template()
                to load the default contrastive analysis template.
            task_message_template: Jinja2 template (str or Template) for the
                task message. Use load_builtin_template() to load a built-in
                template, or provide a custom one. Templates receive:
                traces_folder, params (dict of current tunable params).
            **kwargs: Additional arguments passed to BaseAgenticOptimizer
                (model_config, region_name, boto_config, n_sample_traces,
                stratified_sampling, success_threshold, max_output_chars,
                system_prompt_suffix).
        """
        super().__init__(formula, **kwargs)

        if isinstance(system_prompt_template, str):
            self._system_prompt_template = create_template(system_prompt_template)
        else:
            self._system_prompt_template = system_prompt_template

        if isinstance(task_message_template, str):
            self._task_message_template = create_template(task_message_template)
        else:
            self._task_message_template = task_message_template

        logger.info(
            f"Initialized ContrastiveReflectionOptimizer "
            f"(model={self.model_config.get('model_id', 'default')}, "
            f"n_sample_traces={self.n_sample_traces})"
        )

    def step(self) -> None:
        """
        Analyze accumulated rollouts and update Formula parameters.

        Writes rollouts to a temp folder, runs the reflection agent with
        shell and submit_optimized_params tools, and updates the Formula.
        """
        if not self._rollouts:
            logger.warning("No rollouts accumulated, skipping step")
            return

        try:
            # Sample traces and write to temp folder
            indices = self._sample_traces()
            traces_folder = self._write_traces_to_temp(indices)

            # Run reflection
            params = self.formula.get_tunable_params()
            self._run_reflection(traces_folder, params)

            # Read submitted params
            optimized_params = self._get_submitted_params()

            if optimized_params:
                self.formula.update_params(optimized_params)
                self._prompt_history.append(optimized_params)
                self._step_count += 1
                summary = ", ".join(f"{k} ({len(str(v))} chars)" for k, v in optimized_params.items())
                logger.info(f"Step {self._step_count}: updated params: {summary}")
            else:
                raise RuntimeError(
                    "Optimization agent did not call submit_optimized_params tool. "
                    "The agent must write optimized params to files and submit via the tool."
                )

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error during contrastive reflection: {e}") from e
        finally:
            self._cleanup_temp()

    def _run_reflection(self, traces_folder: str, params: dict) -> None:
        """Run the reflection agent to analyze traces."""
        import os

        template_vars = {
            "traces_folder": os.path.abspath(traces_folder),
            "params": params,
        }

        system_prompt = self._system_prompt_template.render(**template_vars)
        task_message = self._task_message_template.render(**template_vars)

        agent = self._create_agent(system_prompt)
        agent(task_message)

    # get_state() and load_state() inherited from BaseAgenticOptimizer
