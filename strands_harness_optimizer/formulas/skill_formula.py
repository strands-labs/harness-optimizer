"""
Built-in formula for Strands Agent Skill tuning.

SkillFormula exposes a Strands ``Skill``'s ``description`` and/or ``instructions``
as tunable parameters. Strands Skills use progressive disclosure: the
``AgentSkills`` plugin injects a skill's ``name`` + ``description`` into the system
prompt and loads the full ``instructions`` on demand when the agent activates the
skill. Both the description (the discovery hook) and the instructions (the payload)
are optimizable context.

The ``Skill``/``AgentSkills`` classes are a newer Strands feature, so they are
imported lazily in ``__init__`` — importing this module never fails on an older
(but still supported) ``strands-agents``; only constructing a ``SkillFormula``
requires a skills-capable version.
"""

import logging
from typing import TYPE_CHECKING

from strands.hooks.events import BeforeInvocationEvent

from .formula import Formula

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands import Skill

logger = logging.getLogger(__name__)


class SkillFormula(Formula):
    """Formula that manages a Strands ``Skill``'s text as tunable parameters.

    The Formula is the single interface the optimizer talks to. The control flow is:

        optimizer --update_params()--> Formula --process()--> adapter --> AgentSkills

    - ``update_params`` (called by the optimizer) mutates the ``Skill`` this
      formula owns.
    - ``process`` (called by the adapter each invocation) returns the skill under
      the ``"skills"`` context key; :class:`~strands_harness_optimizer.adapters.StrandsAdapter`
      pushes it into the agent's ``AgentSkills`` plugin via ``set_available_skills()``.

    This keeps the Formula authoritative over the skill definition instead of
    relying on the plugin and formula happening to share the same ``Skill``
    object — the adapter actively re-syncs the skill on every invocation.

    Example:
        from strands import Agent, AgentSkills, Skill

        skill = Skill(name="math-solver", description="Solve math problems.",
                      instructions="Solve the problem step by step.")
        formula = SkillFormula(skill)               # tunes instructions by default

        agent = Agent(plugins=[AgentSkills(skills=[skill])])
        apply_formulas_on_strands_agent(agent, [formula])

        formula.get_tunable_params()                # {'instructions': '...'}
        formula.update_params({"instructions": "improved..."})

    Args:
        skill: The Strands ``Skill`` instance this formula owns and optimizes.
        tune_description: If True, expose ``skill.description`` as a tunable param.
        tune_instructions: If True, expose ``skill.instructions`` as a tunable param.
    """

    def __init__(
        self,
        skill: "Skill",
        tune_description: bool = False,
        tune_instructions: bool = True,
    ):
        # Lazy import: only constructing a SkillFormula requires a strands
        # version that ships the AgentSkills plugin / Skill class.
        try:
            from strands import Skill  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SkillFormula requires a strands-agents version that provides the "
                "AgentSkills plugin (from strands import Skill, AgentSkills). "
                "Upgrade strands-agents, or use SystemPromptFormula / "
                "ContextExpansionFormula instead."
            ) from e

        if not (tune_description or tune_instructions):
            raise ValueError("SkillFormula must tune at least one of description/instructions.")

        super().__init__(f"skill_formula:{skill.name}", [BeforeInvocationEvent])
        self.skill = skill
        self.tune_description = tune_description
        self.tune_instructions = tune_instructions

    def process(self, context: dict, **kwargs) -> dict:
        """Hand the (possibly optimizer-updated) skill to the adapter.

        Returns the skill under the ``"skills"`` key; the StrandsAdapter syncs it
        into the AgentSkills plugin. ``system_prompt``/``messages`` are untouched.
        """
        return {"skills": [self.skill]}

    def get_tunable_params(self) -> dict:
        """Return the skill's tunable text (description and/or instructions)."""
        params = {}
        if self.tune_description:
            params["description"] = self.skill.description
        if self.tune_instructions:
            params["instructions"] = self.skill.instructions
        return params

    def update_params(self, params: dict) -> None:
        """Update the owned skill's text from a params dict."""
        if self.tune_description and "description" in params:
            self.skill.description = params["description"]
            logger.info(
                "Updated skill '%s' description (%d chars)",
                self.skill.name,
                len(params["description"]),
            )
        if self.tune_instructions and "instructions" in params:
            self.skill.instructions = params["instructions"]
            logger.info(
                "Updated skill '%s' instructions (%d chars)",
                self.skill.name,
                len(params["instructions"]),
            )
