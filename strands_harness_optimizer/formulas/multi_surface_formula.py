"""
Composite formula over the three text surfaces an agent reads.

An agent's behaviour is governed by text in three places, each reaching the model
at a different moment:

    system prompt      before the task is read, and on every step after
    skill body         when the agent loads it, after its description matched
    tool description   when choosing the tool, filling its inputs, reading its result

A multi-surface optimizer decides, per finding, WHICH of the three should carry it.
Its output therefore touches up to three parameters at once, and
``FormulaOptimizer`` takes exactly one formula. ``MultiSurfaceFormula`` is that one
formula: it wraps a :class:`SystemPromptFormula`, a :class:`SkillLibraryFormula` and
a :class:`ToolDescriptionFormula`, merges their tunable parameters, and fans an
``update_params`` call out to whichever members the params name. A member whose key
is absent is left exactly as it was, which is the legitimate "no change on this
surface" outcome.

Nothing here is specific to the multi-surface optimizer. Any optimizer that wants
to tune several surfaces in one step, or any runtime that wants to apply all three
with one ``apply_to_agent`` call, can use it.
"""

import logging
from typing import Optional

from .formula import Formula
from .skill_library_formula import SkillLibraryFormula
from .system_prompt_formula import SystemPromptFormula
from .tool_description_formula import TOOL_DESCRIPTIONS_KEY, ToolDescriptionFormula

logger = logging.getLogger(__name__)

# Keys each member owns in a params dict. ``update_params`` routes by these.
_SYSTEM_PROMPT_KEYS = ("system_prompt",)
_SKILL_KEYS = ("decisions_dir", "skill_dir")
_TOOL_KEYS = (TOOL_DESCRIPTIONS_KEY,)


class MultiSurfaceFormula(Formula):
    """One formula over the system prompt, the skill library and the tool descriptions.

    Example:
        formula = MultiSurfaceFormula(
            system_prompt=SystemPromptFormula(system_prompt="You are a shopping agent."),
            skills=SkillLibraryFormula(skill_dir="./skills"),        # or None: cold start
            tool_descriptions=ToolDescriptionFormula(base={"search": "...", "click": "..."}),
        )
        formula.get_tunable_params()
        # {'system_prompt': '...', 'skill_dir': './skills', 'tool_descriptions': {}}

        # An optimizer that edited the prompt and one tool, and wrote skill decisions:
        formula.update_params({
            "system_prompt": "...",
            "decisions_dir": "./runs/step_0001/skills",
            "tool_descriptions": {"search": "..."},
        })

    The members stay reachable as ``formula.system_prompt``, ``formula.skills`` and
    ``formula.tool_descriptions``, so a runtime can still call
    ``formula.skills.materialize(...)`` or read ``formula.tool_descriptions.effective()``.

    Args:
        system_prompt: The prompt member. Required; every agent has a prompt.
        skills: The skill-library member. ``None`` builds a cold-start library.
        tool_descriptions: The tool-description member. ``None`` builds one with an
            empty base, which accepts any tool name (the toolset is then unknown).
    """

    def __init__(
        self,
        system_prompt: SystemPromptFormula,
        skills: Optional[SkillLibraryFormula] = None,
        tool_descriptions: Optional[ToolDescriptionFormula] = None,
    ):
        self.system_prompt = system_prompt
        self.skills = skills if skills is not None else SkillLibraryFormula()
        self.tool_descriptions = (
            tool_descriptions if tool_descriptions is not None else ToolDescriptionFormula({})
        )
        # Union of the members' timings, first occurrence wins, so a member that
        # fires on an extra event is still honoured.
        timings: list = []
        for member in self.members:
            for t in member.trigger_timings:
                if t not in timings:
                    timings.append(t)
        super().__init__("multi_surface_formula", timings)

    @property
    def members(self) -> list[Formula]:
        """The three member formulas, in the order ``process`` runs them."""
        return [self.system_prompt, self.skills, self.tool_descriptions]

    # ── Formula protocol ─────────────────────────────────────────────────────
    def process(self, context: dict, **kwargs) -> dict:
        """Run each member and return only the keys they changed.

        Members run in order and each sees the others' updates layered on the
        incoming context. A member that returns the context object it was given
        (its way of saying "nothing to do") contributes no keys, so the adapter
        writes back only what actually changed.
        """
        running = dict(context)
        changed: dict = {}
        for member in self.members:
            if not member.can_process(running):
                continue
            out = member.process(running, **kwargs)
            if out is running or out is context or not out:
                continue
            changed.update(out)
            running.update(out)
        return changed if changed else context

    def get_tunable_params(self) -> dict:
        """The members' parameters merged into one dict.

        Keys: ``system_prompt``, ``skill_dir``, ``tool_descriptions``.
        """
        params: dict = {}
        for member in self.members:
            params.update(member.get_tunable_params() or {})
        return params

    def snapshot(self) -> dict:
        """The members' state, deep enough to restore with :meth:`restore`."""
        return {
            "system_prompt": self.system_prompt.system_prompt,
            "skill_dir": self.skills.skill_dir,
            "members": dict(self.skills.members),
            "overrides": dict(self.tool_descriptions.overrides),
        }

    def restore(self, snapshot: dict) -> None:
        """Put the members back exactly as :meth:`snapshot` saw them."""
        self.system_prompt.system_prompt = snapshot["system_prompt"]
        self.skills.skill_dir = snapshot["skill_dir"]
        self.skills.members = dict(snapshot["members"])
        self.tool_descriptions.overrides = dict(snapshot["overrides"])

    def update_params(self, params: dict) -> None:
        """Fan the update out by key; a member with no key in ``params`` is untouched.

        All or nothing: members are updated in sequence, so a failure on the second
        (a strict tool formula rejecting a name, say) would otherwise leave the first
        already changed. On any exception the members are restored and it is
        re-raised.
        """
        snap = self.snapshot()
        try:
            self._update_members(params)
        except Exception:
            self.restore(snap)
            raise

    def _update_members(self, params: dict) -> None:
        routed = []
        sp = {k: params[k] for k in _SYSTEM_PROMPT_KEYS if k in params}
        if sp:
            self.system_prompt.update_params(sp)
            routed.append("system_prompt")
        sk = {k: params[k] for k in _SKILL_KEYS if k in params}
        if sk:
            self.skills.update_params(sk)
            routed.append("skills")
        td = {k: params[k] for k in _TOOL_KEYS if k in params}
        if td:
            self.tool_descriptions.update_params(td)
            routed.append("tool_descriptions")
        unknown = sorted(set(params) - set(_SYSTEM_PROMPT_KEYS + _SKILL_KEYS + _TOOL_KEYS))
        if unknown:
            logger.warning("MultiSurfaceFormula.update_params: ignoring unknown key(s) %s", unknown)
        logger.info(
            "MultiSurfaceFormula: updated %s", ", ".join(routed) or "nothing (no known keys)"
        )

    # ── conveniences ─────────────────────────────────────────────────────────
    def materialize(self, out_dir: str) -> str:
        """Write the resolved skill set to ``out_dir``; see ``SkillLibraryFormula``."""
        return self.skills.materialize(out_dir)
