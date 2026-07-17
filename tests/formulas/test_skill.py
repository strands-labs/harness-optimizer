"""Tests for SkillFormula."""

import pytest
from strands import Skill
from strands.hooks.events import BeforeInvocationEvent

from strands_harness_optimizer.formulas import SkillFormula


def make_skill(name="math-solver", description="desc", instructions="instr"):
    return Skill(name=name, description=description, instructions=instructions)


class TestSkillFormulaInit:
    def test_name_includes_skill_name(self):
        formula = SkillFormula(make_skill())
        assert formula.name == "skill_formula:math-solver"

    def test_trigger_timings(self):
        formula = SkillFormula(make_skill())
        assert formula.trigger_timings == [BeforeInvocationEvent]

    def test_requires_at_least_one_tunable(self):
        with pytest.raises(ValueError):
            SkillFormula(make_skill(), tune_description=False, tune_instructions=False)


class TestSkillFormulaProcess:
    def test_process_returns_skill_under_skills_key(self):
        skill = make_skill()
        formula = SkillFormula(skill)
        result = formula.process({"system_prompt": "x", "messages": []})
        assert result == {"skills": [skill]}


class TestSkillFormulaTunableParams:
    def test_instructions_only(self):
        # Default construction tunes instructions only.
        formula = SkillFormula(make_skill(instructions="i0"))
        assert formula.get_tunable_params() == {"instructions": "i0"}

    def test_description_only(self):
        formula = SkillFormula(
            make_skill(description="d0"),
            tune_description=True,
            tune_instructions=False,
        )
        assert formula.get_tunable_params() == {"description": "d0"}

    def test_both(self):
        formula = SkillFormula(
            make_skill(description="d0", instructions="i0"),
            tune_description=True,
            tune_instructions=True,
        )
        assert formula.get_tunable_params() == {"description": "d0", "instructions": "i0"}


class TestSkillFormulaUpdateParams:
    def test_updates_instructions_in_place(self):
        skill = make_skill(instructions="old")
        formula = SkillFormula(skill)
        formula.update_params({"instructions": "new"})
        assert skill.instructions == "new"

    def test_updates_description_when_tuned(self):
        skill = make_skill(description="old")
        formula = SkillFormula(skill, tune_description=True, tune_instructions=False)
        formula.update_params({"description": "new"})
        assert skill.description == "new"

    def test_ignores_untuned_key(self):
        skill = make_skill(description="orig")
        formula = SkillFormula(skill)  # instructions only
        formula.update_params({"description": "changed"})
        assert skill.description == "orig"

    def test_ignores_missing_key(self):
        skill = make_skill(instructions="orig")
        formula = SkillFormula(skill)
        formula.update_params({})
        assert skill.instructions == "orig"
