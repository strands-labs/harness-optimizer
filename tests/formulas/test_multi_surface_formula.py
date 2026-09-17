"""Tests for MultiSurfaceFormula."""

import os

import pytest
from strands.hooks.events import AfterInvocationEvent, BeforeInvocationEvent

from strands_harness_optimizer.formulas import (
    MultiSurfaceFormula,
    SkillLibraryFormula,
    SystemPromptFormula,
    ToolDescriptionFormula,
)

FRONTMATTER = "---\nname: {name}\ndescription: Use when {desc}.\n---\n\n{body}\n"
BASE = {"search": "Search the catalog.", "click": "Click an element."}


def write_skill(root, name, desc="testing", body="Do the thing."):
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(FRONTMATTER.format(name=name, desc=desc, body=body))
    return d


def make(tmp_path, with_skills=True):
    skills = None
    if with_skills:
        write_skill(tmp_path / "lib", "alpha")
        skills = SkillLibraryFormula(str(tmp_path / "lib"))
    return MultiSurfaceFormula(
        system_prompt=SystemPromptFormula(system_prompt="base prompt"),
        skills=skills,
        tool_descriptions=ToolDescriptionFormula(BASE),
    )


class TestInit:
    def test_members_and_defaults(self):
        f = MultiSurfaceFormula(SystemPromptFormula(system_prompt="p"))
        assert isinstance(f.skills, SkillLibraryFormula) and f.skills.members == {}
        assert isinstance(f.tool_descriptions, ToolDescriptionFormula)
        assert f.tool_descriptions.base == {}
        assert [m.name for m in f.members] == [
            "system_prompt_formula",
            "skill_library_formula",
            "tool_description_formula",
        ]

    def test_trigger_timings_are_the_union(self):
        sp = SystemPromptFormula(system_prompt="p")
        sp.trigger_timings = [BeforeInvocationEvent, AfterInvocationEvent]
        f = MultiSurfaceFormula(sp)
        assert f.trigger_timings == [BeforeInvocationEvent, AfterInvocationEvent]

    def test_tunable_params_merge_all_three(self, tmp_path):
        f = make(tmp_path)
        assert f.get_tunable_params() == {
            "system_prompt": "base prompt",
            "skill_dir": str(tmp_path / "lib"),
            "tool_descriptions": {},
        }


class TestUpdateParams:
    def test_prompt_only_leaves_other_surfaces_untouched(self, tmp_path):
        f = make(tmp_path)
        f.update_params({"system_prompt": "new prompt"})
        assert f.system_prompt.system_prompt == "new prompt"
        assert f.skills.skill_names == ["alpha"]
        assert f.tool_descriptions.overrides == {}

    def test_tool_descriptions_routed(self, tmp_path):
        f = make(tmp_path)
        f.update_params({"tool_descriptions": {"search": "Search by keyword."}})
        assert f.tool_descriptions.overrides == {"search": "Search by keyword."}
        assert f.system_prompt.system_prompt == "base prompt"

    def test_decisions_dir_routed_to_skills(self, tmp_path):
        f = make(tmp_path)
        out = tmp_path / "step" / "skills"
        write_skill(out / "create", "beta")
        write_skill(out / "update", "alpha", body="REVISED")
        f.update_params({"decisions_dir": str(out)})
        assert f.skills.skill_names == ["alpha", "beta"]
        with open(os.path.join(f.skills.members["alpha"], "SKILL.md")) as fh:
            assert "REVISED" in fh.read()

    def test_all_three_in_one_call(self, tmp_path):
        f = make(tmp_path)
        out = tmp_path / "step" / "skills"
        write_skill(out / "create", "beta")
        f.update_params(
            {
                "system_prompt": "P2",
                "decisions_dir": str(out),
                "tool_descriptions": {"click": "Click it."},
            }
        )
        assert f.system_prompt.system_prompt == "P2"
        assert f.skills.skill_names == ["alpha", "beta"]
        assert f.tool_descriptions.overrides == {"click": "Click it."}

    def test_unknown_key_warns_and_changes_nothing(self, tmp_path, caplog):
        import logging

        f = make(tmp_path)
        with caplog.at_level(logging.WARNING):
            f.update_params({"bogus": 1})
        assert "bogus" in caplog.text
        assert f.get_tunable_params()["system_prompt"] == "base prompt"

    def test_empty_params_is_a_no_op(self, tmp_path):
        f = make(tmp_path)
        before = f.get_tunable_params()
        f.update_params({})
        assert f.get_tunable_params() == before


class TestProcess:
    def test_returns_only_changed_keys(self, tmp_path):
        f = make(tmp_path, with_skills=False)
        f.tool_descriptions.update_params({"tool_descriptions": {"search": "v1"}})
        ctx = {"system_prompt": "runtime prompt", "messages": [{"role": "user"}]}
        out = f.process(ctx)
        # prompt changed, tool override present, skills cold so no 'skills' key,
        # and 'messages' is never echoed back
        assert out == {"system_prompt": "base prompt", "tool_descriptions": {"search": "v1"}}

    def test_nothing_to_do_returns_incoming_context(self):
        f = MultiSurfaceFormula(SystemPromptFormula(system_prompt=None))
        ctx = {"system_prompt": "p"}
        assert f.process(ctx) is ctx

    def test_skills_member_contributes_skill_objects(self, tmp_path):
        f = make(tmp_path)
        out = f.process({"system_prompt": "p"})
        assert "skills" in out and [s.name for s in out["skills"]] == ["alpha"]


class TestMaterialize:
    def test_delegates_to_skills(self, tmp_path):
        f = make(tmp_path)
        dest = f.materialize(str(tmp_path / "flat"))
        assert dest == str(tmp_path / "flat")
        assert os.path.isfile(tmp_path / "flat" / "alpha" / "SKILL.md")
        assert f.get_tunable_params()["skill_dir"] == str(tmp_path / "flat")


class TestAtomicUpdate:
    def test_failure_on_a_later_member_rolls_back_the_earlier_ones(self, tmp_path):
        f = MultiSurfaceFormula(
            system_prompt=SystemPromptFormula(system_prompt="base"),
            tool_descriptions=ToolDescriptionFormula(BASE, strict=True),
        )
        with pytest.raises(ValueError, match="ghost"):
            f.update_params({"system_prompt": "changed", "tool_descriptions": {"ghost": "x"}})
        assert f.system_prompt.system_prompt == "base"
        assert f.tool_descriptions.overrides == {}

    def test_snapshot_restore_round_trip(self, tmp_path):
        f = make(tmp_path)
        snap = f.snapshot()
        f.update_params({"system_prompt": "P2", "tool_descriptions": {"click": "c"}})
        assert f.snapshot() != snap
        f.restore(snap)
        assert f.snapshot() == snap
        assert f.skills.skill_names == ["alpha"]
