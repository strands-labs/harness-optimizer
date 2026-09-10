"""Tests for SkillLibraryFormula."""

import json
import os

import pytest
from strands.hooks.events import BeforeInvocationEvent

from strands_harness_optimizer.formulas import SkillLibraryFormula
from strands_harness_optimizer.formulas.skill_library_formula import (
    collect_decisions,
    read_frontmatter,
    render_skill_index,
    validate_skill_dir,
)

FRONTMATTER = "---\nname: {name}\ndescription: Use when {desc}.\n---\n\n{body}\n"


def write_skill(root, name, desc="testing", body="Do the thing.", extra=None):
    """Create <root>/<name>/SKILL.md, plus optional extra files."""
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(FRONTMATTER.format(name=name, desc=desc, body=body))
    for rel, content in (extra or {}).items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    return d


class TestInit:
    def test_cold_start_has_no_members(self):
        formula = SkillLibraryFormula()
        assert formula.members == {}
        assert formula.get_tunable_params() == {"skill_dir": ""}

    def test_loads_existing_set(self, tmp_path):
        write_skill(tmp_path, "alpha")
        write_skill(tmp_path, "beta")
        formula = SkillLibraryFormula(str(tmp_path))
        assert formula.skill_names == ["alpha", "beta"]

    def test_trigger_timings(self):
        assert SkillLibraryFormula().trigger_timings == [BeforeInvocationEvent]

    def test_tunable_param_is_the_directory(self, tmp_path):
        write_skill(tmp_path, "alpha")
        formula = SkillLibraryFormula(str(tmp_path))
        assert formula.get_tunable_params() == {"skill_dir": str(tmp_path)}

    def test_missing_dir_starts_cold(self, tmp_path):
        formula = SkillLibraryFormula(str(tmp_path / "nope"))
        assert formula.members == {}

    def test_missing_dir_raises_in_strict_mode(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SkillLibraryFormula(str(tmp_path / "nope"), strict=True)

    def test_skill_without_frontmatter_is_skipped(self, tmp_path):
        write_skill(tmp_path, "good")
        bad = os.path.join(str(tmp_path), "bad")
        os.makedirs(bad)
        with open(os.path.join(bad, "SKILL.md"), "w") as f:
            f.write("## Trigger\nUse when something.\n")
        formula = SkillLibraryFormula(str(tmp_path))
        assert formula.skill_names == ["good"]

    def test_bad_skill_raises_in_strict_mode(self, tmp_path):
        bad = os.path.join(str(tmp_path), "bad")
        os.makedirs(bad)
        with open(os.path.join(bad, "SKILL.md"), "w") as f:
            f.write("no frontmatter here")
        with pytest.raises(ValueError, match="frontmatter"):
            SkillLibraryFormula(str(tmp_path), strict=True)


class TestSetAlgebra:
    """existing - retired - optimized_old + created."""

    def test_create_adds(self, tmp_path):
        formula = SkillLibraryFormula()
        out = tmp_path / "out"
        write_skill(out / "create", "new-skill")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["new-skill"]

    def test_optimize_replaces_in_place(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha", body="OLD")
        formula = SkillLibraryFormula(str(existing))

        out = tmp_path / "out"
        write_skill(out / "optimize", "alpha", body="NEW")
        formula.update_params({"decisions_dir": str(out)})

        assert formula.skill_names == ["alpha"]
        # Points at the optimized copy, not the original.
        assert "out" in formula.members["alpha"]

    def test_retire_removes(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        write_skill(existing, "beta")
        formula = SkillLibraryFormula(str(existing))

        out = tmp_path / "out"
        os.makedirs(out)
        (out / "retire.txt").write_text("beta\n")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["alpha"]

    def test_untouched_skills_survive(self, tmp_path):
        """The agent only reports what CHANGED; the rest must carry over."""
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        write_skill(existing, "beta")
        write_skill(existing, "gamma")
        formula = SkillLibraryFormula(str(existing))

        out = tmp_path / "out"
        write_skill(out / "create", "delta")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["alpha", "beta", "delta", "gamma"]

    def test_merge_is_create_plus_retire(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "search-basic")
        write_skill(existing, "search-refine")
        formula = SkillLibraryFormula(str(existing))

        out = tmp_path / "out"
        write_skill(out / "create", "product-search")
        os.makedirs(out, exist_ok=True)
        (out / "retire.txt").write_text("search-basic\nsearch-refine\n")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["product-search"]

    def test_split_is_creates_plus_retire(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "everything")
        formula = SkillLibraryFormula(str(existing))

        out = tmp_path / "out"
        write_skill(out / "create", "part-one")
        write_skill(out / "create", "part-two")
        (out / "retire.txt").write_text("everything\n")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["part-one", "part-two"]

    def test_retire_unknown_skill_is_ignored(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        formula = SkillLibraryFormula(str(existing))

        out = tmp_path / "out"
        os.makedirs(out)
        (out / "retire.txt").write_text("never-deployed\n")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["alpha"]

    def test_no_decisions_leaves_set_unchanged(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        formula = SkillLibraryFormula(str(existing))
        out = tmp_path / "out"
        os.makedirs(out)
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["alpha"]

    def test_malformed_create_is_rejected(self, tmp_path):
        """A skill the runtime could never load must not enter the set."""
        formula = SkillLibraryFormula()
        out = tmp_path / "out"
        d = out / "create" / "broken"
        os.makedirs(d)
        (d / "SKILL.md").write_text("## Trigger\nUse when X.\n")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == []

    def test_deploy_dir_accepted_as_create_alias(self, tmp_path):
        formula = SkillLibraryFormula()
        out = tmp_path / "out"
        write_skill(out / "deploy", "legacy-named")
        formula.update_params({"decisions_dir": str(out)})
        assert formula.skill_names == ["legacy-named"]

    def test_update_params_requires_a_known_key(self):
        formula = SkillLibraryFormula()
        with pytest.raises(ValueError, match="decisions_dir"):
            formula.update_params({"instructions": "nope"})

    def test_skill_dir_replaces_wholesale(self, tmp_path):
        first = tmp_path / "first"
        write_skill(first, "alpha")
        formula = SkillLibraryFormula(str(first))

        second = tmp_path / "second"
        write_skill(second, "beta")
        formula.update_params({"skill_dir": str(second)})
        assert formula.skill_names == ["beta"]


class TestMaterialize:
    def test_writes_flat_folder(self, tmp_path):
        formula = SkillLibraryFormula()
        out = tmp_path / "out"
        write_skill(out / "create", "alpha")
        write_skill(out / "create", "beta")
        formula.update_params({"decisions_dir": str(out)})

        dest = formula.materialize(str(tmp_path / "skill_set"))
        assert sorted(os.listdir(dest)) == ["alpha", "beta"]
        assert os.path.isfile(os.path.join(dest, "alpha", "SKILL.md"))

    def test_repoints_skill_dir(self, tmp_path):
        formula = SkillLibraryFormula()
        write_skill(tmp_path / "out" / "create", "alpha")
        formula.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = formula.materialize(str(tmp_path / "skill_set"))
        assert formula.get_tunable_params() == {"skill_dir": dest}

    def test_carries_subdirectories(self, tmp_path):
        """A skill may ship scripts/ or resources/ alongside SKILL.md."""
        formula = SkillLibraryFormula()
        write_skill(
            tmp_path / "out" / "create",
            "alpha",
            extra={"scripts/check.py": "print('hi')", "resources/notes.md": "# notes"},
        )
        formula.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = formula.materialize(str(tmp_path / "skill_set"))
        assert os.path.isfile(os.path.join(dest, "alpha", "scripts", "check.py"))
        assert os.path.isfile(os.path.join(dest, "alpha", "resources", "notes.md"))

    def test_is_idempotent(self, tmp_path):
        formula = SkillLibraryFormula()
        write_skill(tmp_path / "out" / "create", "alpha")
        formula.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = str(tmp_path / "skill_set")
        formula.materialize(dest)
        formula.materialize(dest)
        assert sorted(os.listdir(dest)) == ["alpha"]

    def test_round_trips_through_a_new_formula(self, tmp_path):
        formula = SkillLibraryFormula()
        write_skill(tmp_path / "out" / "create", "alpha")
        formula.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = formula.materialize(str(tmp_path / "skill_set"))
        assert SkillLibraryFormula(dest).skill_names == ["alpha"]


class TestFrontmatter:
    def test_reads_past_a_long_description(self, tmp_path):
        """A fixed-size window would cut mid-value and lose the closing ---."""
        long_desc = "x" * 900
        d = write_skill(tmp_path, "alpha", desc=long_desc)
        block = read_frontmatter(os.path.join(d, "SKILL.md"))
        assert "name: alpha" in block
        assert "description:" in block
        assert validate_skill_dir(d) == []

    def test_missing_frontmatter_is_reported(self, tmp_path):
        d = os.path.join(str(tmp_path), "bad")
        os.makedirs(d)
        with open(os.path.join(d, "SKILL.md"), "w") as f:
            f.write("## Trigger\nUse when X.\n")
        problems = validate_skill_dir(d)
        assert problems and "frontmatter" in problems[0]

    def test_missing_description_is_reported(self, tmp_path):
        d = os.path.join(str(tmp_path), "bad")
        os.makedirs(d)
        with open(os.path.join(d, "SKILL.md"), "w") as f:
            f.write("---\nname: bad\n---\n\nbody\n")
        problems = validate_skill_dir(d)
        assert any("description:" in p for p in problems)

    def test_missing_skill_md_is_reported(self, tmp_path):
        d = os.path.join(str(tmp_path), "empty")
        os.makedirs(d)
        assert validate_skill_dir(d)


class TestRenderSkillIndex:
    def test_cold_start(self):
        assert render_skill_index(None) == "(none deployed)"
        assert render_skill_index("") == "(none deployed)"

    def test_includes_name_path_and_trigger(self, tmp_path):
        write_skill(tmp_path, "alpha", desc="searching a catalog")
        index = render_skill_index(str(tmp_path))
        assert "### alpha" in index
        assert "searching a catalog" in index
        assert str(tmp_path) in index

    def test_omits_the_body(self, tmp_path):
        """Triggers only: reading bodies up front anchors toward editing them."""
        write_skill(tmp_path, "alpha", body="SECRET_BODY_MARKER")
        assert "SECRET_BODY_MARKER" not in render_skill_index(str(tmp_path))


class TestCollectDecisions:
    def test_empty_folder(self, tmp_path):
        assert collect_decisions(str(tmp_path)) == {"create": [], "optimize": [], "retire": []}

    def test_reads_all_three(self, tmp_path):
        write_skill(tmp_path / "create", "a")
        write_skill(tmp_path / "optimize", "b")
        (tmp_path / "retire.txt").write_text("c\nd\n")
        d = collect_decisions(str(tmp_path))
        assert len(d["create"]) == 1 and len(d["optimize"]) == 1
        assert d["retire"] == ["c", "d"]

    def test_ignores_blank_retire_lines(self, tmp_path):
        (tmp_path / "retire.txt").write_text("a\n\n  \nb\n")
        assert collect_decisions(str(tmp_path))["retire"] == ["a", "b"]


class TestProcess:
    def test_cold_start_leaves_context_untouched(self):
        """Returning {"skills": []} would CLEAR a plugin's own skills."""
        formula = SkillLibraryFormula()
        ctx = {"system_prompt": "x", "messages": []}
        assert formula.process(ctx) == ctx

    def test_returns_skills_for_the_adapter(self, tmp_path):
        write_skill(tmp_path, "alpha")
        formula = SkillLibraryFormula(str(tmp_path))
        result = formula.process({"system_prompt": "x", "messages": []})
        assert "skills" in result
        assert [s.name for s in result["skills"]] == ["alpha"]

    def test_reflects_an_update(self, tmp_path):
        write_skill(tmp_path, "alpha")
        formula = SkillLibraryFormula(str(tmp_path))
        out = tmp_path / "out"
        write_skill(out / "create", "beta")
        formula.update_params({"decisions_dir": str(out)})
        result = formula.process({"system_prompt": "x", "messages": []})
        assert sorted(s.name for s in result["skills"]) == ["alpha", "beta"]
