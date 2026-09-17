"""Tests for ToolDescriptionFormula."""

import logging

import pytest
import yaml
from strands.hooks.events import BeforeInvocationEvent

from strands_harness_optimizer.formulas import ToolDescriptionFormula
from strands_harness_optimizer.formulas.tool_description_formula import load_yaml

BASE = {"search": "Search the catalog.", "click": "Click an element on the page."}


class TestInit:
    def test_trigger_timings(self):
        assert ToolDescriptionFormula(BASE).trigger_timings == [BeforeInvocationEvent]

    def test_starts_with_no_overrides(self):
        f = ToolDescriptionFormula(BASE)
        assert f.get_tunable_params() == {"tool_descriptions": {}}
        assert f.effective() == BASE
        assert f.tool_names == ["click", "search"]

    def test_base_is_copied_not_aliased(self):
        src = dict(BASE)
        f = ToolDescriptionFormula(src)
        src["search"] = "mutated"
        assert f.base["search"] == BASE["search"]

    def test_initial_overrides_are_validated(self):
        f = ToolDescriptionFormula(BASE, overrides={"search": "better", "ghost": "x"})
        assert f.overrides == {"search": "better"}

    def test_initial_overrides_strict_raises_on_unknown(self):
        with pytest.raises(ValueError, match="not in the toolset"):
            ToolDescriptionFormula(BASE, overrides={"ghost": "x"}, strict=True)


class TestUpdateParams:
    def test_edit_lands_in_overrides_only(self):
        f = ToolDescriptionFormula(BASE)
        f.update_params({"tool_descriptions": {"search": "Search by keyword."}})
        assert f.overrides == {"search": "Search by keyword."}
        assert f.base == BASE  # never mutated
        assert f.effective() == {"search": "Search by keyword.", "click": BASE["click"]}

    def test_unedited_override_survives_a_later_partial_edit(self):
        """The carry-forward rule: iteration 2 editing only `click` keeps `search`."""
        f = ToolDescriptionFormula(BASE)
        f.update_params({"tool_descriptions": {"search": "v1 search"}})
        f.update_params({"tool_descriptions": {"click": "v1 click"}})
        assert f.overrides == {"search": "v1 search", "click": "v1 click"}

    def test_edit_wins_over_existing_override(self):
        f = ToolDescriptionFormula(BASE, overrides={"search": "old"})
        f.update_params({"tool_descriptions": {"search": "new"}})
        assert f.overrides["search"] == "new"

    def test_unknown_name_dropped_with_warning(self, caplog):
        f = ToolDescriptionFormula(BASE)
        with caplog.at_level(logging.WARNING):
            f.update_params({"tool_descriptions": {"ghost": "x", "search": "ok"}})
        assert f.overrides == {"search": "ok"}
        assert "ghost" in caplog.text and "not in the toolset" in caplog.text

    def test_unknown_name_raises_in_strict_mode(self):
        f = ToolDescriptionFormula(BASE, strict=True)
        with pytest.raises(ValueError, match="ghost"):
            f.update_params({"tool_descriptions": {"ghost": "x"}})

    def test_over_limit_dropped(self, caplog):
        f = ToolDescriptionFormula(BASE, max_chars=20)
        with caplog.at_level(logging.WARNING):
            f.update_params({"tool_descriptions": {"search": "x" * 21}})
        assert f.overrides == {}
        assert "over the 20-char limit" in caplog.text

    def test_over_limit_raises_in_strict_mode(self):
        f = ToolDescriptionFormula(BASE, strict=True, max_chars=20)
        with pytest.raises(ValueError, match="over the 20-char limit"):
            f.update_params({"tool_descriptions": {"search": "x" * 21}})

    def test_empty_string_keeps_current(self, caplog):
        f = ToolDescriptionFormula(BASE, overrides={"search": "keep me"})
        with caplog.at_level(logging.WARNING):
            f.update_params({"tool_descriptions": {"search": "   ", "click": None}})
        assert f.overrides == {"search": "keep me"}
        assert "empty" in caplog.text

    def test_missing_key_is_a_no_op(self):
        f = ToolDescriptionFormula(BASE, overrides={"search": "x"})
        f.update_params({"system_prompt": "unrelated"})
        assert f.overrides == {"search": "x"}

    def test_non_mapping_raises(self):
        f = ToolDescriptionFormula(BASE)
        with pytest.raises(ValueError, match="mapping"):
            f.update_params({"tool_descriptions": ["search"]})

    def test_empty_base_accepts_any_name(self):
        f = ToolDescriptionFormula({})
        f.update_params({"tool_descriptions": {"anything": "goes"}})
        assert f.overrides == {"anything": "goes"}
        assert f.tool_names == ["anything"]


class TestProcess:
    def test_no_overrides_leaves_context_untouched(self):
        f = ToolDescriptionFormula(BASE)
        ctx = {"system_prompt": "p"}
        assert f.process(ctx) is ctx

    def test_overrides_returned_under_wire_key(self):
        f = ToolDescriptionFormula(BASE, overrides={"search": "x"})
        out = f.process({"system_prompt": "p"})
        assert out == {"tool_descriptions": {"search": "x"}}
        out["tool_descriptions"]["search"] = "mutated"
        assert f.overrides["search"] == "x"  # a copy, not the live dict


class TestYaml:
    def test_render_effective_round_trips(self):
        f = ToolDescriptionFormula(BASE, overrides={"search": "Search by keyword."})
        data = yaml.safe_load(f.render_effective_yaml())
        assert data == {"tool_descriptions": f.effective()}

    def test_load_yaml_wrapped_and_bare(self, tmp_path):
        wrapped = tmp_path / "w.yaml"
        wrapped.write_text(yaml.safe_dump({"tool_descriptions": BASE}))
        bare = tmp_path / "b.yaml"
        bare.write_text(yaml.safe_dump(BASE))
        assert load_yaml(str(wrapped)) == BASE
        assert load_yaml(str(bare)) == BASE

    def test_load_yaml_rejects_non_mapping(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a list\n")
        with pytest.raises(ValueError, match="mapping"):
            load_yaml(str(p))
        p.write_text(yaml.safe_dump({"tool_descriptions": ["x"]}))
        with pytest.raises(ValueError, match="mapping"):
            load_yaml(str(p))

    def test_from_yaml(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(yaml.safe_dump({"tool_descriptions": BASE}))
        ov = tmp_path / "ov.yaml"
        ov.write_text(yaml.safe_dump({"tool_descriptions": {"search": "v1", "ghost": "x"}}))
        f = ToolDescriptionFormula.from_yaml(str(base), str(ov))
        assert f.base == BASE
        assert f.overrides == {"search": "v1"}


class TestValidateAndReplace:
    def test_validate_edits_reports_each_problem(self):
        f = ToolDescriptionFormula(BASE, max_chars=10)
        out = f.validate_edits({"search": "ok", "ghost": "x", "click": "", "search2": "y" * 11})
        assert out["search"] == ""
        assert "not in the toolset" in out["ghost"]
        assert out["click"] == "empty"
        assert "not in the toolset" in out["search2"]
        assert f.overrides == {}  # validation never mutates

    def test_replace_overrides_is_wholesale(self):
        f = ToolDescriptionFormula(BASE, overrides={"click": "stale"})
        f.replace_overrides({"search": "v1", "ghost": "dropped"})
        assert f.overrides == {"search": "v1"}
        f.replace_overrides(None)
        assert f.overrides == {}
