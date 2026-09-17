"""Tests for MultiSurfaceOptimizer with a mocked writer.

The reflector agent is replaced by a function that writes a fixed artifact tree
into the step directory, so every assertion about which surfaces were applied is
deterministic. The real model is exercised only by the integration test.
"""

import json
import logging
import os

import pytest
import yaml

from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import (
    MultiSurfaceFormula,
    SkillLibraryFormula,
    SystemPromptFormula,
    ToolDescriptionFormula,
)
from strands_harness_optimizer.optimizers import MultiSurfaceOptimizer
from strands_harness_optimizer.optimizers.multi_surface import renderers
from strands_harness_optimizer.optimizers.multi_surface.multi_surface import (
    ATTEMPTS_FILE,
    FAILED_FILE,
    SKILL_SET_DIR,
)
from strands_harness_optimizer.optimizers.multi_surface.objective import (
    normalize_weights,
    objective_terms,
    weighted_total,
)

FRONTMATTER = "---\nname: {name}\ndescription: Use when {desc}.\n---\n\n{body}\n"
BASE_TOOLS = {"search": "Search the catalog.", "click": "Click an element."}
PROMPT = 'You are a shopping agent.\nCall search(query: str) first; "quote" <tag>.'


# ── fixtures ─────────────────────────────────────────────────────────────────


def write_skill(root, name, desc="testing", body="Do the thing."):
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(FRONTMATTER.format(name=name, desc=desc, body=body))
    return d


def make_formula(tmp_path, with_skill=True):
    skills = None
    if with_skill:
        write_skill(tmp_path / "lib", "alpha", body="Alpha body.")
        skills = SkillLibraryFormula(str(tmp_path / "lib"))
    return MultiSurfaceFormula(
        system_prompt=SystemPromptFormula(system_prompt=PROMPT),
        skills=skills,
        tool_descriptions=ToolDescriptionFormula(BASE_TOOLS),
    )


def make_rollouts(n=4, scores=None):
    rollouts, rewards = [], []
    for i in range(n):
        msgs = [
            {"role": "user", "content": [{"text": f"task {i}"}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"name": "search", "input": {"q": f"x{i % 2}"}}}],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"status": "success", "content": [{"text": "3 results"}]}}
                ],
            },
            {
                "role": "assistant",
                "content": [{"toolUse": {"name": "search", "input": {"q": f"x{i % 2}"}}}],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"status": "error", "content": [{"text": "rate limited"}]}}
                ],
            },
            {"role": "assistant", "content": [{"text": "done"}]},
        ]
        rollouts.append(
            Rollout(
                data_sample={"task_id": f"t{i}"},
                messages=msgs,
                metadata={"eval_result": {"reward": float(i % 2), "success": bool(i % 2)}},
            )
        )
        meta = {"scores": dict(scores[i])} if scores else {}
        rewards.append(Reward(reward=float(i % 2), metadata=meta))
    return rollouts, rewards


class StubOptimizer(MultiSurfaceOptimizer):
    """Replaces the agent with ``writer(step_dir, optimizer)``."""

    def __init__(self, *args, writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.system_prompts: list[str] = []
        self.task_messages: list[str] = []

    def _create_agent(self, system_prompt):
        self.system_prompts.append(system_prompt)
        return object()

    def _invoke_agent(self, agent, message):
        self.task_messages.append(message)
        self.writer(self._current_step_dir, self)


def make_opt(tmp_path, formula, writer, **kw):
    kw.setdefault("n_sample_traces", -1)
    kw.setdefault("retry_backoff_s", 0)
    opt = StubOptimizer(formula, output_folder=str(tmp_path / "runs"), writer=writer, **kw)
    rollouts, rewards = make_rollouts(scores=kw.pop("_scores", None))
    opt.add_rollouts(rollouts)
    opt.add_rewards(rewards)
    return opt


# ── writers ──────────────────────────────────────────────────────────────────


def _write_prompt(step_dir, text):
    d = os.path.join(step_dir, "system_prompt")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "optimized_prompt.yaml"), "w") as f:
        yaml.safe_dump({"system_prompt": text}, f)


def _write_tools(step_dir, mapping):
    d = os.path.join(step_dir, "tool_descriptions")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "optimized_tool_descriptions.yaml"), "w") as f:
        yaml.safe_dump({"tool_descriptions": mapping}, f)


def _write_findings(step_dir, findings):
    with open(os.path.join(step_dir, "findings.json"), "w") as f:
        json.dump({"findings": findings}, f)


def write_all_three(step_dir, opt):
    write_skill(os.path.join(step_dir, "skills", "create"), "beta", body="Beta body.")
    write_skill(os.path.join(step_dir, "skills", "update"), "alpha", body="Alpha REVISED.")
    _write_prompt(step_dir, PROMPT + "\nNever repeat a search.")
    _write_tools(step_dir, {"search": "Search the catalog by keyword; results are ranked."})
    _write_findings(step_dir, [{"id": "f01", "statement": "x"}])


def write_prompt_only(step_dir, opt):
    _write_prompt(step_dir, PROMPT + "\nEdit.")
    _write_findings(step_dir, [{"id": "f01"}])


def write_findings_only(step_dir, opt):
    _write_findings(step_dir, [])


def write_nothing(step_dir, opt):
    pass


def write_bad_skill_good_prompt(step_dir, opt):
    d = os.path.join(step_dir, "skills", "create", "bad")
    os.makedirs(d)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write("## Trigger\nno frontmatter here\n")
    _write_prompt(step_dir, PROMPT + "\nEdit.")
    _write_findings(step_dir, [{"id": "f01"}])


def write_all_bad(step_dir, opt):
    d = os.path.join(step_dir, "skills", "create", "bad")
    os.makedirs(d)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write("no frontmatter\n")
    d = os.path.join(step_dir, "system_prompt")
    os.makedirs(d)
    with open(os.path.join(d, "optimized_prompt.yaml"), "w") as f:
        f.write("not_the_key: x\n")
    _write_findings(step_dir, [{"id": "f01"}])


# ── the step ─────────────────────────────────────────────────────────────────


class TestStepAppliesSurfaces:
    def test_all_three_surfaces_applied_in_one_step(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_all_three)
        opt.step()

        assert opt.last_applied == ["system_prompt", "skills", "tool_description"]
        assert opt.last_dropped == {}
        assert formula.system_prompt.system_prompt.endswith("Never repeat a search.")
        assert formula.skills.skill_names == ["alpha", "beta"]
        assert formula.tool_descriptions.overrides == {
            "search": "Search the catalog by keyword; results are ranked."
        }
        # base untouched, effective shows the override
        assert formula.tool_descriptions.base == BASE_TOOLS
        assert formula.tool_descriptions.effective()["click"] == BASE_TOOLS["click"]

    def test_skills_materialized_and_repointed(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_all_three)
        opt.step()
        step_dir = opt.last_step_dir
        skill_set = os.path.join(step_dir, SKILL_SET_DIR)
        assert formula.get_tunable_params()["skill_dir"] == skill_set
        assert sorted(os.listdir(skill_set)) == ["alpha", "beta"]
        with open(os.path.join(skill_set, "alpha", "SKILL.md")) as f:
            assert "REVISED" in f.read()

    def test_prompt_only_leaves_other_surfaces(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_prompt_only)
        opt.step()
        assert opt.last_applied == ["system_prompt"]
        assert formula.skills.skill_names == ["alpha"]
        assert formula.tool_descriptions.overrides == {}

    def test_step_count_and_history(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_all_three)
        opt.step()
        assert opt._step_count == 1
        assert opt._prompt_history[-1]["applied"] == ["system_prompt", "skills", "tool_description"]
        assert opt.last_findings == {"findings": [{"id": "f01", "statement": "x"}]}


class TestOutcomeMatrix:
    def test_partial_valid_applies_valid_and_drops_invalid(self, tmp_path, caplog):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_bad_skill_good_prompt)
        with caplog.at_level(logging.WARNING):
            opt.step()
        assert opt.last_applied == ["system_prompt"]
        assert "skills" in opt.last_dropped
        assert formula.skills.skill_names == ["alpha"]  # bad skill never entered the set
        assert "dropping skills" in caplog.text
        # the invalid artifact is kept on disk for inspection
        assert os.path.exists(os.path.join(opt.last_step_dir, "skills", "create", "bad"))

    def test_all_invalid_raises_and_keeps_artifacts(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_all_bad)
        with pytest.raises(RuntimeError, match="none is usable"):
            opt.step()
        step_dir = os.path.join(tmp_path, "runs", "step_0001")
        assert os.path.exists(os.path.join(step_dir, FAILED_FILE))
        assert os.path.exists(os.path.join(step_dir, "skills", "create", "bad", "SKILL.md"))
        assert formula.system_prompt.system_prompt == PROMPT  # nothing applied

    def test_no_artifacts_with_findings_is_a_no_op(self, tmp_path, caplog):
        formula = make_formula(tmp_path)
        before = formula.get_tunable_params()
        opt = make_opt(tmp_path, formula, write_findings_only)
        with caplog.at_level(logging.INFO):
            opt.step()
        assert opt.last_applied == []
        assert opt._step_count == 1
        assert "no surface edited" in caplog.text
        after = formula.get_tunable_params()
        # skill_dir moves to this step's materialized copy; everything else is identical
        assert after["system_prompt"] == before["system_prompt"]
        assert after["tool_descriptions"] == before["tool_descriptions"]
        assert formula.skills.skill_names == ["alpha"]

    def test_no_artifacts_no_findings_raises(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_nothing)
        with pytest.raises(RuntimeError, match="indistinguishable"):
            opt.step()
        assert opt._step_count == 0
        assert os.path.exists(os.path.join(tmp_path, "runs", "step_0001", FAILED_FILE))

    def test_unparseable_findings_counts_as_missing(self, tmp_path):
        def writer(step_dir, opt):
            with open(os.path.join(step_dir, "findings.json"), "w") as f:
                f.write("{not json")

        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, writer)
        with pytest.raises(RuntimeError, match="not valid JSON"):
            opt.step()


class TestStepDirectories:
    def test_each_step_gets_a_fresh_directory_and_never_rereads(self, tmp_path):
        """Stale replay: step 2 writes nothing but findings; step 1's skill must not re-apply."""
        formula = make_formula(tmp_path)
        calls = []

        def writer(step_dir, opt):
            calls.append(step_dir)
            if len(calls) == 1:
                write_all_three(step_dir, opt)
            else:
                write_findings_only(step_dir, opt)

        opt = make_opt(tmp_path, formula, writer)
        opt.step()
        names_after_1 = formula.skills.skill_names
        overrides_after_1 = dict(formula.tool_descriptions.overrides)
        prompt_after_1 = formula.system_prompt.system_prompt
        opt.step()

        assert [os.path.basename(c) for c in calls] == ["step_0001", "step_0002"]
        assert opt.last_step_dir.endswith("step_0002")
        assert opt.last_applied == []
        # nothing from step_0001 leaked into step 2's application
        assert formula.skills.skill_names == names_after_1
        assert formula.tool_descriptions.overrides == overrides_after_1
        assert formula.system_prompt.system_prompt == prompt_after_1
        # step_0001's artifacts are still there, untouched
        assert os.path.exists(os.path.join(calls[0], "skills", "create", "beta", "SKILL.md"))
        # and step 2 saw step 1's resolved library as "deployed"
        assert "----- beta" in opt.task_messages[1]

    def test_failed_step_leaves_marker_and_next_step_skips_its_number(self, tmp_path):
        formula = make_formula(tmp_path)
        calls = []

        def writer(step_dir, opt):
            calls.append(step_dir)
            if len(calls) == 1:
                raise ValueError("boom")  # not transient: no retry
            write_prompt_only(step_dir, opt)

        opt = make_opt(tmp_path, formula, writer)
        with pytest.raises(ValueError, match="boom"):
            opt.step()
        assert opt._step_count == 0
        with open(os.path.join(tmp_path, "runs", "step_0001", FAILED_FILE)) as f:
            assert "ValueError: boom" in f.read()

        opt.step()
        assert opt._step_count == 1
        assert opt.last_step_dir.endswith("step_0002")
        assert len(calls) == 2

    def test_allocation_skips_directories_left_by_someone_else(self, tmp_path):
        formula = make_formula(tmp_path)
        os.makedirs(tmp_path / "runs" / "step_0001")
        opt = make_opt(tmp_path, formula, write_prompt_only)
        opt.step()
        assert opt.last_step_dir.endswith("step_0002")


class TestRetry:
    def test_transient_failure_clears_partial_output_and_retries(self, tmp_path):
        formula = make_formula(tmp_path)
        calls = []

        def writer(step_dir, opt):
            calls.append(step_dir)
            if len(calls) == 1:
                write_skill(os.path.join(step_dir, "skills", "create"), "partial")
                raise RuntimeError("ThrottlingException: Too many requests")
            # second attempt: only a prompt edit
            write_prompt_only(step_dir, opt)

        opt = make_opt(tmp_path, formula, writer, retry_attempts=3)
        opt.step()
        assert len(calls) == 2
        assert opt.last_applied == ["system_prompt"]
        assert formula.skills.skill_names == ["alpha"]  # 'partial' never applied
        assert not os.path.exists(os.path.join(opt.last_step_dir, "skills"))
        with open(os.path.join(opt.last_step_dir, ATTEMPTS_FILE)) as f:
            log = f.read()
        assert "attempt 1: transient" in log and "attempt 2: ok" in log

    def test_transient_failure_exhausts_attempts(self, tmp_path):
        formula = make_formula(tmp_path)

        def writer(step_dir, opt):
            raise RuntimeError("ServiceUnavailable")

        opt = make_opt(tmp_path, formula, writer, retry_attempts=2)
        with pytest.raises(RuntimeError, match="ServiceUnavailable"):
            opt.step()
        assert len(opt.task_messages) == 2

    def test_non_transient_failure_does_not_retry(self, tmp_path):
        formula = make_formula(tmp_path)

        def writer(step_dir, opt):
            raise RuntimeError("ValidationException: bad model id")

        opt = make_opt(tmp_path, formula, writer, retry_attempts=3)
        with pytest.raises(RuntimeError, match="ValidationException"):
            opt.step()
        assert len(opt.task_messages) == 1


class TestCarryForward:
    def test_unedited_tool_override_survives_second_step(self, tmp_path):
        """Iteration 2 edits only `click`; iteration 1's `search` must remain."""
        formula = make_formula(tmp_path)
        calls = []

        def writer(step_dir, opt):
            calls.append(step_dir)
            if len(calls) == 1:
                _write_tools(step_dir, {"search": "v1 search"})
            else:
                _write_tools(step_dir, {"click": "v1 click"})
            _write_findings(step_dir, [{"id": "f01"}])

        opt = make_opt(tmp_path, formula, writer)
        opt.step()
        opt.step()
        assert formula.tool_descriptions.overrides == {
            "search": "v1 search",
            "click": "v1 click",
        }
        # the second step's task message showed the effective map with step 1's edit
        assert "v1 search" in opt.task_messages[1]

    def test_unknown_tool_in_yaml_is_dropped_by_the_formula(self, tmp_path, caplog):
        def writer(step_dir, opt):
            _write_tools(step_dir, {"ghost": "x", "search": "ok"})
            _write_findings(step_dir, [])

        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, writer)
        with caplog.at_level(logging.WARNING):
            opt.step()
        assert formula.tool_descriptions.overrides == {"search": "ok"}
        assert "ghost" in caplog.text


class TestInputsShownToTheAgent:
    def test_template_variables_rendered(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_findings_only)
        opt.step()
        task = opt.task_messages[0]
        # the prompt body, verbatim and unescaped
        assert 'Call search(query: str) first; "quote" <tag>.' in task
        assert f"Length: {len(PROMPT)} characters" in task
        # deployed skills in full
        assert "----- alpha" in task and "Alpha body." in task
        # toolset from the tool-description base, and the effective YAML
        assert "click, search" in task
        assert "Search the catalog." in task
        # census: 4 traces, 8 search calls, 4 errors, 4 duplicates
        assert "Episodes: 4 JSON files" in task
        assert "calls=   8 ok=   4 failed=  4 dup=  4" in task
        assert "never called in these traces: click" in task
        # objective: one default row
        assert "| TaskSuccessScore | 1.00 |" in task
        # the output folder is THIS step's directory
        assert opt.last_step_dir in task
        # the system prompt has no submit-tool suffix appended
        assert "submit_optimized_params" not in opt.system_prompts[0]

    def test_current_files_written_before_agent_runs(self, tmp_path):
        seen = {}

        def writer(step_dir, opt):
            cur = os.path.join(step_dir, "current")
            with open(os.path.join(cur, "system_prompt.yaml")) as f:
                seen["prompt"] = yaml.safe_load(f)["system_prompt"]
            with open(os.path.join(cur, "tool_descriptions.yaml")) as f:
                seen["tools"] = yaml.safe_load(f)["tool_descriptions"]
            seen["rendered"] = os.path.exists(os.path.join(cur, "rendered_task_message.txt"))
            _write_findings(step_dir, [])

        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, writer)
        opt.step()
        assert seen["prompt"] == PROMPT  # byte-exact, no added trailing newline
        assert seen["tools"] == BASE_TOOLS
        assert seen["rendered"]

    def test_trace_files_carry_objective_and_eval_result(self, tmp_path):
        seen = {}

        def writer(step_dir, opt):
            folder = opt._temp_dir
            names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))
            with open(os.path.join(folder, names[0])) as f:
                seen["trace"] = json.load(f)
            seen["count"] = len(names)
            _write_findings(step_dir, [])

        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, writer)
        opt.step()
        d = seen["trace"]
        assert seen["count"] == 4
        assert d["objective"]["weights"] == {"TaskSuccessScore": 1.0}
        assert d["objective"]["terms"] == {"TaskSuccessScore": d["reward"]}
        assert d["objective"]["reward"] == d["reward"]
        assert d["response"]["eval_result"] == d["eval_result"]
        assert "success" in d["eval_result"]  # the runtime's own evaluator output, not a stub
        assert d["response"]["messages"]

    def test_agent_tools_override_and_disagreement_warning(self, tmp_path, caplog):
        formula = make_formula(tmp_path)
        opt = make_opt(
            tmp_path, formula, write_findings_only, agent_tools=["search", "click", "buy"]
        )
        with caplog.at_level(logging.WARNING):
            opt.step()
        assert "search, click, buy" in opt.task_messages[0]
        assert "never called in these traces: buy, click" in opt.task_messages[0]
        assert "disagree" in caplog.text


class TestObjectiveWeights:
    def test_missing_term_raises_before_agent_runs(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(
            tmp_path,
            formula,
            write_findings_only,
            objective_weights={"TaskSuccessScore": 0.5, "Conciseness": 0.5},
        )
        with pytest.raises(ValueError, match="Conciseness"):
            opt.step()
        assert opt.task_messages == []  # never paid for the agent
        assert not os.path.exists(tmp_path / "runs" / "step_0001")  # refused before allocating

    def test_configured_terms_rendered_and_written(self, tmp_path):
        formula = make_formula(tmp_path)
        scores = [
            {"Conciseness": 0.2},
            {"Conciseness": 0.8},
            {"Conciseness": 0.4},
            {"Conciseness": 1.0},
        ]
        seen = {}

        def writer(step_dir, opt):
            folder = opt._temp_dir
            names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))
            seen["traces"] = [json.load(open(os.path.join(folder, n))) for n in names]
            _write_findings(step_dir, [])

        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=writer,
            n_sample_traces=-1,
            retry_backoff_s=0,
            objective_weights={"TaskSuccessScore": 0.5, "Conciseness": 0.5},
        )
        rollouts, rewards = make_rollouts(scores=scores)
        rewards[0].metadata["explanations"] = {"Conciseness": "Restated the whole cart twice."}
        opt.add_rollouts(rollouts)
        opt.add_rewards(rewards)
        opt.step()

        task = opt.task_messages[0]
        assert "| TaskSuccessScore | 0.50 |" in task
        assert "| Conciseness | 0.50 |" in task
        assert "Restated the whole cart twice." in task
        by_task = {t["task_id"]: t for t in seen["traces"]}
        t0 = by_task["t0"]
        assert t0["objective"]["terms"] == {"TaskSuccessScore": 0.0, "Conciseness": 0.2}
        assert t0["reward"] == pytest.approx(0.1)
        assert t0["objective"]["reward"] == pytest.approx(0.1)


class TestConstruction:
    def test_requires_multi_surface_formula(self, tmp_path):
        with pytest.raises(TypeError, match="MultiSurfaceFormula"):
            MultiSurfaceOptimizer(
                SystemPromptFormula(system_prompt="p"), output_folder=str(tmp_path)
            )

    def test_defaults(self, tmp_path):
        opt = MultiSurfaceOptimizer(make_formula(tmp_path), output_folder=str(tmp_path / "r"))
        assert opt.system_prompt_suffix == ""
        assert opt.model_config["max_tokens"] == 32000
        assert opt.objective_weights is None
        assert opt._weights == {"TaskSuccessScore": 1.0}

    def test_tools_are_shell_and_editor(self, tmp_path):
        opt = MultiSurfaceOptimizer(make_formula(tmp_path), output_folder=str(tmp_path / "r"))
        names = [getattr(t, "__name__", "").rsplit(".", 1)[-1] for t in opt._get_tools(None)]
        assert names == ["shell", "editor"]

    def test_conversation_manager_settings(self, tmp_path):
        opt = MultiSurfaceOptimizer(
            make_formula(tmp_path), output_folder=str(tmp_path / "r"), window_size=77, pin_first=1
        )
        cm = opt._conversation_manager()
        assert cm.window_size == 77
        assert getattr(cm, "pin_first", None) == 1

    def test_state_round_trip(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, write_all_three, objective_weights=None)
        opt.step()
        state = opt.get_state()
        assert state["last_applied"] == ["system_prompt", "skills", "tool_description"]
        assert state["formula_params"]["tool_descriptions"] == formula.tool_descriptions.overrides

        fresh_formula = make_formula(tmp_path / "fresh")
        # a stale override that the checkpoint does NOT contain must not survive
        fresh_formula.tool_descriptions.update_params({"tool_descriptions": {"click": "stale"}})
        fresh = MultiSurfaceOptimizer(
            fresh_formula, output_folder=str(tmp_path / "r2"), window_size=5, retry_attempts=9
        )
        fresh.load_state(state)
        assert fresh._step_count == 1
        assert fresh_formula.system_prompt.system_prompt == formula.system_prompt.system_prompt
        assert fresh_formula.tool_descriptions.overrides == formula.tool_descriptions.overrides
        assert "click" not in fresh_formula.tool_descriptions.overrides
        assert (
            fresh_formula.get_tunable_params()["skill_dir"] == state["formula_params"]["skill_dir"]
        )
        assert fresh_formula.skills.skill_names == ["alpha", "beta"]
        assert fresh.window_size == 120 and fresh.retry_attempts == 3  # settings restored
        assert fresh.output_folder == opt.output_folder

    def test_load_state_with_cold_skill_dir_clears_members(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = MultiSurfaceOptimizer(formula, output_folder=str(tmp_path / "r"))
        opt.load_state({"formula_params": {"skill_dir": ""}})
        assert formula.skills.skill_names == []


# ── objective helpers ────────────────────────────────────────────────────────


class TestObjective:
    def test_normalize_default_and_errors(self):
        assert normalize_weights(None) == {"TaskSuccessScore": 1.0}
        assert normalize_weights({"A": 1, "B": 3}) == {"A": 1.0, "B": 3.0}
        with pytest.raises(ValueError):
            normalize_weights({"A": 0})
        with pytest.raises(ValueError):
            normalize_weights({"A": -1, "B": 2})

    def test_terms_and_total(self):
        r = Reward(reward=1.0, metadata={"scores": {"Conciseness": 0.5, "Ignored": 0.0}})
        w = {"TaskSuccessScore": 1.0, "Conciseness": 1.0}
        terms = objective_terms(r, w)
        assert terms == {"TaskSuccessScore": 1.0, "Conciseness": 0.5}
        assert weighted_total(terms, w) == pytest.approx(0.75)

    def test_bool_score_is_not_numeric(self):
        r = Reward(reward=1.0, metadata={"scores": {"X": True}})
        with pytest.raises(ValueError, match="X"):
            objective_terms(r, {"X": 1.0})


# ── renderers ────────────────────────────────────────────────────────────────


class TestRenderers:
    def test_validate_surface_skills_layout(self, tmp_path):
        step = tmp_path / "s"
        write_skill(step / "skills" / "create", "good")
        # wrong depth
        write_skill(step / "skills", "stray")
        problems = renderers.validate_surface(str(step), "skills")
        assert len(problems) == 1 and "stray" in problems[0] and "must live under" in problems[0]

    def test_validate_surface_prompt_and_tools(self, tmp_path):
        step = tmp_path / "s"
        os.makedirs(step / "system_prompt")
        os.makedirs(step / "tool_descriptions")
        assert renderers.validate_surface(str(step), "system_prompt") == [
            "system_prompt/optimized_prompt.yaml was not written"
        ]
        (step / "system_prompt" / "optimized_prompt.yaml").write_text("system_prompt: ''\n")
        assert "non-empty" in renderers.validate_surface(str(step), "system_prompt")[0]
        (step / "system_prompt" / "optimized_prompt.yaml").write_text(": : bad\n")
        assert "not valid YAML" in renderers.validate_surface(str(step), "system_prompt")[0]
        (step / "tool_descriptions" / "optimized_tool_descriptions.yaml").write_text(
            "tool_descriptions:\n  search: 1\n"
        )
        assert "must be a string" in renderers.validate_surface(str(step), "tool_description")[0]

    def test_surface_attempted_is_presence_on_disk(self, tmp_path):
        step = tmp_path / "s"
        os.makedirs(step / "skills")  # empty dir: not an attempt
        assert not renderers.surface_attempted(str(step), "skills")
        os.makedirs(step / "system_prompt")  # dir exists, file missing: an attempt
        assert renderers.surface_attempted(str(step), "system_prompt")
        assert not renderers.surface_attempted(str(step), "tool_description")

    def test_read_findings_shapes(self, tmp_path):
        step = tmp_path / "s"
        os.makedirs(step)
        assert renderers.read_findings(str(step))[0] is None
        (step / "findings.json").write_text('{"findings": "nope"}')
        assert "findings` array" in renderers.read_findings(str(step))[1]
        (step / "findings.json").write_text('{"findings": []}')
        assert renderers.read_findings(str(step)) == ({"findings": []}, "")

    def test_render_deployed_skills_caps_body(self, tmp_path):
        d = write_skill(tmp_path, "long", body="\n".join(f"line {i}" for i in range(300)))
        out = renderers.render_deployed_skills({"long": d}, max_body_lines=10)
        assert out.startswith("----- long")
        assert "more lines; full file at" in out
        assert renderers.render_deployed_skills({}) == "  (none deployed)"

    def test_census_error_heuristic_without_status(self, tmp_path):
        folder = tmp_path / "t"
        folder.mkdir()
        trace = {
            "reward": 0.0,
            "response": {
                "messages": [
                    {"role": "assistant", "content": [{"toolUse": {"name": "x", "input": {}}}]},
                    {
                        "role": "user",
                        "content": [{"toolResult": {"content": [{"text": "Traceback ..."}]}}],
                    },
                ]
            },
        }
        (folder / "a.json").write_text(json.dumps(trace))
        (folder / "notes.txt").write_text("ignored")
        summaries = renderers.load_trace_summaries(str(folder))
        assert len(summaries) == 1 and summaries[0].calls[0].is_error
        census = renderers.render_census(summaries, ["x", "y"])
        assert "never called in these traces: y" in census
        assert "1 traces by total weighted objective: 0 at or above 0.50, 1 below 0.50" in census


class TestReviewFixes:
    def test_stratified_sampling_follows_the_objective_total(self, tmp_path):
        """Raw rewards say t0,t1 succeeded; the weighted objective says t0,t2 did."""
        formula = make_formula(tmp_path)
        seen = {}

        def writer(step_dir, opt):
            seen["names"] = sorted(n for n in os.listdir(opt._temp_dir) if n.endswith(".json"))
            _write_findings(step_dir, [])

        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=writer,
            n_sample_traces=2,
            retry_backoff_s=0,
            objective_weights={"TaskSuccessScore": 0.5, "Quality": 0.5},
        )
        rollouts, _ = make_rollouts()
        raw = [1.0, 1.0, 0.0, 0.0]
        quality = [0.9, -0.8, 1.0, 0.1]  # totals 0.95, 0.10, 0.50, 0.05 -> hi {t0,t2}, lo {t1,t3}
        rewards = [
            Reward(reward=raw[i], metadata={"scores": {"Quality": quality[i]}}) for i in range(4)
        ]
        opt.add_rollouts(rollouts)
        opt.add_rewards(rewards)
        opt.step()
        picked = {n.split("_")[0] for n in seen["names"]}
        assert len(picked) == 2
        assert len(picked & {"t0", "t2"}) == 1, picked  # one from the objective-high bucket
        assert len(picked & {"t1", "t3"}) == 1, picked  # one from the objective-low bucket

    def test_missing_rewards_are_refused_before_sampling(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=write_findings_only,
            retry_backoff_s=0,
        )
        rollouts, rewards = make_rollouts()
        opt.add_rollouts(rollouts)
        opt.add_rewards(rewards[:-1])  # one short
        with pytest.raises(ValueError, match="needs a Reward for every rollout"):
            opt.step()
        assert opt.task_messages == []
        assert not os.path.exists(tmp_path / "runs" / "step_0001")

    def test_extra_rewards_are_refused_too(self, tmp_path):
        """Index alignment is the contract; a surplus is as wrong as a shortage."""
        formula = make_formula(tmp_path)
        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=write_findings_only,
            retry_backoff_s=0,
        )
        rollouts, rewards = make_rollouts()
        opt.add_rollouts(rollouts[:1])
        opt.add_rewards(rewards[:2])
        with pytest.raises(ValueError, match="1 rollout\\(s\\) but 2 reward"):
            opt.step()

    def test_failed_materialize_restores_the_formula(self, tmp_path, monkeypatch):
        """An OSError while copying skills must not leave the prompt or tools changed."""
        formula = make_formula(tmp_path)
        before = formula.snapshot()

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(formula.skills, "materialize", boom)
        opt = make_opt(tmp_path, formula, write_all_three)
        with pytest.raises(OSError, match="disk full"):
            opt.step()
        assert formula.snapshot() == before
        assert formula.system_prompt.system_prompt == PROMPT
        assert formula.tool_descriptions.overrides == {}
        assert formula.skills.skill_names == ["alpha"]
        assert opt._step_count == 0
        assert os.path.exists(os.path.join(tmp_path, "runs", "step_0001", FAILED_FILE))

    def test_strict_tool_rejection_restores_prompt(self, tmp_path):
        """Strict tool formula raises on the third member; the first must be rolled back."""

        def writer(step_dir, opt):
            _write_prompt(step_dir, PROMPT + "\nEdit.")
            _write_tools(step_dir, {"search": "ok", "ghost": "x"})
            _write_findings(step_dir, [{"id": "f01"}])

        formula = MultiSurfaceFormula(
            system_prompt=SystemPromptFormula(system_prompt=PROMPT),
            tool_descriptions=ToolDescriptionFormula(BASE_TOOLS, strict=True),
        )
        opt = make_opt(tmp_path, formula, writer)
        with pytest.raises(ValueError, match="ghost"):
            opt.step()
        assert formula.system_prompt.system_prompt == PROMPT
        assert formula.tool_descriptions.overrides == {}

    def test_sampling_falls_back_to_the_median_when_nothing_separates(self, tmp_path, caplog):
        """All totals above 0.5: split at the median so the sample still has contrast."""
        formula = make_formula(tmp_path)
        seen = {}

        def writer(step_dir, opt):
            seen["names"] = sorted(n for n in os.listdir(opt._temp_dir) if n.endswith(".json"))
            _write_findings(step_dir, [])

        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=writer,
            n_sample_traces=2,
            retry_backoff_s=0,
        )
        rollouts, _ = make_rollouts()
        totals = [0.6, 0.7, 0.9, 1.0]  # median 0.8 -> lo {t0,t1}, hi {t2,t3}
        opt.add_rollouts(rollouts)
        opt.add_rewards([Reward(reward=t) for t in totals])
        with caplog.at_level(logging.INFO):
            opt.step()
        picked = {n.split("_")[0] for n in seen["names"]}
        assert len(picked & {"t0", "t1"}) == 1, picked
        assert len(picked & {"t2", "t3"}) == 1, picked
        assert "pool median" in caplog.text

    def test_sampling_warns_when_objective_is_constant(self, tmp_path, caplog):
        formula = make_formula(tmp_path)
        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=write_findings_only,
            n_sample_traces=2,
            retry_backoff_s=0,
        )
        rollouts, _ = make_rollouts()
        opt.add_rollouts(rollouts)
        opt.add_rewards([Reward(reward=1.0) for _ in rollouts])
        with caplog.at_level(logging.WARNING):
            opt.step()
        assert "constant" in caplog.text

    def test_all_rejected_tool_yaml_is_dropped_not_applied(self, tmp_path):
        def writer(step_dir, opt):
            _write_tools(step_dir, {"ghost": "unknown tool", "search": "x" * 801})
            _write_findings(step_dir, [{"id": "f01"}])

        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, writer)
        with pytest.raises(RuntimeError, match="none is usable"):
            opt.step()
        assert formula.tool_descriptions.overrides == {}

    def test_partially_rejected_tool_yaml_applies_the_rest(self, tmp_path, caplog):
        def writer(step_dir, opt):
            _write_tools(step_dir, {"ghost": "unknown tool", "search": "ok"})
            _write_findings(step_dir, [{"id": "f01"}])

        formula = make_formula(tmp_path)
        opt = make_opt(tmp_path, formula, writer)
        with caplog.at_level(logging.WARNING):
            opt.step()
        assert opt.last_applied == ["tool_description"]
        assert formula.tool_descriptions.overrides == {"search": "ok"}
        assert "will be rejected" in caplog.text

    def test_explanations_are_labelled_with_trace_filenames(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=write_findings_only,
            n_sample_traces=-1,
            retry_backoff_s=0,
            objective_weights={"TaskSuccessScore": 0.5, "Conciseness": 0.5},
        )
        rollouts, rewards = make_rollouts(scores=[{"Conciseness": 0.2}] * 4)
        rewards[2].metadata["explanations"] = {"Conciseness": "Repeated the cart."}
        opt.add_rollouts(rollouts)
        opt.add_rewards(rewards)
        opt.step()
        task = opt.task_messages[0]
        assert "on `t2_0002.json`: Repeated the cart." in task
        assert "episode 2" not in task

    def test_prompt_yaml_is_byte_exact_for_any_trailing_newlines(self):
        from strands_harness_optimizer.optimizers.multi_surface.multi_surface import _yaml_block

        for text in ["no newline", "one\n", "two\n\n", "", "\n", 'q: "x" <t>\n#h\n\n\n']:
            assert yaml.safe_load(_yaml_block("system_prompt", text))["system_prompt"] == text


class TestObjectiveDefinitions:
    def _report(self, weights, definitions=None):
        rewards = [
            Reward(reward=1.0, metadata={"scores": {"Conciseness": 0.5, "MyScore": 0.3}}),
            Reward(reward=0.0, metadata={"scores": {"Conciseness": 1.0, "MyScore": 0.7}}),
        ]
        return renderers.render_objective_report(
            rewards, normalize_weights(weights), ["a.json", "b.json"], definitions
        )

    def test_unconfigured_scores_never_appear(self):
        report = self._report({"TaskSuccessScore": 1.0})
        assert "Conciseness" not in report and "MyScore" not in report

    def test_user_definition_wins_over_builtin(self):
        report = self._report(
            {"Conciseness": 1.0}, {"Conciseness": "Our own: no repeated cart summaries."}
        )
        assert "Our own: no repeated cart summaries." in report
        assert "appropriately brief" not in report

    def test_builtin_definition_is_the_fallback_for_a_matching_name(self):
        report = self._report({"Conciseness": 1.0})
        assert "appropriately brief" in report

    def test_custom_term_without_definition_is_flagged(self, caplog):
        with caplog.at_level(logging.WARNING):
            report = self._report({"MyScore": 1.0})
        assert "| MyScore | 1.00 | (no definition supplied) |" in report
        assert "MyScore" in caplog.text and "objective_definitions" in caplog.text

    def test_custom_term_with_definition_rendered(self, tmp_path):
        formula = make_formula(tmp_path)
        opt = StubOptimizer(
            formula,
            output_folder=str(tmp_path / "runs"),
            writer=write_findings_only,
            n_sample_traces=-1,
            retry_backoff_s=0,
            objective_weights={"TaskSuccessScore": 0.5, "MyScore": 0.5},
            objective_definitions={"MyScore": "Fraction of user constraints honoured."},
        )
        rollouts, rewards = make_rollouts(scores=[{"MyScore": 0.5}] * 4)
        opt.add_rollouts(rollouts)
        opt.add_rewards(rewards)
        opt.step()
        assert "| MyScore | 0.50 | Fraction of user constraints honoured. |" in opt.task_messages[0]
        assert opt.get_state()["objective_definitions"] == {
            "MyScore": "Fraction of user constraints honoured."
        }
