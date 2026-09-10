"""Tests for SkillLibraryOptimizer.

The curator agent is stubbed: `_run_curation` is replaced with a function that
writes the decision tree a real agent would write. That keeps these tests offline
while still exercising the contract that matters — what the optimizer does with
what it finds on disk.
"""

import json
import os

import pytest

from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SkillFormula, SkillLibraryFormula
from strands_harness_optimizer.optimizers import SkillLibraryOptimizer

FRONTMATTER = "---\nname: {name}\ndescription: Use when {desc}.\n---\n\n{body}\n"


def write_skill(root, name, desc="testing", body="Do the thing."):
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(FRONTMATTER.format(name=name, desc=desc, body=body))
    return d


def make_optimizer(formula, output_folder, **kwargs):
    return SkillLibraryOptimizer(
        formula,
        system_prompt_template="persona: {{ skill_folder }}",
        task_message_template="traces: {{ traces_folder }} out: {{ output_folder }}",
        output_folder=str(output_folder),
        **kwargs,
    )


def add_rollouts(opt, n=2):
    """Feed the optimizer enough graded rollouts to have something to sample."""
    opt.add_rollouts(
        [
            Rollout(data_sample={"task_id": f"t{i}"}, messages=[{"role": "user", "content": "x"}])
            for i in range(n)
        ]
    )
    opt.add_rewards([Reward(reward=1.0 if i % 2 else 0.0) for i in range(n)])


def stub_curation(opt, writer):
    """Replace the LLM call with `writer(output_folder)`."""
    opt._run_curation = lambda traces_folder: writer(opt.output_folder)


class TestInit:
    def test_rejects_a_non_set_formula(self, tmp_path):
        """SkillFormula tunes one skill's text; it has no set to curate."""
        from strands import Skill

        formula = SkillFormula(Skill(name="a", description="d", instructions="i"))
        with pytest.raises(TypeError, match="SkillLibraryFormula"):
            make_optimizer(formula, tmp_path / "out")

    def test_accepts_a_skill_library_formula(self, tmp_path):
        opt = make_optimizer(SkillLibraryFormula(), tmp_path / "out")
        assert opt.output_folder == str(tmp_path / "out")

    def test_drops_the_submit_tool_instructions(self, tmp_path):
        """The curator writes files itself; submit_optimized_params does not apply."""
        opt = make_optimizer(SkillLibraryFormula(), tmp_path / "out")
        assert opt.system_prompt_suffix == ""

    def test_drops_the_submit_tool_itself(self, tmp_path):
        """An unused tool in the schema is an invitation to call it."""
        opt = make_optimizer(SkillLibraryFormula(), tmp_path / "out")
        agent = opt._create_agent("persona")
        assert sorted(agent.tool_names) == ["shell"]


class TestStep:
    def test_no_rollouts_is_a_noop(self, tmp_path):
        opt = make_optimizer(SkillLibraryFormula(), tmp_path / "out")
        opt.step()  # must not raise
        assert opt.formula.skill_names == []

    def test_applies_a_create(self, tmp_path):
        formula = SkillLibraryFormula()
        opt = make_optimizer(formula, tmp_path / "out")
        add_rollouts(opt)

        def writer(out):
            write_skill(os.path.join(out, "create"), "new-skill")
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump({"actions": [{"action": "create", "skill_name": "new-skill"}]}, f)

        stub_curation(opt, writer)
        opt.step()
        assert formula.skill_names == ["new-skill"]
        assert opt.last_decisions["create"]

    def test_applies_a_retire(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        write_skill(existing, "beta")
        formula = SkillLibraryFormula(str(existing))
        opt = make_optimizer(formula, tmp_path / "out")
        add_rollouts(opt)

        stub_curation(opt, lambda out: open(os.path.join(out, "retire.txt"), "w").write("beta\n"))
        opt.step()
        assert formula.skill_names == ["alpha"]


class TestSkipIsValid:
    """A library whose recurring patterns are covered should stop, not churn."""

    def test_manifest_only_is_accepted(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        formula = SkillLibraryFormula(str(existing))
        opt = make_optimizer(formula, tmp_path / "out")
        add_rollouts(opt)

        def writer(out):
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump({"actions": [{"action": "skip", "evidence": "no recurring gaps"}]}, f)

        stub_curation(opt, writer)
        opt.step()  # must NOT raise
        assert formula.skill_names == ["alpha"]
        assert opt.last_decisions == {"create": [], "optimize": [], "retire": []}

    def test_skip_still_counts_as_a_step(self, tmp_path):
        opt = make_optimizer(SkillLibraryFormula(), tmp_path / "out")
        add_rollouts(opt)
        stub_curation(
            opt, lambda out: open(os.path.join(out, "manifest.json"), "w").write('{"actions": []}')
        )
        opt.step()
        assert opt._step_count == 1

    def test_nothing_at_all_raises(self, tmp_path):
        """No decisions AND no manifest means the agent did not do the work."""
        opt = make_optimizer(SkillLibraryFormula(), tmp_path / "out")
        add_rollouts(opt)
        stub_curation(opt, lambda out: None)
        with pytest.raises(RuntimeError, match="manifest"):
            opt.step()


class TestTemplateVars:
    def test_receives_the_three_inputs_plus_index(self, tmp_path):
        write_skill(tmp_path / "skills", "alpha", desc="searching a catalog")
        formula = SkillLibraryFormula(str(tmp_path / "skills"))

        seen = {}
        opt = SkillLibraryOptimizer(
            formula,
            system_prompt_template="{{ skill_index }}",
            task_message_template=("{{ traces_folder }}|{{ output_folder }}|{{ skill_folder }}"),
            output_folder=str(tmp_path / "out"),
        )

        def fake_create_agent(system_prompt):
            seen["system"] = system_prompt

            class _A:
                pass

            return _A()

        opt._create_agent = fake_create_agent
        opt._invoke_agent = lambda agent, message: seen.setdefault("task", message)
        add_rollouts(opt)
        stub = opt._run_curation
        # Run the real _run_curation, with the agent construction/invocation faked.
        opt._run_curation = stub
        try:
            opt.step()
        except RuntimeError:
            pass  # the fake agent writes nothing; we only care about the render

        assert "searching a catalog" in seen["system"]
        traces, out, skills = seen["task"].split("|")
        assert os.path.isabs(out) and out.endswith("out")
        assert skills.endswith("skills")
        assert "contrastive_traces_" in traces

    def test_cold_start_passes_an_empty_skill_folder(self, tmp_path):
        seen = {}
        opt = SkillLibraryOptimizer(
            SkillLibraryFormula(),
            system_prompt_template="x",
            task_message_template="[{{ skill_folder }}][{{ skill_index }}]",
            output_folder=str(tmp_path / "out"),
        )
        opt._create_agent = lambda sp: object()
        opt._invoke_agent = lambda agent, message: seen.setdefault("task", message)
        add_rollouts(opt)
        try:
            opt.step()
        except RuntimeError:
            pass
        assert seen["task"] == "[][(none deployed)]"


class TestCheckpointing:
    def test_round_trips_the_set(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        formula = SkillLibraryFormula(str(existing))
        opt = make_optimizer(formula, tmp_path / "out")
        state = json.loads(json.dumps(opt.get_state()))  # must be JSON-able

        restored = make_optimizer(SkillLibraryFormula(), tmp_path / "elsewhere")
        restored.load_state(state)
        assert restored.formula.skill_names == ["alpha"]
        assert restored.output_folder == str(tmp_path / "out")

    def test_state_records_skill_names(self, tmp_path):
        write_skill(tmp_path / "skills", "alpha")
        opt = make_optimizer(SkillLibraryFormula(str(tmp_path / "skills")), tmp_path / "out")
        assert opt.get_state()["skill_names"] == ["alpha"]
