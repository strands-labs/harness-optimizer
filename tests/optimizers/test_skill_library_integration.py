"""Integration test for the skill library optimizer with a real Bedrock model.

Drives the full loop against Bedrock -- fabricated rollouts (no environment
needed), a real curator agent reading the traces and writing a decision tree,
and the formula applying it. Verifies the pieces the offline unit tests cannot:
that the built-in prompts actually elicit the ``create/optimize/retire.txt``
tree, that the agent tolerates the tool-set the optimizer hands it, and that
``manifest.json`` is written when we expect.

Requires AWS credentials; skipped otherwise.
"""

import json
import os

import pytest

from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SkillLibraryFormula
from strands_harness_optimizer.optimizers import SkillLibraryOptimizer
from strands_harness_optimizer.utils import load_builtin_template

has_aws_credentials = bool(os.environ.get("AWS_ACCESS_KEY_ID"))


def _make_rollout(task: str, answer: str, reward: float):
    """Fabricate a rollout that looks like an invoke.py trace to the curator.

    Real rollouts carry their agent messages under ``response.messages`` in the
    trace file, and the curator's Phase 3 skeleton extractor
    (``_traj.py``) reads that path. Building the same shape here means the
    prompt runs against realistic-looking traces without needing a live agent.
    """
    return Rollout(
        data_sample={"task_id": task[:20], "task": task},
        messages=[
            {"role": "user", "content": [{"text": task}]},
            {"role": "assistant", "content": [{"text": answer}]},
        ],
        metadata={
            "response_text": answer,
            # invoke.py-style envelope, so a raw JSON read in the curator's
            # Phase-2 census (grep '"reward"') and its Phase-3 skeleton
            # (response.messages) both find what they expect.
            "response": {
                "messages": [
                    {"role": "user", "content": [{"text": task}]},
                    {"role": "assistant", "content": [{"text": answer}]},
                ],
            },
        },
    ), Reward(reward=reward)


@pytest.mark.integration
@pytest.mark.skipif(not has_aws_credentials, reason="AWS credentials not available")
class TestSkillLibraryEndToEnd:

    def test_cold_start_produces_a_decision_tree(self, tmp_path):
        """The whole loop: rollouts -> curator -> decisions on disk -> set updated.

        Two successful traces solving the same class of task, three failures on
        another -- enough for the curator to see a recurring pattern in each and,
        under its own gates, choose CREATE or SKIP. Either is a valid outcome;
        this test asserts only that the loop completes and the artifacts land
        where the contract says they will.
        """
        formula = SkillLibraryFormula()  # cold start
        assert formula.skill_names == []

        out_dir = tmp_path / "iter_0"
        optimizer = SkillLibraryOptimizer(
            formula,
            system_prompt_template=load_builtin_template("skill_library/system_prompt.jinja"),
            task_message_template=load_builtin_template("skill_library/task_message.jinja"),
            output_folder=str(out_dir),
            model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
            n_sample_traces=-1,
        )

        rollouts, rewards = zip(
            *[
                _make_rollout("Convert 2 hours to minutes.", "2 hours is 120 minutes.", 1.0),
                _make_rollout("Convert 3 hours to minutes.", "3 hours is 180 minutes.", 1.0),
                _make_rollout("What is 15% of 200?", "I think it's around 25 or 30.", 0.0),
                _make_rollout("What is 20% of 150?", "Not sure, maybe 25?", 0.0),
                _make_rollout("What is 10% of 400?", "Something like 30 or 50.", 0.0),
            ]
        )
        optimizer.add_rollouts(list(rollouts))
        optimizer.add_rewards(list(rewards))

        optimizer.step()

        # The curator either wrote decisions or produced a manifest-only SKIP.
        # Both are legitimate; the loop must have completed and left the run's
        # record on disk either way.
        manifest = out_dir / "manifest.json"
        assert manifest.exists(), (
            "output_folder must contain manifest.json after step() -- the run's "
            "record of what the curator decided and why"
        )
        parsed = json.loads(manifest.read_text())
        assert isinstance(
            parsed, dict
        ), f"manifest.json must be JSON object, got {type(parsed).__name__}"

        # step_count increments whether decisions were applied or the curator
        # correctly SKIPped -- both are legitimate outcomes.
        assert optimizer._step_count == 1

        # If the curator wrote any skills, they must be loadable: the whole point
        # of frontmatter validation is that a malformed skill is caught here,
        # not silently at evaluation time.
        for name, path in formula.members.items():
            skill_md = os.path.join(path, "SKILL.md")
            assert os.path.isfile(skill_md), f"{name}: SKILL.md missing at {path}"
            head = open(skill_md).read()[:2000]
            assert head.lstrip().startswith("---"), (
                f"{name}: SKILL.md is missing frontmatter -- the runtime could "
                "not load it. Formula validation should have rejected this."
            )
            assert "description:" in head, (
                f"{name}: frontmatter has no `description:` -- the runtime "
                "reads that field to decide whether to load the skill."
            )

    def test_curator_can_see_deployed_skills(self, tmp_path):
        """A skill in ``skill_folder`` reaches the prompt through ``skill_index``.

        Populates the library with one obvious-if-read skill, hands the curator
        rollouts that would have benefited from it, and checks that whatever the
        curator did, its manifest reflects awareness of the deployed set (either
        by naming the skill in decisions or by explaining the SKIP in terms of
        it). Without that awareness the whole "already covered" test in the
        prompt is a no-op.
        """
        # Set up an existing library with one skill.
        skills = tmp_path / "skills" / "percentage-calculator"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text(
            "---\n"
            "name: percentage-calculator\n"
            "description: Use when computing X% of Y. Multiply X/100 by Y.\n"
            "---\n\n"
            "# Percentage calculator\n\n"
            "## Workflow\n"
            "1. Read X (percent) and Y (base).\n"
            "2. Compute X/100 * Y.\n"
            "3. Return the number, no words.\n"
        )
        formula = SkillLibraryFormula(str(tmp_path / "skills"))
        assert formula.skill_names == ["percentage-calculator"]

        out_dir = tmp_path / "iter_0"
        optimizer = SkillLibraryOptimizer(
            formula,
            system_prompt_template=load_builtin_template("skill_library/system_prompt.jinja"),
            task_message_template=load_builtin_template("skill_library/task_message.jinja"),
            output_folder=str(out_dir),
            model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
            n_sample_traces=-1,
        )

        # Rollouts that would benefit from the deployed skill IF the agent had
        # actually invoked it. The curator's "already covered" test says: this
        # is a trigger/discoverability problem, so OPTIMIZE the trigger or SKIP
        # -- do NOT write a duplicate skill.
        rollouts, rewards = zip(
            *[
                _make_rollout("What is 15% of 200?", "About 25 or 30.", 0.0),
                _make_rollout("What is 20% of 150?", "Maybe 25.", 0.0),
                _make_rollout("What is 10% of 400?", "30-something?", 0.0),
            ]
        )
        optimizer.add_rollouts(list(rollouts))
        optimizer.add_rewards(list(rewards))

        optimizer.step()

        manifest = out_dir / "manifest.json"
        assert manifest.exists()

        # Whatever the curator chose, the deployed skill's name should appear
        # somewhere in its record -- either in an OPTIMIZE targeting it, in a
        # SKIP citing it as already-covering the pattern, or as a covered_by
        # reference in the partition/cluster tables. If it never appears, the
        # curator either did not read the index or did not reason about it.
        blob = manifest.read_text()
        assert "percentage-calculator" in blob, (
            "The deployed skill's name did not appear anywhere in manifest.json, "
            "so the curator either did not receive the skill_index or ignored "
            f"it. Manifest content:\n{blob[:2000]}"
        )
