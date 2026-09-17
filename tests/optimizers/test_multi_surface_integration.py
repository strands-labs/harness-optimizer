"""Integration test for MultiSurfaceOptimizer against Bedrock.

A contract smoke test: the reflector runs on real or fabricated traces, the step
completes, ``findings.json`` parses, and the formula state matches the surfaces
the optimizer reports as applied. It asserts NOTHING about which surfaces the
model chose to edit -- a real run may legitimately edit none of them -- and it
does not require every file on disk to be valid, because an invalid artifact is
deliberately kept on disk after being dropped.

Requires AWS credentials; skipped otherwise. Set ``MULTI_SURFACE_TRACES_DIR`` to
a folder of trace JSON files (``reward`` + ``response.messages``) to run on real
traces instead of the fabricated ones.
"""

import json
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

has_aws_credentials = bool(os.environ.get("AWS_ACCESS_KEY_ID"))

PROMPT = """You are an online shopping assistant.

Find and buy the product that matches every requirement the user states.
Use search to find candidates and click to open a product or an option.
Finish by clicking "buy now"."""

TOOLS = {
    "search": "Search for products using space-separated keywords.",
    "click": "Click an element on the page, such as a product, an option, or a button.",
}


def _fabricated(n=8):
    rollouts, rewards = [], []
    for i in range(n):
        ok = i % 2 == 1
        msgs = [
            {"role": "user", "content": [{"text": f"Buy a blue cotton shirt under $30, size {i}"}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"name": "search", "input": {"query": "blue shirt"}}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "status": "success",
                            "content": [{"text": "B001 Blue Shirt $25\nB002 Navy Shirt $40"}],
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"name": "click", "input": {"element": "B001" if ok else "B002"}}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "status": "success",
                            "content": [{"text": "Product page. Options: size"}],
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"toolUse": {"name": "click", "input": {"element": "buy now"}}}],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"status": "success", "content": [{"text": "Purchased"}]}}
                ],
            },
            {"role": "assistant", "content": [{"text": "Done."}]},
        ]
        rollouts.append(
            Rollout(
                data_sample={"task_id": f"fab-{i}"},
                messages=msgs,
                metadata={"eval_result": {"reward": 1.0 if ok else 0.25, "success": ok}},
            )
        )
        rewards.append(Reward(reward=1.0 if ok else 0.25))
    return rollouts, rewards


def _from_dir(folder):
    rollouts, rewards = [], []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name)) as f:
            d = json.load(f)
        messages = (d.get("response") or {}).get("messages") or []
        if not messages:
            continue
        reward = float(d.get("reward") or 0.0)
        rollouts.append(
            Rollout(
                data_sample={"task_id": str(d.get("task_id") or name[:-5])},
                messages=messages,
                metadata={
                    "eval_result": (d.get("response") or {}).get("eval_result")
                    or d.get("eval_result")
                    or {}
                },
            )
        )
        rewards.append(Reward(reward=reward))
    return rollouts, rewards


@pytest.mark.integration
@pytest.mark.skipif(not has_aws_credentials, reason="AWS credentials not available")
def test_multi_surface_step_contract(tmp_path):
    traces_dir = os.environ.get("MULTI_SURFACE_TRACES_DIR")
    rollouts, rewards = _from_dir(traces_dir) if traces_dir else _fabricated()
    assert rollouts, "no traces to run on"

    formula = MultiSurfaceFormula(
        system_prompt=SystemPromptFormula(system_prompt=PROMPT),
        skills=SkillLibraryFormula(),
        tool_descriptions=ToolDescriptionFormula(TOOLS),
    )
    optimizer = MultiSurfaceOptimizer(
        formula,
        output_folder=str(tmp_path / "runs"),
        n_sample_traces=min(8, len(rollouts)),
        model_config={
            "model_id": os.environ.get(
                "MULTI_SURFACE_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
            )
        },
    )
    optimizer.add_rollouts(rollouts)
    optimizer.add_rewards(rewards)
    optimizer.step()

    step_dir = optimizer.last_step_dir
    assert step_dir and step_dir.endswith("step_0001")
    assert optimizer._step_count == 1

    # The ledger is always written and parses.
    findings, problem = renderers.read_findings(step_dir)
    assert findings is not None, problem

    # Formula state matches what the optimizer reports as applied.
    applied = set(optimizer.last_applied)
    if "system_prompt" in applied:
        with open(os.path.join(step_dir, renderers.PROMPT_FILE)) as f:
            assert formula.system_prompt.system_prompt == yaml.safe_load(f)["system_prompt"]
    else:
        assert formula.system_prompt.system_prompt == PROMPT
    if "tool_description" in applied:
        assert formula.tool_descriptions.overrides
        assert set(formula.tool_descriptions.overrides) <= set(TOOLS)
    else:
        assert formula.tool_descriptions.overrides == {}
    if "skills" in applied:
        assert formula.skills.skill_names
        assert formula.get_tunable_params()["skill_dir"].endswith("skill_set")
    else:
        assert formula.skills.skill_names == []

    print(
        f"\n[multi-surface smoke] applied={sorted(applied)} dropped={sorted(optimizer.last_dropped)} "
        f"findings={len(findings['findings'])} metrics={optimizer.last_metrics} "
        f"wall={optimizer.last_wall_clock_s:.0f}s"
    )
