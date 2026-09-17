# Optimizers

Optimizers analyze agent rollouts and propose improved Formula parameters. The optimization loop follows a PyTorch-inspired pattern: accumulate data, then step to update.

Rollouts are scored by a [RewardFunction](./rewards.md) before being fed to the optimizer.

## FormulaOptimizer

`FormulaOptimizer` follows a PyTorch-style interface:

1. Initialize with a Formula (like `torch.optim.Adam(model.parameters())`)
2. Accumulate rollouts and rewards (like `loss.backward()`)
3. Call `step()` to update Formula parameters (like `optimizer.step()`)
4. Call `zero()` to reset (like `optimizer.zero_grad()`)

Rollouts are expected to contain the data sample information alongside the conversation trace (in the `data_sample` key), so there is no separate `add_data_samples()` method.

```python
from strands_harness_optimizer.optimizers import FormulaOptimizer

optimizer = MyOptimizer(formula)

# Accumulate data (rollouts contain data_sample)
optimizer.add_rollouts(rollouts)
optimizer.add_rewards(rewards)

# Update formula parameters
optimizer.step()

# Reset for next round
optimizer.zero()
```

### Creating a custom optimizer

Subclass `FormulaOptimizer` and implement `step()`:

```python
class MyOptimizer(FormulaOptimizer):
    def step(self):
        # Access accumulated data via self._rollouts and self._rewards
        # Each rollout contains "data_sample" with the input task data
        best_idx = max(range(len(self._rewards)),
                       key=lambda i: self._rewards[i]["reward_value"])
        # Update formula parameters
        self.formula.update_params({"system_prompt": "Improved prompt..."})
```

### Checkpointing

By default, `get_state()` and `load_state()` raise `NotImplementedError`. Override them in subclasses that maintain state across steps:

```python
class StatefulOptimizer(FormulaOptimizer):
    def __init__(self, formula):
        super().__init__(formula)
        self.history = []

    def step(self):
        self.history.append(len(self._rollouts))
        self.formula.update_params(self.formula.get_tunable_params())

    def get_state(self):
        return {"history": self.history}

    def load_state(self, state):
        self.history = state["history"]
```

## BaseAgenticOptimizer

`BaseAgenticOptimizer` is the base class for optimizers that use a strands Agent with tools to analyze rollouts. It is Formula-agnostic — `ContrastiveReflectionOptimizer` (system prompts, skill text) and `SkillLibraryOptimizer` (a whole skill library) both build on it. It provides infrastructure that specific agentic optimizers build on:

- In-memory trace sampling (random or stratified by reward)
- Writing sampled traces to temp folders as JSON
- Agent creation with configurable `model_config` and boto settings
- `ToolOutputGuardrail` for truncating excessive tool output
- `submit_optimized_params` tool for reliable parameter extraction (file-per-param)
- Full `get_state()`/`load_state()` checkpointing (config + step_count + prompt_history)

Subclass it to build custom agentic optimizers:

```python
from strands_harness_optimizer.optimizers import BaseAgenticOptimizer

class MyAgenticOptimizer(BaseAgenticOptimizer):
    def step(self):
        indices = self._sample_traces()
        traces_folder = self._write_traces_to_temp(indices)
        agent = self._create_agent("Analyze traces and optimize...")
        agent(f"Analyze {traces_folder}")
        params = self._get_submitted_params()
        if params:
            self.formula.update_params(params)
        self._cleanup_temp()
```

## ToolOutputGuardrail

`ToolOutputGuardrail` truncates excessive tool output and warns the agent. It's automatically registered on agents created by `BaseAgenticOptimizer`, but can be used independently:

```python
from strands_harness_optimizer.utils.guardrails import ToolOutputGuardrail

guardrail = ToolOutputGuardrail(max_chars=50000)
guardrail.register(agent)
```

## MultiSurfaceOptimizer

`MultiSurfaceOptimizer` edits the three text surfaces an agent reads — system prompt, skill library, tool descriptions — in one pass. Its reflector reads a sample of traces plus a harness-computed census and objective table, is shown the current prompt, the deployed skills in full and the current tool descriptions, and places each finding in the one surface that reaches the agent when it matters. It takes a `MultiSurfaceFormula`.

```python
from strands_harness_optimizer.formulas import (
    MultiSurfaceFormula, SkillLibraryFormula, SystemPromptFormula, ToolDescriptionFormula,
)
from strands_harness_optimizer.optimizers import MultiSurfaceOptimizer

formula = MultiSurfaceFormula(
    system_prompt=SystemPromptFormula(system_prompt=PROMPT),
    skills=SkillLibraryFormula(skill_dir="./skills"),                     # None for a cold start
    tool_descriptions=ToolDescriptionFormula.from_yaml("./tool_descriptions.yaml"),
)
optimizer = MultiSurfaceOptimizer(formula, output_folder="./runs", n_sample_traces=20)
optimizer.add_rollouts(rollouts)
optimizer.add_rewards(rewards)
optimizer.step()

formula.get_tunable_params()
# {'system_prompt': '...', 'skill_dir': './runs/step_0001/skill_set', 'tool_descriptions': {'search': '...'}}
```

Every `step()` writes into a fresh `output_folder/step_NNNN/`. The harness writes `current/` (the prompt and tool descriptions the agent started from, and the rendered prompts it was given); the agent writes only the surfaces it changed — `skills/create/<name>/SKILL.md`, `skills/update/<name>/SKILL.md`, `system_prompt/optimized_prompt.yaml`, `tool_descriptions/optimized_tool_descriptions.yaml` — plus a `findings.json` ledger, always. After applying, the harness materializes the resolved skill set to `skill_set/` and repoints the formula at it. Earlier step directories are never read again, and a failed step leaves a `FAILED.txt` behind.

Validation is structural: an artifact that could not take effect (a skill without frontmatter, a prompt YAML that does not parse, a tool-description file without a mapping) is dropped with a logged reason while the other surfaces apply. If every attempted surface is invalid the step raises. A pass that edits nothing is legitimate as long as `findings.json` was written; no artifacts and no ledger raises, because an empty pass and a crashed agent would otherwise be indistinguishable.

**Objective.** By default the objective is the single term `TaskSuccessScore = Reward.reward`. To optimize a weighted combination of scores, put the components in the reward's metadata and name them:

```python
Reward(reward=0.5, metadata={
    "scores": {"TaskSuccessScore": 1.0, "Conciseness": 0.0},
    "explanations": {"Conciseness": "The agent restated the full cart three times ..."},
})
optimizer = MultiSurfaceOptimizer(formula, output_folder="./runs",
                                  objective_weights={"TaskSuccessScore": 0.5, "Conciseness": 0.5})
```

Every configured term must be scored on every reward; a missing term raises before the agent runs. Scores not named in the weights are never shown to the agent. Explanations, where present, are shown for episodes the judge scored below perfect.

Tell the agent what each score measures with `objective_definitions={"Conciseness": "..."}`. Your wording wins; a term named after an Amazon Bedrock AgentCore built-in evaluator (`GoalSuccessRate`, `Conciseness`, `Helpfulness`, ...) falls back to that evaluator's own definition; a term with neither is shown as undefined and logged. `TaskSuccessScore` is the reserved name for `Reward.reward`. Weights need not sum to one; the total is a weighted mean, and it is what the trace's `reward` field holds and what stratified sampling classifies on. Your raw `Reward.reward` is kept in the trace under `data.reward` and inside `eval_result`.

**Reflector settings.** The agent has `shell` and `editor`, runs with a sliding window of 120 messages with the task message pinned (`window_size`, `pin_first`, `compression_threshold`), and is rebuilt and re-run on a transient Bedrock error (`retry_attempts`, `retry_backoff_s`); agent-written paths are cleared between attempts. `agent_tools` names the toolset for the census and defaults to the tool-description formula's base.

**Delivery to a runtime.** `formula.get_tunable_params()` gives the three things to ship: the prompt string, the materialized skill directory (pack it inline as `skills_folder`, see the skill-library examples), and the sparse `tool_descriptions` overrides. A runtime applies the overrides by patching only the named tools; every other tool keeps its own description.

## Built-in Optimizers

- [ContrastiveReflectionOptimizer](./optimizers/contrastive-reflection.md) — contrastive learning on rollout traces
- `SkillLibraryOptimizer` — curate a library of skills from rollout traces (see [Formulas](formulas.md#using-skilllibraryformula))
- `MultiSurfaceOptimizer` — prompt, skills and tool descriptions in one pass (above)

## What's Next

- [Training](training.md) — automated training loop with Dataset, DataLoader, and Trainer
