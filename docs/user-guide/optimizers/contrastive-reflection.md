# ContrastiveReflectionOptimizer

The `ContrastiveReflectionOptimizer` uses an LLM agent with shell tools to analyze rollout traces contrastively — comparing successful vs failed trajectories — and generate optimized parameters.

## How it works

1. Traces are sampled from accumulated rollouts (in memory, with optional stratified sampling by reward)
2. Sampled traces are written to a temp folder as JSON files
3. A strands Agent with shell tool access analyzes the traces
4. The agent compares successful and failed traces to identify patterns
5. The agent writes optimized parameters to files and submits via the `submit_optimized_params` tool
6. The Formula is updated with the new parameters

## Quick start

```python
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.optimizers import ContrastiveReflectionOptimizer
from harness_optimizer.utils import load_builtin_template

formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")
optimizer = ContrastiveReflectionOptimizer(
    formula,
    system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
    task_message_template=load_builtin_template("contrastive_reflection/task_message_system_prompt.jinja"),
)

# After collecting rollouts and rewards...
optimizer.add_rollouts(rollouts)
optimizer.add_rewards(rewards)
optimizer.step()
optimizer.zero()
```

Both `system_prompt_template` and `task_message_template` are required. Use `load_builtin_template()` to load the built-in templates, or provide your own.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `system_prompt_template` | (required) | Jinja2 template for the reflection agent's system prompt |
| `task_message_template` | (required) | Jinja2 template for the task message |
| `model_config` | `{"model_id": "claude-sonnet-4-20250514", "temperature": 1.0, "streaming": True}` | Bedrock model configuration dict. Supports all BedrockModel fields |
| `region_name` | `us-west-2` | AWS region |
| `boto_config` | `None` | BotocoreConfig or dict to merge with defaults |
| `n_sample_traces` | `-1` (all) | Number of traces to sample per step |
| `stratified_sampling` | `True` | Balance successful/failed traces. Requires rewards |
| `success_threshold` | `0.5` | Reward threshold for classifying success |
| `max_output_chars` | `100000` | Max tool output before truncation |
| `system_prompt_suffix` | Default | Instructions for using submit_optimized_params tool |

## Templates

Templates are Jinja2 files that control what the reflection agent sees. They receive these variables:

| Variable | Type | Description |
|----------|------|-------------|
| `traces_folder` | `str` | Absolute path to the temp folder containing trace JSON files |
| `params` | `dict` | Full dict from `formula.get_tunable_params()` |

### Built-in templates

The library ships with templates for system prompt optimization:

- `contrastive_reflection/system_prompt.jinja` — general contrastive analysis instructions for the reflection agent
- `contrastive_reflection/task_message_system_prompt.jinja` — task message tailored for system prompt optimization (preserve original, append insights)

Load them with `load_builtin_template()`:

```python
from harness_optimizer.utils import load_builtin_template

system_tmpl = load_builtin_template("contrastive_reflection/system_prompt.jinja")
task_tmpl = load_builtin_template("contrastive_reflection/task_message_system_prompt.jinja")
```

List all available templates:

```python
from harness_optimizer.utils import list_builtin_templates
print(list_builtin_templates())
# ['contrastive_reflection/system_prompt.jinja',
#  'contrastive_reflection/task_message_system_prompt.jinja']
```

### Custom templates

For non-system-prompt use cases, provide your own task message template. The system prompt template is general and can be reused:

```python
optimizer = ContrastiveReflectionOptimizer(
    formula,
    system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
    task_message_template="Analyze traces in {{ traces_folder }}.\n\nCurrent params: {{ params }}\n\nOptimize them.",
)
```

## Checkpointing

`ContrastiveReflectionOptimizer` inherits full checkpointing from `BaseAgenticOptimizer`. The state includes all config, step count, prompt history, and sampled trace indices:

```python
# Save
state = optimizer.get_state()

# Restore
new_optimizer = ContrastiveReflectionOptimizer(formula, ...)
new_optimizer.load_state(state)
```

## Full example

See [`examples/gsm8k_optimization.py`](../../../examples/gsm8k_optimization.py) for a complete optimization loop using GSM8K math problems with a Strands agent and shell tool.
