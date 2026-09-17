# Formulas

A Formula is the core optimizable unit in Harness Optimizer. It processes agent context and exposes tunable parameters that can be optimized through rollout-based feedback.

## Overview

Formulas define:
- **What context to process** — via `process(context) -> dict`
- **What parameters are tunable** — via `get_tunable_params()` and `update_params()`
- **When to run** — via `trigger_timings` (strands event types or strings)

## Using SystemPromptFormula

The built-in `SystemPromptFormula` manages a system prompt string as its tunable parameter.

```python
from strands_harness_optimizer.formulas import SystemPromptFormula

# Create a formula
formula = SystemPromptFormula(system_prompt="You are a helpful coding assistant.")

# Get tunable parameters
params = formula.get_tunable_params()
# {'system_prompt': 'You are a helpful coding assistant.'}

# Update parameters (e.g., after optimization)
formula.update_params({"system_prompt": "You are an expert Python developer."})
```

`SystemPromptFormula` triggers on `BeforeInvocationEvent` — it updates the agent's system prompt before each invocation.

## Using ToolDescriptionFormula

`ToolDescriptionFormula` tunes the descriptions of the agent's tools — the text the model reads when deciding whether to call a tool, how to fill its inputs, and how to read its result. It holds two maps: a full **base** (one entry per tool, never mutated) and a sparse set of **overrides** (only the tools an optimizer has edited). The overrides are the tunable parameter and the only thing delivered to the agent; every other tool keeps its own description.

```python
from strands_harness_optimizer.formulas import ToolDescriptionFormula

formula = ToolDescriptionFormula(
    base={"search": "Search the catalog.", "click": "Click an element on the page."}
)
formula.get_tunable_params()
# {'tool_descriptions': {}}

# An optimizer submits only the tools it edited; unedited overrides are carried forward.
formula.update_params({"tool_descriptions": {"search": "Search the catalog by keyword."}})
formula.get_tunable_params()
# {'tool_descriptions': {'search': 'Search the catalog by keyword.'}}

formula.effective()          # base with the overrides applied — what the agent sees
formula.render_effective_yaml()
```

A name not in `base` or a description over 800 characters is dropped with a warning (or raises with `strict=True`). An empty string keeps the current text. `ToolDescriptionFormula.from_yaml(base_path, overrides_path)` reads `tool_descriptions: {name: text}` files.

For in-process agents the strands adapter patches each named tool's spec in place — `mcp_tool.description` for MCP tools, the `tool_spec` dict for `@tool` functions — and checks at attach time that the agent exposes that seam. Remote agents receive the overrides as a `tool_descriptions` payload key and patch on their side.

## Using MultiSurfaceFormula

`MultiSurfaceFormula` is one formula over the three surfaces an agent reads: the system prompt, the skill library and the tool descriptions. It wraps a `SystemPromptFormula`, a `SkillLibraryFormula` and a `ToolDescriptionFormula`, merges their tunable parameters, and routes `update_params` to whichever members the params name. A surface with no key in the update is left untouched.

```python
from strands_harness_optimizer.formulas import (
    MultiSurfaceFormula, SkillLibraryFormula, SystemPromptFormula, ToolDescriptionFormula,
)

formula = MultiSurfaceFormula(
    system_prompt=SystemPromptFormula(system_prompt="You are a shopping agent."),
    skills=SkillLibraryFormula(skill_dir="./skills"),          # None for a cold start
    tool_descriptions=ToolDescriptionFormula(base={"search": "...", "click": "..."}),
)
formula.get_tunable_params()
# {'system_prompt': '...', 'skill_dir': './skills', 'tool_descriptions': {}}

# An optimizer that edited the prompt, wrote skill decisions, and edited one tool:
formula.update_params({
    "system_prompt": "...",
    "decisions_dir": "./runs/step_0001/skills",
    "tool_descriptions": {"search": "..."},
})
formula.materialize("./runs/step_0001/skill_set")   # delegates to the skill member
```

The members stay reachable as `formula.system_prompt`, `formula.skills` and `formula.tool_descriptions`. Attached to an in-process agent, `process` runs the members in order and returns only the keys that changed, so the adapter writes back exactly what moved.

## Creating a Custom Formula

Subclass `Formula` and implement the abstract methods:

```python
from strands_harness_optimizer.formulas import Formula
from strands.hooks.events import BeforeInvocationEvent

class PrefixFormula(Formula):
    """Formula that prepends a tunable preamble to the system prompt."""

    def __init__(self, prefix: str):
        super().__init__("prefix_formula", [BeforeInvocationEvent])
        self.prefix = prefix

    def process(self, context: dict, **kwargs) -> dict:
        current_prompt = context.get("system_prompt", "")
        return {"system_prompt": f"{self.prefix}\n\n{current_prompt}"}

    def get_tunable_params(self) -> dict:
        return {"prefix": self.prefix}

    def update_params(self, params: dict) -> None:
        if "prefix" in params:
            self.prefix = params["prefix"]
```

### Required methods

| Method | Description |
|--------|-------------|
| `__init__(name, trigger_timings)` | Set formula name and when it runs |
| `process(context, **kwargs) -> dict` | Process agent context, return updated context |
| `get_tunable_params() -> dict` | Return current tunable parameters |
| `update_params(params) -> None` | Update parameters from a dict |

### Optional methods

| Method | Default | Description |
|--------|---------|-------------|
| `can_process(context) -> bool` | `True` | Override to conditionally skip processing |

## Trigger Timings

`trigger_timings` defines when the formula runs during the agent lifecycle. Since Strands Agents is the native platform we aim to support, we support trigger timing definitions with Strands Agents' event classes directly. String-based timings are also supported for framework-agnostic usage:

```python
from strands.hooks.events import BeforeInvocationEvent, AfterInvocationEvent

# Using strands event types (recommended for strands agents)
Formula("my_formula", [BeforeInvocationEvent])
Formula("my_formula", [BeforeInvocationEvent, AfterInvocationEvent])

# Using strings
Formula("my_formula", ["before_invocation"])
Formula("my_formula", ["before_invocation", "after_invocation"])
```

Supported string values for the strands adapter:

| String | Strands Event Type |
|--------|-------------------|
| `"agent_initialized"` | `AgentInitializedEvent` |
| `"before_invocation"` | `BeforeInvocationEvent` |
| `"after_invocation"` | `AfterInvocationEvent` |
| `"before_model_call"` | `BeforeModelCallEvent` |
| `"after_model_call"` | `AfterModelCallEvent` |
| `"before_tool_call"` | `BeforeToolCallEvent` |
| `"after_tool_call"` | `AfterToolCallEvent` |
| `"message_added"` | `MessageAddedEvent` |

## Supported Context

Currently supported:
- **System prompt** — the agent's system prompt string (`system_prompt`)
- **Skills** — the set of skills in the agent's `AgentSkills` plugin (`skills`; see `SkillFormula` and `SkillLibraryFormula`)
- **Tool descriptions** — `{tool_name: description}` overrides patched into the agent's tool specs (`tool_descriptions`; see `ToolDescriptionFormula`)

Future support planned:
- **Tools** — tool definitions and configurations beyond the description text
- **MCP servers** — Model Context Protocol server configurations

## What's Next

- [Adapters](adapters.md) — attach Formulas to agent frameworks
