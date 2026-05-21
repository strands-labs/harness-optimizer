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
from harness_optimizer.formulas import SystemPromptFormula

# Create a formula
formula = SystemPromptFormula(system_prompt="You are a helpful coding assistant.")

# Get tunable parameters
params = formula.get_tunable_params()
# {'system_prompt': 'You are a helpful coding assistant.'}

# Update parameters (e.g., after optimization)
formula.update_params({"system_prompt": "You are an expert Python developer."})
```

`SystemPromptFormula` triggers on `BeforeInvocationEvent` — it updates the agent's system prompt before each invocation.

## Creating a Custom Formula

Subclass `Formula` and implement the abstract methods:

```python
from harness_optimizer.formulas import Formula
from strands.hooks.events import BeforeInvocationEvent

class ToolDescriptionFormula(Formula):
    """Formula that injects tool descriptions into the system prompt."""

    def __init__(self, tool_descriptions: dict[str, str]):
        super().__init__("tool_description_formula", [BeforeInvocationEvent])
        self.tool_descriptions = tool_descriptions

    def process(self, context: dict, **kwargs) -> dict:
        current_prompt = context.get("system_prompt", "")
        tool_section = "\n## Available Tools\n"
        for name, desc in self.tool_descriptions.items():
            tool_section += f"- **{name}**: {desc}\n"
        return {"system_prompt": f"{current_prompt}\n{tool_section}"}

    def get_tunable_params(self) -> dict:
        return {"tool_descriptions": self.tool_descriptions.copy()}

    def update_params(self, params: dict) -> None:
        if "tool_descriptions" in params:
            self.tool_descriptions = params["tool_descriptions"]
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
- **System prompt** — the agent's system prompt string

Future support planned:
- **Tools** — tool definitions and configurations
- **Skills** — modular instruction sets
- **MCP servers** — Model Context Protocol server configurations

## What's Next

- [Adapters](adapters.md) — attach Formulas to agent frameworks
