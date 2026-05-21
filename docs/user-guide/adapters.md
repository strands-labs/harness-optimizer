# Adapters

An adapter bridges Formulas to a specific agent framework by translating between the framework's native context representation and the unified dict format that Formulas operate on.

## Overview

Adapters are responsible for:
- **Extracting context** from an agent into a unified dict (`extract_context`)
- **Updating context** on an agent from a unified dict (`update_context`)
- **Attaching formulas** to the agent's lifecycle (`apply_to_agent`)

## Using StrandsAdapter

The built-in `StrandsAdapter` connects Formulas to [strands-agents](https://github.com/strands-agents/sdk-python).

### Basic usage

```python
from strands import Agent
from strands.models import BedrockModel
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import StrandsAdapter

model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
agent = Agent(model=model, system_prompt="You are a helpful assistant.")

formula = SystemPromptFormula(system_prompt="You are an expert Python developer.")
adapter = StrandsAdapter()
adapter.apply_to_agent([formula], agent)

# When agent is invoked, the formula's process() runs before each invocation,
# updating the system prompt
response = agent("Write a hello world program")
```

### Convenience function

```python
from harness_optimizer.adapters import apply_formulas_on_strands_agent

agent = apply_formulas_on_strands_agent(agent, [formula])
```

### Agent subclass

`StrandsAgentWithFormulas` accepts formulas directly in the constructor:

```python
from harness_optimizer.adapters import StrandsAgentWithFormulas

agent = StrandsAgentWithFormulas(
    model=model,
    system_prompt="You are a helpful assistant.",
    formulas=[formula],
)
```

## Creating a Custom Adapter

Subclass `AgentAdapter` to support a different agent framework:

```python
from harness_optimizer.adapters import AgentAdapter
from harness_optimizer.formulas import Formula

class MyFrameworkAdapter(AgentAdapter):

    def extract_context(self, agent) -> dict:
        return {
            "system_prompt": agent.config["prompt"],
            "messages": agent.config.get("history", []),
            # Future: extract tools, skills, MCP server configs
        }

    def update_context(self, agent, context: dict) -> None:
        if "system_prompt" in context:
            agent.config["prompt"] = context["system_prompt"]

    def apply_to_agent(self, formulas: list[Formula], agent):
        for formula in formulas:
            # Framework-specific hook registration
            agent.register_middleware(
                lambda ctx: formula.process(self.extract_context(ctx))
            )
        return agent
```

### Required methods

| Method | Description |
|--------|-------------|
| `extract_context(agent) -> dict` | Convert agent's native context to unified dict |
| `update_context(agent, context)` | Apply unified dict back to agent |
| `apply_to_agent(formulas, agent)` | Attach formulas and return the agent |

## Context Dict Format

The unified context dict uses standardized keys:

```python
{
    "system_prompt": str,   # The agent's system prompt
    "messages": list[dict], # Conversation history
    # Future support planned:
    # "tools": list[dict],        # Tool definitions and configurations
    # "skills": list[dict],       # Modular instruction sets
    # "mcp_servers": list[dict],  # MCP server configurations
}
```

Formulas read from and write to this dict. The adapter handles translation to/from the framework's native representation.

## What's Next

- [Optimizers](optimizers.md) — optimize Formula parameters from rollouts
