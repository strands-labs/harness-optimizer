---
title: Harness Optimizer
description: A framework for optimizing LLM agent context through Formulas.
---

Harness Optimizer provides a framework for defining, attaching, and optimizing context units (e.g., system prompts, tool definitions, skills, and message history) for LLM agents. The core idea: optimize the LLM agent harness by using tunable Formulas to dynamically enhance the agent, and improving those Formulas with optimizers based on collected agent rollout trajectories.

## Getting Started

1. [Formulas](/user-guide/formulas/) — Define tunable context units
2. [Adapters](/user-guide/adapters/) — Attach Formulas to agent frameworks
3. [Rewards](/user-guide/rewards/) — Score agent rollouts
4. [Optimizers](/user-guide/optimizers/) — Optimize Formula parameters from rollouts
   - [ContrastiveReflectionOptimizer](/user-guide/optimizers/contrastive-reflection/) — Contrastive learning on rollout traces
5. [Training](/user-guide/training/) — Automated training loop with Dataset, DataLoader, and Trainer

## Installation

```bash
pip install harness-optimizer
```

## Quick Example

```python
from strands import Agent
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import apply_formulas_on_strands_agent

formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")
agent = Agent(model=model)
apply_formulas_on_strands_agent(agent, [formula])
```
