<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>Harness Optimizer</h1>

  <h2>Evolving LLM agent harness in just a few lines of code.</h2>

  <div align="center">
    <a href="https://github.com/strands-labs/harness-optimizer/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/strands-labs/harness-optimizer" alt="Commit Activity"></a>
    <a href="https://github.com/strands-labs/harness-optimizer/issues"><img src="https://img.shields.io/github/issues/strands-labs/harness-optimizer" alt="Open Issues"></a>
    <a href="https://github.com/strands-labs/harness-optimizer/pulls"><img src="https://img.shields.io/github/issues-pr/strands-labs/harness-optimizer" alt="Open PRs"></a>
    <a href="https://github.com/strands-labs/harness-optimizer/blob/main/LICENSE"><img src="https://img.shields.io/github/license/strands-labs/harness-optimizer" alt="License"></a>
  </div>

  <p>
    Built on <a href="https://github.com/strands-agents/sdk-python">Strands Agents</a>
  </p>
</div>

A framework for optimizing LLM agent harness through Formulas.

## Overview

Harness Optimizer provides a framework for defining, attaching, and optimizing context units (e.g., system prompts) for LLM agents. The core idea: optimize the LLM agent harness by using tunable Formulas to dynamically enhance the agent, and improving those Formulas with optimizers based on collected agent rollout trajectories.

> **Naming**: `ContextUnitProcessor` (CUP) has been renamed to `Formula`. Legacy names are available via `harness_optimizer.compat` with deprecation warnings.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Trainer.fit()                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌─────────────────────┐   Adapter      ┌───────────────────────────┐ │
│    │ Formula (CUP)       ├───────────────▶│        LLM Agent          │ │
│    │ get_tunable_params()│ (attach to     │    (e.g. Strands Agent)   │ │
│    │ update_params()     │  agent)        └───────────▲┬──────────────┘ │
│    └──▲──────────────────┘               invoke agent ││                │
│       │                                               ││ collect rollout│
│       │  ┌─────────────┐  ┌───────────────────────────┴▼──────────────┐ │
│       │  │ DataLoader  │─▶│ AgentRolloutEngine                        │ │
│       │  │  Out: [data]│  │ In: [data], cup_params                    │ │
│       │  └─────────────┘  │ Out: [(rollout, data)...]                 │ │
│       │                   └──────────┬────────────────────────────────┘ │
│       │                              ▼                                  │
│       │               ┌───────────────────────────────────┐             │
│       │               │ Rollouts: [(rollout, data) ...]   │             │
│       │               └───┬───────────────────────┬───────┘             │
│       │                   ▼                       ▼                     │
│       │  ┌────────────────────┐  ┌───────────────────────────────────┐  │
│       │  │ RewardFunction     │  │ FormulaOptimizer                  │  │
│       │  │ In: rollout, data  │─▶│ In: params, [(rollout,data,rwrd)] │  │
│       │  │ Out: reward        │  │ Out: new_params                   │  │
│       │  └────────────────────┘  └───────────────┬───────────────────┘  │
│       │                                          │                      │
│       └──────────────────────────────────────────┘                      │
│                            new_params                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. **DataLoader** yields batches of task samples from the Dataset
2. **AgentRolloutEngine** executes the agent on each sample using current Formula parameters, producing rollouts
   - **Adapter** bridges Formula parameters to the agent framework (e.g., `apply_formulas_on_strands_agent`)
   - **LLM Agent** runs the task and produces a rollout (conversation trace)
3. **RewardFunction** scores each rollout
4. Rollouts, data, and rewards are collected into a batch
5. **FormulaOptimizer** analyzes the batch to propose new Formula parameters
6. **Formula** updates its parameters and the loop repeats

## Installation

```bash
pip install harness-optimizer
```

## Quick Example

```python
from strands import Agent
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import apply_formulas_on_strands_agent

# Create a Formula
formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")

# Attach to a strands agent
agent = Agent(model=model)
apply_formulas_on_strands_agent(agent, [formula])

# Get / update parameters (e.g., after optimization)
params = formula.get_tunable_params()
# {'system_prompt': 'You are a helpful assistant.'}
formula.update_params({'system_prompt': 'You are an expert coding assistant.'})
```

## Core Interfaces

### Formula (formulas/)

The optimizable unit, formerly known as ContextUnitProcessor (CUP).

```python
class Formula(ABC):
    def process(self, context: dict, **kwargs) -> dict: ...
    def get_tunable_params(self) -> dict: ...
    def update_params(self, params: dict) -> None: ...
    def can_process(self, context: dict) -> bool: ...
```

### Strands Adapter (adapters/)

Attaches Formulas to strands agents as hook callbacks.

```python
apply_formulas_on_strands_agent(agent, [formula1, formula2])
```

### RewardFunction (rewards/)

Scores agent rollouts. Returns a dict with `reward_value` and optional metadata.

```python
class RewardFunction(ABC):
    def __call__(self, **kwargs) -> dict: ...
```

### FormulaOptimizer (optimizers/)

Optimizes Formula parameters based on accumulated rollouts and rewards.
Follows PyTorch's pattern: `add_rollouts()`, `add_rewards()`, `step()`, `zero()`.

```python
class FormulaOptimizer(ABC):
    def add_rollouts(self, rollouts: list[dict]) -> None: ...
    def add_rewards(self, rewards: list[dict]) -> None: ...
    def step(self) -> None: ...
    def zero(self) -> None: ...
    def get_state(self) -> dict: ...
    def load_state(self, state: dict) -> None: ...
```

### AgentRolloutEngine (rollout_engines/)

Generates rollouts by executing agents on data samples.

```python
class AgentRolloutEngine(ABC):
    def generate_batch(self, data_samples: list[dict]) -> Iterator[list[dict]]: ...
```

## Package Structure

```
harness_optimizer/
├── __init__.py
├── compat.py                       # Legacy names (ContextUnitProcessor, etc.)
├── trainer.py                      # Minimal training loop
├── data/                           # PyTorch-style data loading (stdlib adapted)
│   ├── dataset.py                  # Dataset, IterableDataset, ConcatDataset, Subset
│   ├── sampler.py                  # Sampler, SequentialSampler, RandomSampler, BatchSampler
│   └── dataloader.py              # Simplified DataLoader
├── formulas/                       # Formula framework
│   ├── formula.py                  # Formula ABC
│   ├── system_prompt_formula.py   # Built-in: SystemPromptFormula
│   └── context_expansion_formula.py # Built-in: ContextExpansionFormula
├── optimizers/                     # Optimization framework
│   ├── optimizer.py                # FormulaOptimizer ABC
│   └── system_prompt/              # Built-in optimizers
│       ├── base_agentic_optimizer.py   # BaseAgenticOptimizer
│       └── contrastive_reflection.py   # ContrastiveReflectionOptimizer
├── rewards/                        # Reward computation
│   └── reward_function.py         # RewardFunction ABC
├── rollout_engines/                # Agent rollout generation
│   ├── agent_rollout_engine.py    # AgentRolloutEngine ABC
│   ├── parallel_engine.py         # ParallelAgentRolloutEngine (utilities)
│   ├── local_engine.py            # LocalRolloutEngine
│   └── agentcore_engine.py        # AgentCoreRolloutEngine
├── templates/                      # Jinja2 templates for optimizers
│   └── contrastive_reflection/
│       ├── system_prompt.jinja
│       └── task_message_system_prompt.jinja
├── adapters/                       # Agent framework adapters
│   ├── agent_adapter.py           # AgentAdapter ABC
│   └── strands_adapter.py         # StrandsAdapter, StrandsAgentWithFormulas
└── utils/                          # Utilities
    ├── templates.py               # load_builtin_template, list_builtin_templates
    ├── params_store.py            # FormulaParamsStore, S3FormulaParamsStore
    └── guardrails/
        └── tool_output.py         # ToolOutputGuardrail
```

## Design Decisions

1. **Dict-based data**: No wrapper classes for context or evaluation results — plain dicts throughout for simplicity and flexibility.
2. **Formula = CUP**: `ContextUnitProcessor` renamed to `Formula`. Legacy names available via `harness_optimizer.compat`.
3. **Minimal dependencies**: Core depends on `strands-agents`, `strands-agents-tools`, `jinja2`, and `botocore` for the built-in adapter and optimizer.
4. **PyTorch Dataset/DataLoader reuse**: Copied from PyTorch source with `torch` replaced by stdlib `random`. No PyTorch dependency.
5. **Minimal Trainer**: Users can easily write their own training loop. The built-in Trainer is just two nested for-loops.

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
