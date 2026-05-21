# Harness Optimizer Implementation Roadmap

> **Naming**: `ContextUnitProcessor` (CUP) has been renamed to `Formula`. Legacy names are available via `harness_optimizer.compat`. Both terms may appear in docs.

## Stage 1: Formula on LLM Agent (completed)

| Scope | Files |
|-------|-------|
| `Formula` ABC + `SystemPromptFormula` + compat module | `formulas/formula.py`, `formulas/system_prompt_formula.py`, `compat.py` |
| `AgentAdapter` ABC + `StrandsAdapter` + `StrandsAgentWithFormulas` | `adapters/agent_adapter.py`, `adapters/strands_adapter.py` |
| Stage 1 docs + examples | `docs/user-guide/formulas.md`, `docs/user-guide/adapters.md`, `examples/agent_with_system_prompt_formula.py` |

## Stage 2: Optimization (completed)

| Scope | Files |
|-------|-------|
| `RewardFunction` ABC + `FormulaOptimizer` ABC | `rewards/reward_function.py`, `optimizers/optimizer.py` |
| `ContrastiveReflectionOptimizer` + `BaseAgenticOptimizer` + `ToolOutputGuardrail` | `optimizers/system_prompt/`, `templates/`, `utils/`, `examples/gsm8k_optimization.py` |
| Stage 2 docs | `docs/user-guide/rewards.md`, `docs/user-guide/optimizers.md`, `docs/user-guide/optimizers/contrastive-reflection.md`, `README.md` |

## Stage 3: Training Loop (completed)

| Scope | Files |
|-------|-------|
| `Dataset` + `Sampler` + `DataLoader` | `data/dataset.py`, `data/sampler.py`, `data/dataloader.py` |
| `AgentRolloutEngine` + `LocalRolloutEngine` + `AgentCoreRolloutEngine` + `Trainer` | `rollout_engines/`, `trainer.py`, `utils/parallel_rollout.py`, `examples/gsm8k_trainer.py` |
| Stage 3 docs + Astro + Starlight docs site | `docs/user-guide/training.md`, `docs/astro.config.mjs`, `examples/` |

## Future

- Additional built-in Formulas, optimizers, and adapters
- Online learning support (OnlineTrainer + OnlineAgentRolloutEngine)
