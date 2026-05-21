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
from harness_optimizer.optimizers import FormulaOptimizer

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

`BaseAgenticOptimizer` is the base class for optimizers that use a strands Agent with tools to analyze rollouts. It provides infrastructure that specific agentic optimizers build on:

- In-memory trace sampling (random or stratified by reward)
- Writing sampled traces to temp folders as JSON
- Agent creation with configurable `model_config` and boto settings
- `ToolOutputGuardrail` for truncating excessive tool output
- `submit_optimized_params` tool for reliable parameter extraction (file-per-param)
- Full `get_state()`/`load_state()` checkpointing (config + step_count + prompt_history)

Subclass it to build custom agentic optimizers:

```python
from harness_optimizer.optimizers import BaseAgenticOptimizer

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
from harness_optimizer.utils.guardrails import ToolOutputGuardrail

guardrail = ToolOutputGuardrail(max_chars=50000)
guardrail.register(agent)
```

## Built-in Optimizers

- [ContrastiveReflectionOptimizer](./optimizers/contrastive-reflection.md) — contrastive learning on rollout traces

## What's Next

- [Training](training.md) — automated training loop with Dataset, DataLoader, and Trainer
