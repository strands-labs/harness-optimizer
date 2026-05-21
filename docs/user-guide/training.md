# Training

The Trainer automates the optimization loop: generate rollouts, compute rewards, and optimize Formula parameters across epochs. Developers can also orchestrate the components (Formula, Optimizer, RewardFunction, Engine) directly for more flexibility — see [Custom training loop](#custom-training-loop). For reward functions, see [Rewards](./rewards.md). For optimizers, see [Optimizers](./optimizers.md).

## Dataset and DataLoader

Dataset and DataLoader follow PyTorch conventions (adapted with stdlib, no torch dependency). Also compatible with HuggingFace datasets and plain Python lists via duck typing.

### Creating a Dataset

Subclass `Dataset` and implement `__getitem__` and `__len__`:

```python
from harness_optimizer.data import Dataset

class MathDataset(Dataset):
    def __init__(self, problems):
        self.problems = problems

    def __getitem__(self, index):
        return self.problems[index]

    def __len__(self):
        return len(self.problems)

dataset = MathDataset([
    {"prompt": "What is 2+3?", "expected_answer": "5"},
    {"prompt": "What is 10-4?", "expected_answer": "6"},
])
```

Or use a plain list or HuggingFace dataset directly — anything with `__getitem__` and `__len__` works:

```python
# Plain list
dataset = [{"prompt": "What is 2+3?", "expected_answer": "5"}, ...]

# HuggingFace dataset
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="test[:100]")
```

### DataLoader

Wraps a Dataset with batching and shuffling:

```python
from harness_optimizer.data import DataLoader

loader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in loader:
    # batch is a list of 4 data samples
    ...
```

## AgentRolloutEngine

The engine generates rollouts by invoking an agent on data samples.

### LocalRolloutEngine

Framework-agnostic engine for local agents. Uses `agent_create` and `agent_invoke` callables for thread-safe parallel execution:

```python
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import apply_formulas_on_strands_agent
from harness_optimizer.rollout_engines import LocalRolloutEngine

formula = SystemPromptFormula(system_prompt="You are helpful.")

def create_agent():
    agent = Agent(model=model, system_prompt="...")
    apply_formulas_on_strands_agent(agent, [formula])
    return agent

def invoke_agent(agent, sample):
    agent.messages.clear()
    response = agent(sample["prompt"])
    return {
        "messages": list(agent.messages),
        "response_text": response.message["content"][0]["text"],
        "data_sample": sample,
    }

engine = LocalRolloutEngine(
    formula=formula,
    agent_create=create_agent,
    agent_invoke=invoke_agent,
    num_workers=4,
)
rollouts = list(engine.generate_batch([{"prompt": "What is 2+3?"}]))
```

### AgentCoreRolloutEngine

For remote AgentCore runtimes. Params are included in the invocation payload:

```python
from harness_optimizer.rollout_engines import AgentCoreRolloutEngine

engine = AgentCoreRolloutEngine(
    agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/abc",
    formula=formula,
    num_workers=4,
)
```

### Custom engine

Subclass `AgentRolloutEngine` and implement `generate_batch`:

```python
from harness_optimizer.rollout_engines import AgentRolloutEngine

class MyEngine(AgentRolloutEngine):
    def generate_batch(self, data_samples):
        for sample in data_samples:
            yield {"messages": [...], "response_text": my_agent(sample["prompt"])}
```

### Multiple rollouts per sample

Configure `num_rollouts` for GRPO-style training or best-of-N:

```python
engine = LocalRolloutEngine(
    formula=formula,
    agent_create=create_agent,
    agent_invoke=invoke_agent,
    num_rollouts=4,
)
rollouts = list(engine.generate_batch([{"prompt": "Q1"}, {"prompt": "Q2"}]))
# Returns 8 rollouts: 4 for Q1, 4 for Q2
```

### Params synchronization

The engine calls `ensure_sync_params()` before each batch:

- **Local**: No-op (formula hooks sync automatically on agent invocation)
- **AgentCore**: Captures params into `_synced_params`, included in each invocation payload

## Trainer

The Trainer ties everything together in an automated loop:

```python
from harness_optimizer.trainer import Trainer

trainer = Trainer(
    formula=formula,
    optimizer=optimizer,
    reward_fn=reward_fn,
    engine=engine,
    dataloader=dataloader,
    n_epochs=3,
)

stats = trainer.fit()
# [{"epoch": 1, "avg_reward": 0.4}, {"epoch": 2, "avg_reward": 0.8}, ...]
```

### What fit() does

```
for epoch in range(n_epochs):
    for batch in dataloader:
        rollouts = list(engine.generate_batch(batch))
        rewards = [reward_fn(**rollout) for rollout in rollouts]
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)
    optimizer.step()
    optimizer.zero()
```

### Custom training loop

For more control, write your own loop:

```python
for epoch in range(3):
    for batch in dataloader:
        rollouts = list(engine.generate_batch(batch))
        rewards = [reward_fn(**rollout) for rollout in rollouts]
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)

    optimizer.step()
    optimizer.zero()

    print(f"Epoch {epoch}: prompt = {formula.get_tunable_params()['system_prompt'][:50]}")
```

## Full Example

See [`examples/gsm8k_trainer.py`](../../examples/gsm8k_trainer.py) for a complete example using the Trainer with LocalRolloutEngine and DataLoader, and [`examples/gsm8k_optimization.py`](../../examples/gsm8k_optimization.py) for a manual optimization loop.
