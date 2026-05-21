# Rewards

Reward functions score agent rollouts — they tell the optimizer how well the agent performed on each task.

## Design

A reward function scores rollouts based on their content. It receives rollout data as keyword arguments and computes a reward score.

For verifiable rewards (e.g., code execution results, math correctness), the recommended pattern is to run verification during agent invocation and include the result in the rollout data. The reward function then reads this value to compute the score. This follows the same pattern used by VERL and OpenRLHF:
- **Agent invocation**: Produces rollout data including any verification results
- **Reward function**: Reads rollout data, applies scoring logic, returns reward

Note: Some frameworks (e.g., TRL) allow reward functions to access environments directly for verification during reward computation. Our current design focuses on the rollout-data-based approach, which keeps the reward function simple and decoupled from the environment.

## RewardFunction

A `RewardFunction` takes rollout data as keyword arguments and returns a dict with at least a `reward_value` field.

```python
from harness_optimizer.rewards import RewardFunction

class ExactMatchReward(RewardFunction):
    def __call__(self, **kwargs):
        response = kwargs.get("response_text", "").strip()
        expected = kwargs.get("expected_answer", "").strip()
        match = response == expected
        return {
            "reward_value": 1.0 if match else 0.0,
            "reason": "exact match" if match else "mismatch",
        }
```

### Verifiable rewards

When the reward depends on environment verification (e.g., running tests, checking execution output), include the verification result in the rollout:

```python
# During agent invocation — verification happens here
def invoke_agent(agent, sample):
    response = agent(sample["prompt"])
    test_result = run_tests(response)  # environment verification
    return {
        "messages": list(agent.messages),
        "response_text": response.message["content"][0]["text"],
        "test_passed": test_result.passed,  # verification result in rollout
        "data_sample": sample,
    }

# Reward function — reads the verification result, no environment access
class TestPassReward(RewardFunction):
    def __call__(self, **kwargs):
        return {
            "reward_value": 1.0 if kwargs.get("test_passed", False) else 0.0,
        }
```

### Return format

The returned dict must contain `reward_value` (float). Additional fields are optional and can provide context for the optimizer:

```python
{
    "reward_value": 0.8,              # required
    "reason": "partial match",         # optional — why this score
    "metrics": {"precision": 0.9},     # optional — detailed breakdown
    "response_text": "...",            # optional — for logging
}
```

### Usage

Reward functions are called with rollout data as keyword arguments. The rollout dict is unpacked directly:

```python
reward_fn = ExactMatchReward()

# From a rollout dict
rollout = {"response_text": "5", "expected_answer": "5", "messages": [...]}
reward = reward_fn(**rollout)
# {"reward_value": 1.0, "reason": "exact match"}
```

### With the optimizer

Rewards are accumulated alongside rollouts, aligned by index:

```python
rollouts = [...]
rewards = [reward_fn(**r) for r in rollouts]

optimizer.add_rollouts(rollouts)
optimizer.add_rewards(rewards)
optimizer.step()
```

## What's Next

- [Optimizers](./optimizers.md) — how rewards feed into the optimization loop
