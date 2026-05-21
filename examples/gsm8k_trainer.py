"""Example: Automated GSM8K optimization using Trainer + LocalRolloutEngine.

Demonstrates the automated training loop (compare with gsm8k_optimization.py
which uses the manual loop with direct optimizer calls).

The Trainer handles:
- Iterating over DataLoader batches
- Generating rollouts via the engine
- Computing rewards
- Feeding rollouts/rewards to the optimizer
- Calling optimizer.step() and zero() per epoch

Requirements:
    pip install harness-optimizer datasets
    AWS credentials configured for Bedrock access
"""

import re

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "This example requires the 'datasets' package. "
        "Install it with: pip install datasets"
    )

from strands import Agent
from strands.models import BedrockModel
from strands_tools import shell

from harness_optimizer.data import DataLoader
from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import apply_formulas_on_strands_agent
from harness_optimizer.rewards import RewardFunction
from harness_optimizer.optimizers import ContrastiveReflectionOptimizer
from harness_optimizer.rollout_engines import LocalRolloutEngine
from harness_optimizer.trainer import Trainer
from harness_optimizer.utils import load_builtin_template


# --- Reward function ---

class GSM8KReward(RewardFunction):
    def __call__(self, **kwargs):
        response_text = kwargs.get("response_text", "")
        expected = kwargs.get("expected_answer", "")
        numbers = re.findall(r'-?\d[\d,]*\.?\d*', response_text.replace(",", ""))
        predicted = numbers[-1] if numbers else None
        return {
            "reward_value": 1.0 if predicted == expected else 0.0,
            "predicted": predicted,
            "expected": expected,
        }


# --- Helpers ---

def parse_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', answer_text)
    return match.group(1).replace(",", "") if match else ""


def prepare_gsm8k(split: str):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {"question": row["question"], "expected_answer": parse_gsm8k_answer(row["answer"])}
        for row in ds
    ]


def main():
    print("Loading GSM8K dataset...")
    train_data = prepare_gsm8k("test[:10]")
    test_data = prepare_gsm8k("test[10:15]")
    train_loader = DataLoader(train_data, batch_size=10)

    print(f"Train: {len(train_data)} problems, Test: {len(test_data)} problems")

    # --- Setup ---
    model = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
    )

    formula = SystemPromptFormula(
        system_prompt=(
            "You are a math problem solver. Solve the problem step by step. "
            "Give your final answer as a single number on the last line."
        )
    )

    def create_agent():
        agent = Agent(model=model, tools=[shell])
        apply_formulas_on_strands_agent(agent, [formula])
        return agent

    def invoke_agent(agent, data_sample):
        agent.messages.clear()
        response = agent(data_sample["question"])
        return {
            "messages": list(agent.messages),
            "response_text": response.message["content"][0]["text"],
            "expected_answer": data_sample["expected_answer"],
            "data_sample": data_sample,
        }

    engine = LocalRolloutEngine(
        formula=formula,
        agent_create=create_agent,
        agent_invoke=invoke_agent,
    )

    optimizer = ContrastiveReflectionOptimizer(
        formula,
        system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
        task_message_template=load_builtin_template("contrastive_reflection/task_message_system_prompt.jinja"),
        model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        n_sample_traces=-1,
    )

    reward_fn = GSM8KReward()

    # --- Train with Trainer ---
    trainer = Trainer(
        formula=formula,
        optimizer=optimizer,
        reward_fn=reward_fn,
        engine=engine,
        dataloader=train_loader,
        n_epochs=2,
    )

    print("\n=== Training with Trainer ===")
    stats = trainer.fit()
    for s in stats:
        print(f"  Epoch {s['epoch']}: avg_reward = {s['avg_reward']:.2f}")

    # --- Test evaluation ---
    print("\n=== Test Evaluation ===")
    test_agent = create_agent()
    test_rewards = []
    for i, sample in enumerate(test_data):
        test_agent.messages.clear()
        try:
            response = test_agent(sample["question"])
            response_text = response.message["content"][0]["text"]
        except Exception as e:
            response_text = f"Error: {e}"

        reward = reward_fn(response_text=response_text, expected_answer=sample["expected_answer"])
        test_rewards.append(reward["reward_value"])
        status = "correct" if reward["reward_value"] == 1.0 else "wrong"
        print(f"  [{i+1}/{len(test_data)}] {status} "
              f"(predicted={reward['predicted']}, expected={reward['expected']})")

    print(f"\nTest accuracy: {sum(test_rewards) / len(test_rewards):.0%}")
    print(f"\nFinal prompt:\n{formula.get_tunable_params()['system_prompt'][:200]}...")


if __name__ == "__main__":
    main()
