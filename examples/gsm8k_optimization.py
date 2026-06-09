"""Example: Optimize a math agent's system prompt using GSM8K.

Demonstrates the full optimization loop:
1. Load GSM8K math word problems via HuggingFace datasets
2. Use our DataLoader to batch the HF dataset directly (no wrapper needed)
3. Create a Strands agent with shell tool (python as calculator)
4. Collect rollouts — agent solves problems, reward checks final answer
5. ContrastiveReflectionOptimizer analyzes success/failure patterns
6. System prompt is updated with learned insights

Requirements:
    pip install strands-harness-optimizer datasets
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

from strands_harness_optimizer.data import DataLoader
from strands_harness_optimizer.formulas import SystemPromptFormula
from strands_harness_optimizer.adapters import apply_formulas_on_strands_agent
from strands_harness_optimizer.rewards import RewardFunction
from strands_harness_optimizer.optimizers import ContrastiveReflectionOptimizer
from strands_harness_optimizer.utils import load_builtin_template


# --- Reward function: extract final number and compare ---

class GSM8KReward(RewardFunction):
    """Reward 1.0 if the agent's final number matches the expected answer."""

    def __call__(self, **kwargs):
        response_text = kwargs.get("response_text", "")
        expected = kwargs.get("expected_answer", "")

        predicted = self._extract_final_number(response_text)
        correct = predicted is not None and predicted == expected

        return {
            "reward_value": 1.0 if correct else 0.0,
            "predicted": predicted,
            "expected": expected,
        }

    def _extract_final_number(self, text: str) -> str | None:
        """Extract the last number from the response."""
        numbers = re.findall(r'-?\d[\d,]*\.?\d*', text.replace(",", ""))
        return numbers[-1] if numbers else None


# --- Helpers ---

def parse_gsm8k_answer(answer_text: str) -> str:
    """Extract the final answer number from GSM8K format (#### <number>)."""
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def prepare_gsm8k(split: str = "test[:20]"):
    """Load GSM8K and extract question + expected_answer."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {
            "question": row["question"],
            "expected_answer": parse_gsm8k_answer(row["answer"]),
        }
        for row in ds
    ]


def main():
    # --- Load GSM8K ---
    print("Loading GSM8K dataset...")
    train_data = prepare_gsm8k("test[:15]")
    test_data = prepare_gsm8k("test[15:20]")

    # HF datasets work directly with our DataLoader — no wrapper needed
    train_loader = DataLoader(train_data, batch_size=15)

    print(f"Train: {len(train_data)} problems, Test: {len(test_data)} problems")

    # --- Setup agent ---
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

    agent = Agent(model=model, tools=[shell])
    apply_formulas_on_strands_agent(agent, [formula])

    reward_fn = GSM8KReward()
    optimizer = ContrastiveReflectionOptimizer(
        formula,
        system_prompt_template=load_builtin_template("contrastive_reflection/system_prompt.jinja"),
        task_message_template=load_builtin_template("contrastive_reflection/task_message_system_prompt.jinja"),
        model_config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        n_sample_traces=-1,
    )

    # --- Optimization loop ---
    n_rounds = 3

    for round_idx in range(n_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1}/{n_rounds}")
        print(f"Current prompt: {formula.get_tunable_params()['system_prompt'][:100]}...")
        print(f"{'='*60}")

        # Collect rollouts using DataLoader batches
        rollouts = []
        rewards = []
        sample_idx = 0
        for batch in train_loader:
            for sample in batch:
                agent.messages.clear()
                try:
                    response = agent(sample["question"])
                    response_text = response.message["content"][0]["text"]
                except Exception as e:
                    response_text = f"Error: {e}"

                rollout = {
                    "messages": list(agent.messages),
                    "response_text": response_text,
                    "data_sample": sample,
                }
                reward = reward_fn(
                    response_text=response_text,
                    expected_answer=sample["expected_answer"],
                )
                rollouts.append(rollout)
                rewards.append(reward)

                sample_idx += 1
                status = "correct" if reward["reward_value"] == 1.0 else "wrong"
                print(f"  [{sample_idx}/{len(train_data)}] {status} "
                      f"(predicted={reward['predicted']}, expected={reward['expected']})")

        train_acc = sum(r["reward_value"] for r in rewards) / len(rewards)
        print(f"\nTrain accuracy: {train_acc:.0%}")

        # Optimize (skip on last round)
        if round_idx < n_rounds - 1:
            optimizer.add_rollouts(rollouts)
            optimizer.add_rewards(rewards)
            print("\nOptimizing system prompt...")
            optimizer.step()
            optimizer.zero()
            print(f"New prompt: {formula.get_tunable_params()['system_prompt'][:100]}...")

    # --- Test evaluation ---
    print(f"\n{'='*60}")
    print("Test Evaluation (with optimized prompt)")
    print(f"{'='*60}")

    test_rewards = []
    for i, sample in enumerate(test_data):
        agent.messages.clear()
        try:
            response = agent(sample["question"])
            response_text = response.message["content"][0]["text"]
        except Exception as e:
            response_text = f"Error: {e}"

        reward = reward_fn(
            response_text=response_text,
            expected_answer=sample["expected_answer"],
        )
        test_rewards.append(reward["reward_value"])

        status = "correct" if reward["reward_value"] == 1.0 else "wrong"
        print(f"  [{i+1}/{len(test_data)}] {status} "
              f"(predicted={reward['predicted']}, expected={reward['expected']})")

    test_acc = sum(test_rewards) / len(test_rewards)
    print(f"\nTest accuracy: {test_acc:.0%}")
    print(f"\nFinal optimized prompt:\n{formula.get_tunable_params()['system_prompt']}")


if __name__ == "__main__":
    main()
