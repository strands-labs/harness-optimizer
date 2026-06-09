"""Example: Using the Trainer for automated optimization.

Demonstrates the full end-to-end flow using the Trainer class:
1. Define a Formula and attach to a strands Agent
2. Create a Dataset and DataLoader
3. Set up RewardFunction, Optimizer, and Engine
4. Run trainer.fit() for automated multi-epoch optimization
5. Evaluate on a held-out test set

Requirements:
    pip install strands-harness-optimizer
    AWS credentials configured for Bedrock access
"""

from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel

from strands_harness_optimizer.data import Dataset, DataLoader
from strands_harness_optimizer.datamodels import Reward, Rollout
from strands_harness_optimizer.formulas import SystemPromptFormula
from strands_harness_optimizer.adapters import apply_formulas_on_strands_agent
from strands_harness_optimizer.rewards import RewardFunction
from strands_harness_optimizer.optimizers import FormulaOptimizer
from strands_harness_optimizer.rollout_engines import LocalRolloutEngine
from strands_harness_optimizer.trainer import Trainer

load_dotenv()


# --- Custom components ---

class MathDataset(Dataset):
    def __init__(self, problems):
        self.problems = problems

    def __getitem__(self, index):
        return self.problems[index]

    def __len__(self):
        return len(self.problems)


class ExactMatchReward(RewardFunction):
    def __call__(self, rollout: Rollout) -> Reward:
        response = rollout.messages[-1].get("content", [{}])[-1].get("text", "") if rollout.messages else ""
        expected = rollout.data_sample.get("expected_answer", "")
        match = response.strip() == expected.strip()
        return Reward(reward=1.0 if match else 0.0)


class LLMPromptOptimizer(FormulaOptimizer):
    """Uses an LLM to analyze rollouts and generate an improved system prompt."""

    def __init__(self, formula, optimizer_model=None):
        super().__init__(formula)
        self.optimizer_model = optimizer_model or BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            region_name="us-west-2",
        )

    def step(self):
        analysis = []
        for rollout, reward in zip(self._rollouts, self._rewards):
            sample = rollout.data_sample
            response_text = rollout.messages[-1].get("content", [{}])[-1].get("text", "") if rollout.messages else ""
            analysis.append(
                f"Task: {sample.get('prompt', '')}\n"
                f"Expected: {sample.get('expected_answer', '')}\n"
                f"Response: {response_text}\n"
                f"Reward: {reward.reward}"
            )

        current = self.formula.get_tunable_params().get("system_prompt", "")
        prompt = (
            f"You are optimizing a system prompt for an LLM agent.\n\n"
            f"Current system prompt: \"{current}\"\n\n"
            f"Rollouts and rewards:\n\n"
            + "\n\n".join(analysis) +
            f"\n\nAnalyze patterns and generate an improved system prompt.\n"
            f"Reply with ONLY the new system prompt, nothing else."
        )

        response = Agent(model=self.optimizer_model)(prompt)
        new_prompt = response.message["content"][0]["text"].strip()
        self.formula.update_params({"system_prompt": new_prompt})


def main():
    # --- Setup ---
    model = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
    )

    formula = SystemPromptFormula(system_prompt="You are a helpful assistant.")

    def create_agent():
        agent = Agent(model=model, system_prompt="You are a helpful assistant.")
        apply_formulas_on_strands_agent(agent, [formula])
        return agent

    def invoke_agent(agent, sample):
        agent.messages.clear()
        response = agent(sample["prompt"])
        return Rollout(
            data_sample=sample,
            messages=list(agent.messages),
        )

    # --- Data ---
    train_dataset = MathDataset([
        {"prompt": "What is 2+3?", "expected_answer": "5"},
        {"prompt": "What is 10-4?", "expected_answer": "6"},
        {"prompt": "What is 7*8?", "expected_answer": "56"},
        {"prompt": "What is 100/4?", "expected_answer": "25"},
        {"prompt": "What is 15+27?", "expected_answer": "42"},
    ])
    train_loader = DataLoader(train_dataset, batch_size=5)

    test_samples = [
        {"prompt": "What is 11+14?", "expected_answer": "25"},
        {"prompt": "What is 90-35?", "expected_answer": "55"},
        {"prompt": "What is 12*5?", "expected_answer": "60"},
    ]

    # --- Components ---
    reward_fn = ExactMatchReward()
    optimizer = LLMPromptOptimizer(formula)
    engine = LocalRolloutEngine(
        formula=formula,
        agent_creator=create_agent,
        agent_invoker=invoke_agent,
    )

    # --- Train ---
    trainer = Trainer(
        formula=formula,
        optimizer=optimizer,
        reward_fn=reward_fn,
        engine=engine,
        dataloader=train_loader,
        n_epochs=3,
    )

    print("=== Training ===")
    stats = trainer.fit()
    for s in stats:
        print(f"  Epoch {s['epoch']}: avg_reward = {s['avg_reward']:.2f}")

    print(f"\nOptimized prompt: {formula.get_tunable_params()['system_prompt']}")

    # --- Test ---
    print("\n=== Test Evaluation ===")
    test_agent = create_agent()  # fresh agent with optimized formula
    test_rewards = []
    for sample in test_samples:
        test_agent.messages.clear()
        response = test_agent(sample["prompt"])
        response_text = response.message["content"][0]["text"]
        rollout = Rollout(data_sample=sample, messages=list(test_agent.messages))
        reward = reward_fn(rollout)
        test_rewards.append(reward.reward)
        print(f"  Q: {sample['prompt']}  A: [{response_text}]  Reward: {reward.reward}")

    print(f"\n  Test avg reward: {sum(test_rewards) / len(test_rewards):.2f}")
    print(f"  Test accuracy: {sum(1 for r in test_rewards if r == 1.0)}/{len(test_rewards)}")


if __name__ == "__main__":
    main()
