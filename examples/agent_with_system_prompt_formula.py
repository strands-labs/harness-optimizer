"""Example: Attach a Formula to a strands agent.

This example demonstrates:
1. Creating a SystemPromptFormula
2. Attaching it to a strands Agent via StrandsAdapter
3. Invoking the agent (the formula updates the system prompt before each invocation)
4. Updating formula parameters (simulating what an optimizer would do)

Requirements:
    pip install harness-optimizer
    AWS credentials configured for Bedrock access
"""

from strands import Agent
from strands.models import BedrockModel

from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import StrandsAdapter


def main():
    # 1. Create a model and agent
    model = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
    )
    agent = Agent(model=model, system_prompt="You are a helpful assistant.")

    # 2. Create a formula and attach it to the agent
    formula = SystemPromptFormula(system_prompt="You are an expert Python developer. Be concise.")
    adapter = StrandsAdapter()
    adapter.apply_to_agent([formula], agent)

    # 3. Invoke the agent — the formula runs before each invocation
    print("=== First invocation ===")
    response = agent("Write a function that checks if a number is prime.")
    print(f"System prompt after invocation: {agent.system_prompt}")
    print()

    # 4. Update formula parameters (simulating optimizer feedback)
    formula.update_params({
        "system_prompt": (
            "You are an expert Python developer. Be concise. "
            "Always include type hints and docstrings."
        )
    })

    print("=== Second invocation (after param update) ===")
    response = agent("Write a function that merges two sorted lists.")
    print(f"System prompt after invocation: {agent.system_prompt}")


if __name__ == "__main__":
    main()
