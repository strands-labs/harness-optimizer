"""Example: Attach a ContextExpansionFormula to a strands agent.

This example demonstrates:
1. Creating a ContextExpansionFormula to add instructions to the first user message
2. Attaching it to a strands Agent via StrandsAdapter
3. Invoking the agent (the formula modifies the first user message)
4. Updating formula parameters (simulating what an optimizer would do)

Requirements:
    pip install strands-harness-optimizer
    AWS credentials configured for Bedrock access
"""

from strands import Agent
from strands.models import BedrockModel

from strands_harness_optimizer.formulas import ContextExpansionFormula
from strands_harness_optimizer.adapters import StrandsAdapter


def main():
    # Create a model
    model = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
    )

    # Example 1: Add instruction to first user message
    print("=== First invocation ===")
    agent = Agent(model=model, system_prompt="You are a helpful assistant.", callback_handler=None)

    # Create a formula that adds an instruction
    formula = ContextExpansionFormula(instruction="Answer it in French.")
    adapter = StrandsAdapter()
    adapter.apply_to_agent([formula], agent)

    user_message = "Tell me a story about AI."
    print(f"User: {user_message}")
    response = agent(user_message)

    # response should be in French now
    print(f"Assistant: {response}")
    print()

    # Example 2: Update instruction parameter
    print("=== Second invocation with updated instruction ===")
    print(f"User: {user_message}")
    formula.update_params({"instruction": "Answer it in Spanish and show your work."})
    agent = Agent(model=model, system_prompt="You are a helpful assistant.", callback_handler=None)
    adapter.apply_to_agent([formula], agent)

    # response should be in Spanish
    response = agent(user_message)
    print(f"Assistant: {response}")
    print()


if __name__ == "__main__":
    main()
