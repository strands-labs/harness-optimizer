"""Integration tests for strands adapter with real Bedrock model invocation.

Requires AWS credentials; skipped otherwise.
"""

import os
import pytest
from dotenv import load_dotenv
from strands import Agent

from harness_optimizer.formulas import SystemPromptFormula
from harness_optimizer.adapters import StrandsAdapter

load_dotenv()

has_aws_credentials = bool(os.environ.get("AWS_ACCESS_KEY_ID"))


@pytest.mark.integration
@pytest.mark.skipif(not has_aws_credentials, reason="AWS credentials not available")
class TestEndToEndWithModel:

    @pytest.fixture
    def model(self):
        from strands.models import BedrockModel
        return BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name="us-west-2",
        )

    def test_hook_fires_on_invocation(self, model):
        """Formula hook updates system prompt when agent is invoked."""
        formula = SystemPromptFormula(system_prompt="Reply with only: HOOK_FIRED")
        agent = Agent(model=model, system_prompt="Original")
        StrandsAdapter().apply_to_agent([formula], agent)

        assert agent.system_prompt == "Original"
        agent("Say hello")
        assert agent.system_prompt == "Reply with only: HOOK_FIRED"

    def test_update_params_between_invocations(self, model):
        """Optimizer updates formula params between agent invocations."""
        formula = SystemPromptFormula(system_prompt="v1")
        agent = Agent(model=model, system_prompt="Original")
        StrandsAdapter().apply_to_agent([formula], agent)

        agent("Say hello")
        assert agent.system_prompt == "v1"

        formula.update_params({"system_prompt": "v2"})
        agent("Say hello again")
        assert agent.system_prompt == "v2"
