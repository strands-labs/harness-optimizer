"""
AgentCore rollout engine — generate rollouts by invoking an AgentCore runtime.

Formula parameters are included in the HTTP payload per invocation.
"""

import json
import logging
import uuid
from typing import Iterator, Optional

import boto3
from botocore.config import Config as BotocoreConfig

from ..datamodels import Rollout
from ..formulas import Formula
from ..utils.parallel_rollout import expand_for_num_rollouts, run_parallel
from .agent_rollout_engine import AgentRolloutEngine

logger = logging.getLogger(__name__)

_DEFAULT_BOTO_CONFIG = {
    "read_timeout": 900,
    "max_pool_connections": 50,
    "retries": {"max_attempts": 1, "mode": "adaptive"},
}


class AgentCoreClient:
    """Client for invoking AgentCore runtimes.

    Handles boto client setup and response parsing (JSON and SSE).
    """

    def __init__(
        self,
        agent_arn: str,
        region_name: str = "us-west-2",
        boto_config: dict | BotocoreConfig | None = None,
    ):
        self.agent_arn = agent_arn

        if isinstance(boto_config, BotocoreConfig):
            config = boto_config
        else:
            merged = {**_DEFAULT_BOTO_CONFIG, **(boto_config or {})}
            config = BotocoreConfig(**merged)

        self.boto_client = boto3.client(
            "bedrock-agentcore",
            region_name=region_name,
            config=config,
        )

    def invoke(
        self,
        payload: dict,
        session_id: Optional[str] = None,
        qualifier: str = "DEFAULT",
    ) -> dict:
        """Invoke the AgentCore runtime with a payload.

        Args:
            payload: Dict to send as JSON payload.
            session_id: Session ID. Auto-generated if None.
            qualifier: Runtime qualifier.

        Returns:
            Parsed response dict with "session_id" included.
        """
        session_id = session_id or str(uuid.uuid4())

        response = self.boto_client.invoke_agent_runtime(
            agentRuntimeArn=self.agent_arn,
            runtimeSessionId=session_id,
            payload=json.dumps(payload),
            qualifier=qualifier,
        )

        response_data = self._parse_response(response)
        response_data["session_id"] = session_id
        return response_data

    def _parse_response(self, response: dict) -> dict:
        """Parse the raw response from AgentCore (JSON or SSE)."""
        response_data = {}
        content_type = response.get("contentType", "")

        if content_type == "application/json":
            raw = response["response"].read()
            response_data = json.loads(raw)
        elif "text/event-stream" in content_type:
            for line in response["response"].iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        response_data = json.loads(line[6:].strip())
                    if "response" in response_data:
                        break

        return response_data


class AgentCoreRolloutEngine(AgentRolloutEngine):
    """
    Rollout engine that invokes an AgentCore runtime.

    The runtime must already be deployed. Formula parameters are synced
    via ensure_sync_params() and included in each invocation payload.

    Args:
        formula: The Formula being optimized.
        agent_arn: ARN of the deployed AgentCore agent runtime.
        region_name: AWS region where the runtime is deployed.
        boto_config: BotocoreConfig or dict to merge with defaults.
        num_rollouts: Default number of rollouts per data sample.
        num_workers: Number of parallel workers for concurrent invocations.

    Example:
        engine = AgentCoreRolloutEngine(
            formula=formula,
            agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/abc",
            num_workers=4,
        )
    """

    def __init__(
        self,
        formula: Formula,
        agent_arn: str,
        region_name: str = "us-west-2",
        boto_config=None,
        num_rollouts: int = 1,
        num_workers: int = 1,
    ):
        super().__init__(formula, num_rollouts)
        self.num_workers = num_workers
        self._synced_params: dict = {}
        self._client = AgentCoreClient(
            agent_arn=agent_arn,
            region_name=region_name,
            boto_config=boto_config,
        )

        logger.info(f"Initialized AgentCoreRolloutEngine (num_workers={num_workers})")

    def ensure_sync_params(self) -> None:
        """Capture current formula params for the upcoming batch."""
        self._synced_params = self.formula.get_tunable_params()

    def generate_batch(self, data_samples: list[dict]) -> Iterator[Rollout]:
        """Generate rollouts by invoking the AgentCore runtime."""
        self.ensure_sync_params()
        tasks = expand_for_num_rollouts(data_samples, self.num_rollouts)
        results = run_parallel(self._invoke_runtime, tasks, self.num_workers)
        yield from results

    def _invoke_runtime(self, data_sample: dict) -> Rollout:
        """Invoke the AgentCore runtime for a single data sample."""
        payload = {
            "data_sample": data_sample,
            "params": self._synced_params,
        }

        response_data = self._client.invoke(payload)

        messages = response_data.get("messages", [])

        metadata = {
            "response_text": str(response_data.get("response", "")),
            "session_id": response_data.get("session_id", ""),
            "eval_result": response_data.get("eval_result", {}),
        }

        return Rollout(
            data_sample=data_sample,
            messages=messages,
            metadata=metadata,
        )
