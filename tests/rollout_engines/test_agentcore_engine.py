"""Unit tests for the AgentCore rollout engines' response handling (no network)."""

from strands_harness_optimizer.formulas import SystemPromptFormula
from strands_harness_optimizer.rollout_engines.agentcore_engine import (
    AgentCoreHTTPRolloutEngine,
    AgentCoreRolloutEngine,
)


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.payloads = []

    def invoke(self, payload, session_id=None):
        self.payloads.append(payload)
        return dict(self.response)


RESPONSE = {
    "response": "done",
    "messages": [{"role": "assistant", "content": [{"text": "done"}]}],
    "eval_result": {"reward": 1.0, "success": True},
    "skills_applied": 2,
    "tool_descriptions_applied": ["search"],
    "unrelated": "dropped",
}


def _http_engine(client):
    engine = AgentCoreHTTPRolloutEngine.__new__(AgentCoreHTTPRolloutEngine)
    engine.formula = SystemPromptFormula(system_prompt="p")
    engine._client = client
    engine._payload_mapper = None
    engine._synced_params = {}
    return engine


def _arn_engine(client):
    engine = AgentCoreRolloutEngine.__new__(AgentCoreRolloutEngine)
    engine.formula = SystemPromptFormula(system_prompt="p")
    engine._client = client
    engine._payload_mapper = None
    engine._synced_params = {}
    return engine


def test_http_engine_keeps_runtime_echoes_in_metadata():
    engine = _http_engine(_FakeClient(RESPONSE))
    rollout = engine._invoke_runtime({"task_id": "t1"})
    assert rollout.messages == RESPONSE["messages"]
    assert rollout.metadata["eval_result"] == RESPONSE["eval_result"]
    assert rollout.metadata["skills_applied"] == 2
    assert rollout.metadata["tool_descriptions_applied"] == ["search"]
    assert "unrelated" not in rollout.metadata


def test_arn_engine_keeps_runtime_echoes_in_metadata():
    engine = _arn_engine(_FakeClient(RESPONSE))
    rollout = engine._invoke_runtime({"task_id": "t1"})
    assert rollout.metadata["skills_applied"] == 2
    assert rollout.metadata["tool_descriptions_applied"] == ["search"]


def test_echo_keys_absent_when_runtime_does_not_send_them():
    """A runtime that predates the keys must leave them out, not fill in a default."""
    old = {
        k: v
        for k, v in RESPONSE.items()
        if k not in ("skills_applied", "tool_descriptions_applied")
    }
    rollout = _http_engine(_FakeClient(old))._invoke_runtime({"task_id": "t1"})
    assert "skills_applied" not in rollout.metadata
    assert "tool_descriptions_applied" not in rollout.metadata
