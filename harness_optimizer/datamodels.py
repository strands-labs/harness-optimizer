from dataclasses import dataclass, field
from typing import Protocol, Union

from strands import Agent

SupportedAgent = Union["Agent"]


@dataclass
class Reward:
    reward: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Rollout:
    data_sample: dict
    messages: list[dict]
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class AgentCreator(Protocol):
    """Protocol for callables that create fresh agent instances."""

    def __call__(self) -> SupportedAgent: ...


class AgentInvoker(Protocol):
    """Protocol for callables that invoke agents and return rollouts."""

    def __call__(self, agent: SupportedAgent, data_sample: dict) -> Rollout: ...
