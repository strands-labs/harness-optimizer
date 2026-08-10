"""System prompt optimizers — optimize system prompts via agent-based reflection."""

from ..base_agentic_optimizer import BaseAgenticOptimizer
from .contrastive_reflection import ContrastiveReflectionOptimizer
from .multi_agent import MultiAgentOptimizer

# BaseAgenticOptimizer is Formula-agnostic and now lives one level up (it is shared
# with optimizers/skills/). Re-exported here so the historical
# `optimizers.system_prompt.BaseAgenticOptimizer` import keeps working.
__all__ = ["BaseAgenticOptimizer", "ContrastiveReflectionOptimizer", "MultiAgentOptimizer"]
