"""
Harness Optimizer — A framework for optimizing LLM agent context through Formulas.
"""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback when not built via hatch-vcs
    __version__ = "0.0.0.dev0"

# Load compat module to register legacy import paths (e.g., harness_optimizer.processors)
from . import compat as _compat  # noqa: F401,E402
