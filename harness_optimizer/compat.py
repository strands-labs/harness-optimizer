"""Legacy names and module aliases for backward compatibility.

This module serves as the target for the virtual `harness_optimizer.processors` module.
Only legacy names are exposed, with deprecation warnings:

    # Legacy names (deprecation warning)
    from harness_optimizer.processors import ContextUnitProcessor, SystemPromptProcessor

For new code, use:
    from harness_optimizer.formulas import Formula, SystemPromptFormula
"""

import sys
import warnings

# Register this module as harness_optimizer.processors so legacy imports work.
_parent = __name__.rsplit(".", 1)[0]  # "harness_optimizer"
sys.modules[f"{_parent}.processors"] = sys.modules[__name__]


def _get_aliases():
    from .formulas import Formula, SystemPromptFormula
    return {
        "ContextUnitProcessor": Formula,
        "SystemPromptProcessor": SystemPromptFormula,
    }


def __getattr__(name):
    aliases = _get_aliases()
    if name in aliases:
        warnings.warn(
            f"{name} is deprecated, use {aliases[name].__name__} "
            f"from harness_optimizer.formulas instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return aliases[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContextUnitProcessor",
    "SystemPromptProcessor",
]
