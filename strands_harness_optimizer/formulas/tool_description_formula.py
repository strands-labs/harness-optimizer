"""
Built-in formula for optimizing an agent's tool descriptions.

A tool description is the text the model reads when deciding whether to call a
tool, how to fill its inputs, and how to read its result. It is a third place,
next to the system prompt and the skills, where guidance can reach the agent, and
it is the right place for guidance that is local to one tool.

The formula keeps two maps:

``base``
    The full ``{tool_name: description}`` the agent starts with, one entry per
    tool in its toolset. Never mutated. It defines which names are legal and is
    what a reflector reads as "the current tool descriptions".

``overrides``
    Sparse: only the tools an optimizer has edited, carried forward across steps.
    This is what gets DELIVERED. A runtime patches only the names it receives and
    every other tool keeps its own description, so sending the full map would
    overwrite a runtime's text with the seed copy wherever the two have drifted,
    and would make "which tools did the optimizer change" unanswerable.

``effective()`` is ``base`` with ``overrides`` applied, and is the view a reflector
should be shown.
"""

import logging
from typing import Optional

from strands.hooks.events import BeforeInvocationEvent

from .formula import Formula

logger = logging.getLogger(__name__)

# The wire key, in payloads, contexts and the optimizer's output YAML alike.
TOOL_DESCRIPTIONS_KEY = "tool_descriptions"

# Bedrock rejects longer tool descriptions outright, and a description this long
# has stopped being a contract and become a manual that belongs in a skill.
DEFAULT_MAX_CHARS = 800


class ToolDescriptionFormula(Formula):
    """Formula whose tunable parameter is a sparse set of tool-description overrides.

    Example:
        formula = ToolDescriptionFormula(
            base={"search": "Search the catalog.", "click": "Click an element."}
        )
        formula.get_tunable_params()            # {'tool_descriptions': {}}

        formula.update_params({"tool_descriptions": {"search": "Search ... by keyword."}})
        formula.get_tunable_params()            # {'tool_descriptions': {'search': '...'}}
        formula.effective()                     # base with 'search' replaced

    In-process agents receive the overrides through ``StrandsAdapter``, which patches
    the agent's tool specs. Remote agents receive them as the ``tool_descriptions``
    payload key and patch on their side.

    Args:
        base: Full ``{tool_name: description}`` for the agent's toolset. An EMPTY
            base means the toolset is unknown, in which case override names cannot
            be checked and every name is accepted with a log line.
        overrides: Overrides already in force (a previous run's output), validated
            the same way ``update_params`` validates new ones.
        strict: If True, an override for a name not in ``base``, or one over
            ``max_chars``, raises instead of being dropped with a warning.
        max_chars: Upper bound on one description. Defaults to 800.
    """

    def __init__(
        self,
        base: dict[str, str],
        overrides: Optional[dict[str, str]] = None,
        strict: bool = False,
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        super().__init__("tool_description_formula", [BeforeInvocationEvent])
        self.strict = strict
        self.max_chars = max_chars
        self.base: dict[str, str] = {str(k): str(v) for k, v in (base or {}).items()}
        self.overrides: dict[str, str] = {}
        if not self.base:
            logger.info(
                "ToolDescriptionFormula: empty base, so override names cannot be "
                "checked against a toolset"
            )
        if overrides:
            self._merge(overrides)

    # ── Formula protocol ─────────────────────────────────────────────────────
    def process(self, context: dict, **kwargs) -> dict:
        """Hand the overrides to the adapter, for IN-PROCESS agents.

        Returns them under ``"tool_descriptions"``; ``StrandsAdapter`` patches the
        named tools' specs. With no overrides the context is left untouched, so a
        formula that has not been optimized yet changes nothing.
        """
        if not self.overrides:
            return context
        return {TOOL_DESCRIPTIONS_KEY: dict(self.overrides)}

    def get_tunable_params(self) -> dict:
        """Return the sparse overrides, which is exactly what gets delivered."""
        return {TOOL_DESCRIPTIONS_KEY: dict(self.overrides)}

    def update_params(self, params: dict) -> None:
        """Merge an optimizer's edits into the overrides.

        ``params`` is ``{"tool_descriptions": {name: text, ...}}`` holding ONLY the
        tools the optimizer edited. Edited entries win; every other override
        survives. A name not in ``base`` is dropped (or raises under ``strict``),
        as is a description over ``max_chars``; an empty string is ignored, since
        no prompt offers a "delete a description" operation.
        """
        edits = params.get(TOOL_DESCRIPTIONS_KEY)
        if edits is None:
            return
        if not isinstance(edits, dict):
            raise ValueError(
                f"ToolDescriptionFormula.update_params expects '{TOOL_DESCRIPTIONS_KEY}' "
                f"to be a mapping, got {type(edits).__name__}"
            )
        accepted = self._merge(edits)
        logger.info(
            "ToolDescriptionFormula: %d edit(s) applied (%s), %d override(s) in force",
            len(accepted),
            ", ".join(sorted(accepted)) or "none",
            len(self.overrides),
        )

    # ── views ────────────────────────────────────────────────────────────────
    def effective(self) -> dict[str, str]:
        """``base`` with ``overrides`` applied: what the agent actually sees."""
        out = dict(self.base)
        out.update(self.overrides)
        return out

    @property
    def tool_names(self) -> list[str]:
        """The toolset, as far as this formula knows it."""
        return sorted(set(self.base) | set(self.overrides))

    def render_effective_yaml(self) -> str:
        """The effective map as ``tool_descriptions: {name: |\\n  text}`` YAML.

        This is the shape a reflector is shown and the shape it writes back, so
        one format serves both directions.
        """
        return dump_yaml(self.effective())

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(
        cls,
        base_path: str,
        overrides_path: Optional[str] = None,
        strict: bool = False,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> "ToolDescriptionFormula":
        """Build from YAML files of the ``tool_descriptions: {...}`` shape."""
        base = load_yaml(base_path)
        overrides = load_yaml(overrides_path) if overrides_path else None
        return cls(base, overrides=overrides, strict=strict, max_chars=max_chars)

    def validate_edits(self, edits: dict) -> dict[str, str]:
        """``{name: problem}`` for every entry; an empty string means acceptable.

        The same rules ``update_params`` applies, exposed so a caller can learn
        BEFORE applying whether an edit set would change anything at all.
        """
        return {
            str(name): self._problem(str(name), "" if text is None else str(text))
            for name, text in (edits or {}).items()
        }

    def replace_overrides(self, overrides: Optional[dict]) -> None:
        """Set the overrides wholesale (checkpoint restore), validated like edits.

        Unlike ``update_params`` this does not merge: an override absent from
        ``overrides`` is removed.
        """
        self.overrides = {}
        if overrides:
            self._merge(overrides)

    # ── internals ────────────────────────────────────────────────────────────
    def _merge(self, edits: dict) -> list[str]:
        accepted: list[str] = []
        for name, text in edits.items():
            name = str(name)
            text = "" if text is None else str(text)
            problem = self._problem(name, text)
            if problem == "empty":
                logger.warning("tool description for %r is empty -- keeping the current text", name)
                continue
            if problem:
                msg = f"rejecting tool description for {name!r}: {problem}"
                if self.strict:
                    raise ValueError(msg)
                logger.warning("%s", msg)
                continue
            self.overrides[name] = text
            accepted.append(name)
        return accepted

    def _problem(self, name: str, text: str) -> str:
        if not text.strip():
            return "empty"
        if self.base and name not in self.base:
            return f"not in the toolset (known: {sorted(self.base)})"
        if len(text) > self.max_chars:
            return f"{len(text)} chars, over the {self.max_chars}-char limit"
        return ""


def load_yaml(path: str) -> dict[str, str]:
    """Read a ``tool_descriptions: {name: text}`` YAML file into a plain dict.

    A file holding the bare mapping (no ``tool_descriptions:`` wrapper) is accepted
    too. Anything else raises, because a silently-empty map would make the first
    optimization step reject every edit as "not in the toolset".
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping, got {type(data).__name__}")
    inner = data.get(TOOL_DESCRIPTIONS_KEY, data)
    if not isinstance(inner, dict):
        raise ValueError(
            f"{path}: '{TOOL_DESCRIPTIONS_KEY}' must be a mapping, got {type(inner).__name__}"
        )
    return {str(k): ("" if v is None else str(v)) for k, v in inner.items()}


def dump_yaml(descriptions: dict[str, str]) -> str:
    """``tool_descriptions: {name: text}`` as YAML, multi-line text as literal blocks.

    The default dumper quotes and folds multi-line strings, doubling every blank
    line. A literal block (``|``) keeps each description readable exactly as the
    agent sees it, which matters because this text is shown to a reflector and
    edited by it.
    """
    import yaml

    class _Dumper(yaml.SafeDumper):
        pass

    def _str(dumper, value):
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    _Dumper.add_representer(str, _str)
    clean = {str(k): str(v).replace("\r\n", "\n").rstrip() for k, v in descriptions.items()}
    return yaml.dump(
        {TOOL_DESCRIPTIONS_KEY: clean},
        Dumper=_Dumper,
        sort_keys=True,
        width=100,
        default_flow_style=False,
        allow_unicode=True,
    )
