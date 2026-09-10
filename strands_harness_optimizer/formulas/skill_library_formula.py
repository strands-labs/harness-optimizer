"""
Built-in formula for optimizing a whole Strands Agent Skill *library*.

Where :class:`~strands_harness_optimizer.formulas.SkillFormula` tunes the text of
ONE skill, ``SkillLibraryFormula`` owns a **library** of skills and lets an optimizer
change its membership: create a skill that does not exist, revise one that does,
retire one that hurts. That is a diff over a set, not a value to overwrite, and it
is why the tunable parameter here is a *directory* rather than the skill text.

A skill is a DIRECTORY (``<name>/SKILL.md``, plus any ``scripts/`` or
``resources/`` it needs), so the directory is the unit this formula manages:

    skill_dir/
      web-search/SKILL.md
      checkout/SKILL.md

``get_tunable_params`` therefore returns ``{"skill_dir": "<path>"}``. The optimizer
does not rewrite that string — it writes a DECISION TREE and calls
``update_params`` with the folder it wrote to; see :meth:`update_params`.

Frontmatter is validated on the way in. A ``SKILL.md`` without a YAML
``description:`` can never be loaded by the runtime, so accepting one would spend a
whole evaluation to measure nothing — the skill would simply never fire, which is
indistinguishable from "the skill did not help".
"""

import logging
import os
import shutil
from typing import TYPE_CHECKING, Optional

from strands.hooks.events import BeforeInvocationEvent

from .formula import Formula

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands import Skill

logger = logging.getLogger(__name__)

# Written by the optimizer agent, read by _collect_decisions.
CREATE_DIRS = ("create", "deploy")  # "deploy" accepted as a legacy alias
OPTIMIZE_DIR = "optimize"
RETIRE_FILE = "retire.txt"
MANIFEST_FILE = "manifest.json"

SKILL_FILE = "SKILL.md"
# Bound on how far to read looking for the closing `---`. Frontmatter is metadata;
# anything beyond this is the body.
_FRONTMATTER_MAX_BYTES = 8000


class SkillLibraryFormula(Formula):
    """Formula that manages a *set* of Strands Skills as a tunable directory.

    The tunable parameter is the skill directory, not the skill text::

        formula.get_tunable_params()      # {'skill_dir': '/path/to/skills'}

    An optimizer changes the set by writing a decision tree and handing back the
    folder it wrote to::

        formula.update_params({"decisions_dir": "/path/to/optimizer/output"})

    which is resolved as ``existing - retired - optimized_old + created``.

    Example:
        from strands import Agent, AgentSkills

        formula = SkillLibraryFormula(skill_dir="./skills")     # or None for a cold start

        # in-process agents: the adapter pushes the set into AgentSkills
        agent = Agent(plugins=[AgentSkills(skills=[])])
        apply_formulas_on_strands_agent(agent, [formula])

        # remote agents: materialize, then deliver the folder yourself
        formula.materialize("./optimized_skills")

    Args:
        skill_dir: Directory holding the starting skill set, as
            ``<skill_dir>/<name>/SKILL.md``. ``None`` or a missing path means a
            COLD START — an empty library, which is a normal first iteration.
        strict: If True, a skill whose frontmatter is unusable raises instead of
            being skipped with a warning. Off by default so one damaged skill in a
            starting set cannot abort a run.
    """

    def __init__(self, skill_dir: Optional[str] = None, strict: bool = False):
        super().__init__("skill_library_formula", [BeforeInvocationEvent])
        self.strict = strict
        # name -> directory holding that skill. The values are absolute paths that
        # may live outside skill_dir: an optimized skill is read straight from the
        # optimizer's output folder until materialize() collects the set.
        self.members: dict[str, str] = {}
        self.skill_dir = skill_dir
        if skill_dir:
            self.members = _load_skill_dirs(skill_dir, strict=strict)
            logger.info(
                "SkillLibraryFormula: loaded %d skill(s) from %s", len(self.members), skill_dir
            )
        else:
            logger.info("SkillLibraryFormula: cold start, no skills deployed")

    # ── Formula protocol ─────────────────────────────────────────────────────
    def process(self, context: dict, **kwargs) -> dict:
        """Hand the current set to the adapter, for IN-PROCESS agents.

        Returns the skills under the ``"skills"`` key; ``StrandsAdapter`` pushes
        them into the agent's ``AgentSkills`` plugin via ``set_available_skills()``.

        ``Skill`` objects are rebuilt from disk on every invocation rather than
        cached. ``AgentSkills`` reads a filesystem path once, at ``init_agent``
        time, so a folder rewritten mid-run would NOT reach a live agent; passing
        freshly-loaded objects is what makes an optimizer update take effect.

        Remote agents never call this — they receive the set through the payload
        (see :meth:`materialize`), and their own container-side plugin loads it.
        """
        if not self.members:
            # Cold start. Returning {"skills": []} would CLEAR a plugin that was
            # constructed with skills of its own, so leave the context untouched.
            return context
        skills = self._load_skills()
        if not skills:
            return context
        return {"skills": skills}

    def get_tunable_params(self) -> dict:
        """Return the skill directory as the tunable parameter.

        A directory, not the skill text: a skill is a tree of files, and the set's
        membership is itself what gets optimized. Both are outside what a
        ``{name: text}`` mapping can express.
        """
        return {"skill_dir": self.skill_dir or ""}

    def update_params(self, params: dict) -> None:
        """Apply the optimizer's decisions to the set.

        Accepts either form:

        ``{"decisions_dir": "<folder>"}``
            The folder an optimizer agent wrote, holding ``create/<name>/``,
            ``optimize/<name>/`` and ``retire.txt``. Resolved as
            ``existing - retired - optimized_old + created``.

        ``{"skill_dir": "<folder>"}``
            Replace the set wholesale from an already-resolved flat folder. Used
            when restoring a checkpoint rather than applying a step.

        Skills whose frontmatter cannot be loaded are REJECTED here rather than at
        evaluation time — the runtime keys on the frontmatter ``description`` to
        decide whether to load a skill, so a malformed one silently never fires.
        """
        if "skill_dir" in params and "decisions_dir" not in params:
            new_dir = params["skill_dir"] or None
            self.skill_dir = new_dir
            self.members = _load_skill_dirs(new_dir, strict=self.strict) if new_dir else {}
            logger.info(
                "SkillLibraryFormula: set replaced from %s (%d skill(s))",
                new_dir,
                len(self.members),
            )
            return

        decisions_dir = params.get("decisions_dir")
        if not decisions_dir:
            raise ValueError(
                "SkillLibraryFormula.update_params needs 'decisions_dir' (a folder with "
                "create/, optimize/ and/or retire.txt) or 'skill_dir' (an "
                f"already-resolved flat skill folder). Got keys: {sorted(params)}"
            )

        decisions = collect_decisions(decisions_dir)
        added, modified, removed = self._apply_decisions(decisions)
        logger.info(
            "SkillLibraryFormula: +%d ~%d -%d -> %d skill(s)",
            len(added),
            len(modified),
            len(removed),
            len(self.members),
        )

    # ── set algebra ──────────────────────────────────────────────────────────
    def _apply_decisions(self, decisions: dict) -> tuple[list, list, list]:
        """``existing - retired - optimized_old + created``.

        MERGE and SPLIT need no cases of their own: both are written as a create
        plus a retire of every source, so they resolve through the same algebra.
        """
        added: list[str] = []
        modified: list[str] = []
        removed: list[str] = []

        for name in decisions.get("retire", []):
            if self.members.pop(name, None) is not None:
                removed.append(name)
            else:
                # Not fatal, but it means the optimizer's view of the library
                # disagreed with ours -- worth saying out loud.
                logger.warning(
                    "retire %r: not in the current set, ignored (known: %s)",
                    name,
                    sorted(self.members),
                )

        for path in decisions.get("optimize", []):
            name = os.path.basename(os.path.normpath(path))
            if not self._accept(name, path):
                continue
            self.members[name] = path
            modified.append(name)

        for path in decisions.get("create", []):
            name = os.path.basename(os.path.normpath(path))
            if not self._accept(name, path):
                continue
            # A "create" naming a deployed skill is a revision in practice; record
            # it as one so the counts describe what happened.
            if name in self.members:
                modified.append(name)
            else:
                added.append(name)
            self.members[name] = path

        return added, modified, removed

    def _accept(self, name: str, path: str) -> bool:
        """Gate a skill on loadable frontmatter, before it enters the set."""
        problems = validate_skill_dir(path)
        if not problems:
            return True
        msg = f"rejecting skill {name!r} at {path}: " + "; ".join(problems)
        if self.strict:
            raise ValueError(msg)
        logger.warning("%s", msg)
        return False

    # ── artifact ─────────────────────────────────────────────────────────────
    def materialize(self, out_dir: str) -> str:
        """Write the resolved set to ``out_dir`` as a flat ``<name>/SKILL.md`` tree.

        The optimizer's output is a set of DECISIONS (``create/``, ``optimize/``,
        ``retire.txt``), and members may still point into it. This collects them
        into the one flat folder an agent can actually load, and repoints
        ``skill_dir`` at it.

        Copies whole trees, so a skill carrying ``scripts/`` or ``resources/``
        survives intact.

        Delivering that folder to a REMOTE agent (uploading it, or packing it into
        a payload) is deliberately not done here: it needs a bucket, credentials
        and a transport this library has no business assuming. See the skills
        example for helpers.

        Returns:
            The path written (``out_dir``).
        """
        os.makedirs(out_dir, exist_ok=True)
        for name, src in sorted(self.members.items()):
            dest = os.path.join(out_dir, name)
            # A second materialize() to the same folder would otherwise delete the
            # destination and then copy FROM it: the first call repoints members at
            # out_dir, so src and dest are the same directory.
            if os.path.realpath(src) == os.path.realpath(dest):
                continue
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        self.skill_dir = out_dir
        self.members = {n: os.path.join(out_dir, n) for n in self.members}
        logger.info(
            "SkillLibraryFormula: materialized %d skill(s) -> %s", len(self.members), out_dir
        )
        return out_dir

    # ── helpers ──────────────────────────────────────────────────────────────
    def _load_skills(self) -> list["Skill"]:
        """Build ``Skill`` objects for the current set, skipping unloadable ones."""
        from strands import Skill

        out = []
        for name, path in sorted(self.members.items()):
            try:
                out.append(Skill.from_file(path))
            except Exception as e:  # noqa: BLE001 - a bad skill must not kill the run
                logger.warning("skill %r at %s could not be loaded: %s", name, path, e)
        return out

    @property
    def skill_names(self) -> list[str]:
        """Names in the current set."""
        return sorted(self.members)


# ── module-level helpers (shared with the optimizer) ─────────────────────────


def collect_decisions(out_dir: str) -> dict:
    """Read an optimizer agent's decision tree off disk.

    Returns ``{"create": [paths], "optimize": [paths], "retire": [names]}``.
    Presence on disk is the source of truth — not what the agent said it did — so
    a write that silently failed cannot be recorded as a change.
    """
    out: dict = {"create": [], "optimize": [], "retire": []}
    for folder in CREATE_DIRS:
        d = os.path.join(out_dir, folder)
        if os.path.isdir(d):
            out["create"] = skill_dirs_in(d)
            break
    d = os.path.join(out_dir, OPTIMIZE_DIR)
    if os.path.isdir(d):
        out["optimize"] = skill_dirs_in(d)
    f = os.path.join(out_dir, RETIRE_FILE)
    if os.path.isfile(f):
        with open(f) as fh:
            out["retire"] = [ln.strip() for ln in fh if ln.strip()]
    return out


def skill_dirs_in(root: str) -> list[str]:
    """Directories containing a ``SKILL.md``, at either depth.

    ``root`` itself may be the skill (``<root>/SKILL.md``) or a parent of several.
    """
    if not os.path.isdir(root):
        return []
    if os.path.isfile(os.path.join(root, SKILL_FILE)):
        return [root]
    out = []
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, SKILL_FILE)):
            out.append(p)
    return out


def read_frontmatter(skill_md: str) -> str:
    """Return the leading ``---`` block, however long its values are.

    Read up to the closing delimiter rather than a fixed number of characters: a
    legitimate multi-line ``description:`` easily overruns a guessed window, and
    the closing ``---`` then falls outside it — reporting a VALID skill as missing
    its keys, which would drop a correct artifact.
    """
    with open(skill_md, encoding="utf-8", errors="replace") as f:
        text = f.read(_FRONTMATTER_MAX_BYTES)
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return ""
    body = stripped[3:]
    end = body.find("\n---")
    return body if end == -1 else body[:end]


def validate_skill_dir(path: str) -> list[str]:
    """Structural checks a skill must pass to be loadable at runtime.

    Returns a list of problems; empty means usable. ``description`` is the field
    the runtime matches on to decide whether to load a skill, so a skill missing
    it can never fire — an expensive silent no-op, hence a hard reject.
    """
    problems: list[str] = []
    md = os.path.join(path, SKILL_FILE)
    if not os.path.isfile(md):
        return [f"no {SKILL_FILE} in {path}"]
    block = read_frontmatter(md)
    if not block:
        return [
            f"{SKILL_FILE} has no YAML frontmatter. It must OPEN with a '---' block "
            "containing `name:` and `description:`. The runtime reads `description` "
            "to decide whether to load the skill, so without it this skill can "
            "never fire (a '## Trigger' heading does not work)."
        ]
    if "description:" not in block:
        problems.append(
            "frontmatter has no `description:` -- that field IS the trigger the "
            "runtime matches on."
        )
    if "name:" not in block:
        problems.append("frontmatter has no `name:`.")
    return problems


def _load_skill_dirs(skill_dir: Optional[str], *, strict: bool = False) -> dict:
    """Map ``name -> directory`` for a flat skill folder, skipping unusable ones."""
    if not skill_dir or not os.path.isdir(skill_dir):
        if skill_dir and strict:
            raise FileNotFoundError(f"skill_dir does not exist: {skill_dir}")
        if skill_dir:
            logger.warning("skill_dir %s does not exist; starting cold", skill_dir)
        return {}
    members: dict[str, str] = {}
    for path in skill_dirs_in(skill_dir):
        name = os.path.basename(os.path.normpath(path))
        problems = validate_skill_dir(path)
        if problems:
            msg = f"skipping skill {name!r} in {skill_dir}: " + "; ".join(problems)
            if strict:
                raise ValueError(msg)
            logger.warning("%s", msg)
            continue
        members[name] = path
    return members


def render_skill_index(skill_dir: Optional[str]) -> str:
    """Render the library as ``### <name>`` + frontmatter, for a prompt.

    TRIGGERS ONLY, deliberately. The optimizer agent needs to know what already
    exists before deciding whether to create something, but reading every full
    body up front anchors it toward editing them — so the index carries each
    skill's frontmatter and the agent reads a body only for a skill some real
    pattern implicates.

    Derived from ``skill_dir`` on every render, so the index and the folder the
    agent inspects cannot disagree.
    """
    members = _load_skill_dirs(skill_dir)
    if not members:
        return "(none deployed)"
    blocks = []
    for name, path in sorted(members.items()):
        block = read_frontmatter(os.path.join(path, SKILL_FILE)).strip()
        blocks.append(f"### {name}\n{path}\n---\n{block}\n---")
    return "\n\n".join(blocks)
