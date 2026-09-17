"""Harness-side rendering and checking for the multi-surface optimizer.

Everything here is computed by the harness, not by the agent, for two reasons.
Cost: an agent asked to count calls per tool over twenty traces spends hundreds of
shell turns re-deriving what a Counter gives for free. Trust: the agent is the
party whose output is being checked, so the checks read files on disk and never
the agent's own statements about them.

Inputs are the trace files the agent itself reads (``*.json`` in the traces
folder), so the census and the evidence cannot disagree.
"""

import json
import logging
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ...datamodels import Reward
from ...formulas.skill_library_formula import SKILL_FILE, validate_skill_dir
from .objective import TASK_SUCCESS_TERM, explanations_for, objective_terms

logger = logging.getLogger(__name__)

# A tool result whose status is not recorded is judged by its text.
ERROR_WORDS = ("error", "exception", "failed", "traceback")

# Where the agent writes each surface, relative to the step directory.
SKILLS_DIR = "skills"
SKILL_DECISION_DIRS = ("create", "update", "optimize")
PROMPT_FILE = os.path.join("system_prompt", "optimized_prompt.yaml")
TOOL_DESC_FILE = os.path.join("tool_descriptions", "optimized_tool_descriptions.yaml")
FINDINGS_FILE = "findings.json"

SURFACES = ("system_prompt", "skills", "tool_description")

# Definitions shown next to each objective term. Names are the built-in evaluator
# names of Amazon Bedrock AgentCore; the wording is theirs. A score named
# `Conciseness` is a number; the definition is what makes a fix conceivable.
OBJECTIVE_TERM_DEFS = {
    TASK_SUCCESS_TERM: "The benchmark-provided score for successful task completion.",
    "GoalSuccessRate": (
        "Task Completion Metric. Evaluates whether the conversation successfully meets "
        "the user's goals."
    ),
    "InstructionFollowing": (
        "Response Quality Metric. Measures how well the agent follows the provided "
        "system instructions."
    ),
    "Correctness": (
        "Response Quality Metric. Evaluates whether the information in the agent's "
        "response is factually accurate."
    ),
    "Helpfulness": (
        "Response Quality Metric. Evaluates from the user's perspective how useful and "
        "valuable the agent's response is."
    ),
    "Conciseness": (
        "Response Quality Metric. Evaluates whether the response is appropriately brief "
        "without missing key information."
    ),
    "Coherence": (
        "Response Quality Metric. Evaluates whether the response is logically structured "
        "and coherent."
    ),
    "Faithfulness": (
        "Response Quality Metric. Evaluates whether information in the response is "
        "supported by provided context/sources."
    ),
    "ResponseRelevance": (
        "Response Quality Metric. Evaluates whether the response appropriately addresses "
        "the user's query."
    ),
    "ToolSelectionAccuracy": (
        "Component Level Metric. Evaluates whether the agent selected the appropriate "
        "tool for the task."
    ),
    "ToolParameterAccuracy": (
        "Component Level Metric. Evaluates how accurately the agent extracts parameters "
        "from user queries."
    ),
}

# Explanations are rendered only where the judge withheld marks, capped per term,
# and trimmed: a first draft explained perfect scores too and spent 21K characters
# saying nothing was wrong.
EXPLAIN_BELOW = 1.0
MAX_EXPLANATIONS_PER_TERM = 5
EXPLANATION_CHARS = 450


# ── traces ───────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    tool: str
    params: Any
    result_text: str
    is_error: bool


@dataclass
class TraceSummary:
    filename: str
    reward: float
    calls: list[ToolCall] = field(default_factory=list)


def _result_text(tr: dict) -> str:
    content = tr.get("content", [])
    if isinstance(content, str):
        return content
    parts = []
    for item in content if isinstance(content, list) else []:
        if isinstance(item, dict):
            if "text" in item:
                parts.append(str(item["text"]))
            elif "json" in item:
                parts.append(json.dumps(item["json"], default=str))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _calls_from_messages(messages: list) -> list[ToolCall]:
    calls: list[ToolCall] = []
    pending = None
    for msg in messages or []:
        for item in (msg or {}).get("content") or []:
            if not isinstance(item, dict):
                continue
            if "toolUse" in item and isinstance(item["toolUse"], dict):
                tu = item["toolUse"]
                pending = (tu.get("name", "unknown"), tu.get("input", {}))
            elif "toolResult" in item and isinstance(item["toolResult"], dict) and pending:
                tr = item["toolResult"]
                text = _result_text(tr)
                status = tr.get("status")
                if status in ("success", "error"):
                    is_error = status == "error"
                else:
                    is_error = any(w in text.lower() for w in ERROR_WORDS)
                calls.append(ToolCall(pending[0], pending[1], text, is_error))
                pending = None
    return calls


def load_trace_summaries(traces_folder: str) -> list[TraceSummary]:
    """One summary per ``*.json`` in the folder, read the way the agent reads them."""
    out: list[TraceSummary] = []
    for name in sorted(os.listdir(traces_folder)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(traces_folder, name)
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("census: skipping unreadable trace %s: %s", name, e)
            continue
        reward = d.get("reward")
        try:
            reward = float(reward) if reward is not None else 0.0
        except (TypeError, ValueError):
            reward = 0.0
        messages = ((d.get("response") or {}).get("messages")) or (
            (d.get("data") or {}).get("messages")
        )
        out.append(TraceSummary(name, reward, _calls_from_messages(messages or [])))
    return out


# ── census ───────────────────────────────────────────────────────────────────


@dataclass
class ToolStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    duplicate_calls: int = 0
    calls_in_lower_objective: int = 0
    calls_in_higher_objective: int = 0

    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.total_calls if self.total_calls else 0.0


def per_tool_stats(traces: list[TraceSummary], success_threshold: float) -> dict[str, ToolStats]:
    stats: dict[str, ToolStats] = defaultdict(ToolStats)
    for tr in traces:
        seen_in_episode: dict[str, set] = defaultdict(set)
        high = tr.reward >= success_threshold
        for c in tr.calls:
            s = stats[c.tool]
            s.total_calls += 1
            try:
                sig = json.dumps(c.params, sort_keys=True, default=str)
            except (TypeError, ValueError):
                sig = str(c.params)
            if sig in seen_in_episode[c.tool]:
                s.duplicate_calls += 1
            else:
                seen_in_episode[c.tool].add(sig)
            if c.is_error:
                s.failed_calls += 1
            else:
                s.successful_calls += 1
            if high:
                s.calls_in_higher_objective += 1
            else:
                s.calls_in_lower_objective += 1
    return dict(stats)


def render_census(
    traces: list[TraceSummary],
    agent_tools: Optional[list[str]] = None,
    success_threshold: float = 0.5,
) -> str:
    """Per-trace facts the agent would otherwise grep for, plus per-tool statistics.

    Rows are sorted by objective ascending so the weakest episodes come first, the
    order the task prompt tells the agent to read them in.
    """
    rows = sorted(traces, key=lambda t: t.reward)
    lines = ["  objective  calls  errs  trace file"]
    for t in rows:
        errs = sum(1 for c in t.calls if c.is_error)
        lines.append(f"  {t.reward:>9.2f}{len(t.calls):>7}{errs:>6}  {t.filename}")
    n = len(rows)
    lower = sum(1 for t in rows if t.reward < success_threshold)
    lines.append(
        f"  -- {n} traces by total weighted objective: {n - lower} at or above "
        f"{success_threshold:.2f}, {lower} below {success_threshold:.2f}."
    )
    stats = per_tool_stats(traces, success_threshold)
    if stats:
        lines.append("  per-tool:")
        for tool, s in sorted(stats.items(), key=lambda kv: -kv[1].total_calls):
            lines.append(
                f"    {tool:<32} calls={s.total_calls:4d} ok={s.successful_calls:4d} "
                f"failed={s.failed_calls:3d} dup={s.duplicate_calls:3d} "
                f"rate={s.success_rate:.2f}  "
                f"(in lower-objective episodes: {s.calls_in_lower_objective})"
            )
    if agent_tools:
        unused = sorted(set(agent_tools) - set(stats))
        if unused:
            lines.append(f"  never called in these traces: {', '.join(unused)}")
    return "\n".join(lines)


# ── objective report ─────────────────────────────────────────────────────────


def render_objective_report(
    rewards: list[Reward],
    weights: dict[str, float],
    names: Optional[list[str]] = None,
    definitions: Optional[dict[str, str]] = None,
) -> str:
    """The configured objective, its definitions, and the judge's own explanations.

    Only configured terms appear. Raises if a term has no score in any reward,
    which ``objective_terms`` already guarantees per reward. ``names`` are the trace
    filenames aligned with ``rewards``; an explanation is labelled with its trace so
    the agent can cite it the way the task prompt requires.

    ``definitions`` is the user's ``{term: what it measures}``. It wins over the
    built-in wording, which is consulted only for a configured term whose name
    matches a built-in evaluator; a term with neither is rendered as undefined and
    logged, since a bare number gives the agent nothing to reason from.
    """
    defs = dict(OBJECTIVE_TERM_DEFS)
    defs.update({str(k): str(v) for k, v in (definitions or {}).items() if v})
    undefined = [k for k in weights if k not in defs]
    if undefined:
        logger.warning(
            "objective term(s) %s have no definition; pass objective_definitions so the "
            "agent knows what the score measures",
            undefined,
        )
    if not rewards:
        return ""
    if names is not None and len(names) != len(rewards):
        raise ValueError("names must align with rewards")
    agg: dict[str, list[float]] = defaultdict(list)
    expl: list[tuple[float, str, int, str]] = []
    for i, r in enumerate(rewards):
        terms = objective_terms(r, weights)
        for k, v in terms.items():
            agg[k].append(v)
        for k, text in explanations_for(r, weights).items():
            v = terms.get(k)
            if v is not None and v < EXPLAIN_BELOW:
                expl.append((v, k, i, text))

    keep = sorted(weights, key=lambda k: (-weights[k], k))
    out = [
        "**The objective.** These are exactly the terms selected by the customer. The "
        "total weighted reward orders and samples episodes. Use the supplied weights when "
        "improvements trade off. Do not introduce or privilege any unlisted score. Each "
        "episode's component scores are on its own record under `objective.terms`.\n",
        "| objective term | weight | what it measures | mean over these episodes |",
        "|---|---|---|---|",
    ]
    for k in keep:
        mean = statistics.mean(agg[k]) if agg.get(k) else float("nan")
        out.append(
            f"| {k} | {weights[k]:.2f} | "
            f"{defs.get(k, '(no definition supplied)')} | {mean:.2f} |"
        )

    if expl:
        expl.sort(key=lambda x: (x[0], x[1], x[2]))
        out.append(
            "\n**Why the judge scored these episodes low, in its own words.** Read these "
            "before proposing anything: they name the behaviour that cost the score, which "
            "the number alone does not.\n"
        )
        seen: Counter = Counter()
        for v, k, i, text in expl:
            if seen[k] >= MAX_EXPLANATIONS_PER_TERM:
                continue
            seen[k] += 1
            label = f"`{names[i]}`" if names is not None else f"episode {i}"
            out.append(
                f"- **{k} = {v:g}** on {label}: {' '.join(text.split())[:EXPLANATION_CHARS]}"
            )
    return "\n".join(out)


# ── deployed skills ──────────────────────────────────────────────────────────


def render_deployed_skills(members: dict[str, str], max_body_lines: int = 200) -> str:
    """The deployed skills IN FULL, frontmatter and body.

    Two of the prompt's questions cannot be answered from a name: whether a
    finding is `already_present_in` a skill (judged from its text) and whether the
    agent followed a skill it had loaded. Bodies sit in the cached prefix, so the
    tokens are close to free.
    """
    if not members:
        return "  (none deployed)"
    out: list[str] = []
    for name in sorted(members):
        md = os.path.join(members[name], SKILL_FILE)
        out.append(f"----- {name}")
        if not os.path.isfile(md):
            out.append("  (SKILL.md missing)")
            continue
        with open(md) as f:
            lines = f.read().splitlines()
        out.extend(lines[:max_body_lines])
        if len(lines) > max_body_lines:
            out.append(f"  ... [{len(lines) - max_body_lines} more lines; full file at {md}]")
    return "\n".join(out)


# ── structural validation ────────────────────────────────────────────────────


def surface_attempted(step_dir: str, surface: str) -> bool:
    """Did the agent write anything at all for this surface?

    Presence on disk, not the agent's word. A directory the agent created, even
    with a broken file in it, counts as an attempt; validation then says whether
    the attempt is usable.
    """
    if surface == "skills":
        d = os.path.join(step_dir, SKILLS_DIR)
        return os.path.isdir(d) and any(os.scandir(d))
    if surface == "system_prompt":
        return os.path.isdir(os.path.join(step_dir, os.path.dirname(PROMPT_FILE)))
    if surface == "tool_description":
        return os.path.isdir(os.path.join(step_dir, os.path.dirname(TOOL_DESC_FILE)))
    raise ValueError(f"unknown surface {surface!r}")


def validate_surface(step_dir: str, surface: str) -> list[str]:
    """Structural checks an artifact must pass to take effect at all.

    Empty list means usable. These are the checks whose failure would make the
    artifact silently do nothing: a skill without frontmatter never loads, a prompt
    YAML that does not parse never applies. Whether the content is any good is a
    different question and not asked here.
    """
    problems: list[str] = []
    if surface == "skills":
        root = os.path.join(step_dir, SKILLS_DIR)
        found: list[str] = []
        for dirpath, _dirs, files in os.walk(root):
            if SKILL_FILE in files:
                found.append(dirpath)
        if not found:
            return [f"no {SKILL_FILE} written anywhere under {SKILLS_DIR}/"]
        for skill_dir in found:
            rel = os.path.relpath(skill_dir, root)
            parts = rel.split(os.sep)
            if len(parts) != 2 or parts[0] not in SKILL_DECISION_DIRS:
                problems.append(
                    f"{rel}/{SKILL_FILE}: skills must live under "
                    f"{SKILLS_DIR}/create/<name>/ or {SKILLS_DIR}/update/<name>/"
                )
                continue
            problems.extend(f"{rel}: {p}" for p in validate_skill_dir(skill_dir))
        return problems

    import yaml

    if surface == "system_prompt":
        p = os.path.join(step_dir, PROMPT_FILE)
        if not os.path.isfile(p):
            return [f"{PROMPT_FILE} was not written"]
        try:
            with open(p) as f:
                d = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return [f"{PROMPT_FILE} is not valid YAML: {e}"]
        if not isinstance(d, dict) or not str(d.get("system_prompt") or "").strip():
            problems.append(f"{PROMPT_FILE} must be YAML with a non-empty `system_prompt:` string")
        return problems

    if surface == "tool_description":
        p = os.path.join(step_dir, TOOL_DESC_FILE)
        if not os.path.isfile(p):
            return [f"{TOOL_DESC_FILE} was not written"]
        try:
            with open(p) as f:
                d = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            return [f"{TOOL_DESC_FILE} is not valid YAML: {e}"]
        inner = d.get("tool_descriptions") if isinstance(d, dict) else None
        if not isinstance(inner, dict) or not inner:
            problems.append(
                f"{TOOL_DESC_FILE} needs a non-empty `tool_descriptions:` mapping of "
                "tool name to description"
            )
        elif any(not isinstance(v, str) for v in inner.values()):
            problems.append(f"{TOOL_DESC_FILE}: every description must be a string")
        return problems

    raise ValueError(f"unknown surface {surface!r}")


def read_findings(step_dir: str) -> tuple[Optional[dict], str]:
    """``(findings.json as a dict, "")`` or ``(None, why it is unusable)``."""
    p = os.path.join(step_dir, FINDINGS_FILE)
    if not os.path.isfile(p):
        return None, f"{FINDINGS_FILE} was not written"
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{FINDINGS_FILE} is not valid JSON: {e}"
    if not isinstance(d, dict) or not isinstance(d.get("findings"), list):
        return None, f"{FINDINGS_FILE} must be an object with a `findings` array"
    return d, ""
