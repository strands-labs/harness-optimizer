"""
Optimizer that edits an agent's system prompt, skill library and tool descriptions
in one pass, routing each finding to the surface that reaches the agent when it
matters.

One reflector agent reads a sample of execution traces plus a harness-computed
census and objective table, and is shown the current prompt, the deployed skills
in full and the current tool descriptions. It writes ONLY the artifacts for the
surfaces it decided to change, plus a ``findings.json`` ledger, into a fresh step
directory::

    output_folder/step_0001/
      current/                       harness-written before the agent runs
        system_prompt.yaml           the prompt the agent starts from
        tool_descriptions.yaml       the effective tool descriptions
        rendered_*.txt               the prompts the reflector was given
      skills/create/<name>/SKILL.md  agent
      skills/update/<name>/SKILL.md  agent
      system_prompt/optimized_prompt.yaml               agent
      tool_descriptions/optimized_tool_descriptions.yaml agent
      findings.json                  agent, always
      skill_set/<name>/SKILL.md      harness-written after update_params
      attempts.log                   one line per agent attempt
      FAILED.txt                     only if the step raised

Every ``step()`` gets a NEW ``step_NNNN/``; earlier ones are never read again, so
a no-op pass cannot re-apply a previous step's artifacts. A failed step leaves its
directory behind for inspection and the next step takes the next unused number.

Validation here is STRUCTURAL: would this artifact take effect at all. A skill
without frontmatter never loads; a prompt YAML that does not parse never applies.
Whether the content is good is the reflector's job, and the runner does not
second-guess it beyond that. The outcome of a step is decided by what is on disk:

    some surfaces valid, some not   apply the valid ones, log each drop
    surfaces attempted, all invalid raise, listing every problem
    nothing attempted, findings ok  legitimate no-op
    nothing attempted, no findings  raise -- an empty pass and a crashed agent look
                                    the same otherwise
"""

import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, Optional

from ...datamodels import Reward
from ...formulas.multi_surface_formula import MultiSurfaceFormula
from ...formulas.tool_description_formula import TOOL_DESCRIPTIONS_KEY
from ...utils.templates import create_template, load_builtin_template
from ..base_agentic_optimizer import BaseAgenticOptimizer
from . import renderers
from .objective import normalize_weights, objective_terms, weighted_total

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jinja2 import Template

logger = logging.getLogger(__name__)

STEP_PREFIX = "step_"
CURRENT_DIR = "current"
CURRENT_PROMPT_FILE = "system_prompt.yaml"
CURRENT_TOOLS_FILE = "tool_descriptions.yaml"
SKILL_SET_DIR = "skill_set"
FAILED_FILE = "FAILED.txt"
ATTEMPTS_FILE = "attempts.log"

# Paths inside a step directory the AGENT writes. Cleared between retry attempts
# so a partial write from a failed attempt is not read as the next one's output.
AGENT_OWNED_PATHS = (
    renderers.SKILLS_DIR,
    os.path.dirname(renderers.PROMPT_FILE),
    os.path.dirname(renderers.TOOL_DESC_FILE),
    renderers.FINDINGS_FILE,
)

# Server-side failures worth re-running the whole call for. A ValidationException
# (bad model id, bad temperature) is deterministic and must surface at once.
TRANSIENT_ERRORS = (
    "internalServerException",
    "ServiceUnavailable",
    "ThrottlingException",
    "ModelTimeoutException",
    "modelStreamErrorException",
)

# Room for a long findings.json plus a full prompt copy in one turn.
DEFAULT_MAX_TOKENS = 32000

BUILTIN_SYSTEM_PROMPT = "multi_surface/system_prompt.jinja"
BUILTIN_TASK_MESSAGE = "multi_surface/task_message.jinja"


class MultiSurfaceOptimizer(BaseAgenticOptimizer):
    """Edit prompt, skills and tool descriptions together, one finding per surface.

    Example:
        formula = MultiSurfaceFormula(
            system_prompt=SystemPromptFormula(system_prompt=PROMPT),
            skills=SkillLibraryFormula(skill_dir="./skills"),
            tool_descriptions=ToolDescriptionFormula.from_yaml("./tool_descriptions.yaml"),
        )
        optimizer = MultiSurfaceOptimizer(formula, output_folder="./runs")
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)
        optimizer.step()
        formula.get_tunable_params()   # prompt, skill_dir, tool_descriptions overrides

    The templates receive exactly these variables: ``traces_folder``,
    ``trace_count``, ``census``, ``objective_report``, ``prompt_path``,
    ``prompt_chars``, ``prompt_body``, ``existing_skills``, ``agent_tools``,
    ``tool_desc_path``, ``tool_desc_body``, ``output_folder``.

    Args:
        formula: A ``MultiSurfaceFormula``.
        output_folder: Root under which each step writes its own ``step_NNNN/``.
        system_prompt_template: Reflector persona. Defaults to the built-in.
        task_message_template: Task message. Defaults to the built-in.
        objective_weights: ``{term: weight}`` naming the scores that make up the
            objective; each non-``TaskSuccessScore`` term must be present in
            ``Reward.metadata["scores"]`` for every sampled reward. ``None`` means
            the single term ``TaskSuccessScore = Reward.reward``.
        objective_definitions: ``{term: what it measures}``, shown to the agent next
            to each configured term. The user's wording wins; a term with a built-in
            evaluator name falls back to that evaluator's own definition; a term with
            neither is shown as undefined and logged.
        agent_tools: The agent's toolset, for the census and the task prompt.
            Defaults to the tool-description formula's base names.
        window_size, pin_first, compression_threshold: Conversation-manager
            settings for the reflector. The defaults keep the task message pinned
            and size the window for long shell/editor sessions; pass ``None`` for
            any of them to take Strands' default for that knob.
        retry_attempts: Whole-agent retries on a transient Bedrock error.
        retry_backoff_s: Base back-off between attempts (multiplied by attempt).
        **kwargs: Forwarded to ``BaseAgenticOptimizer``.
    """

    def __init__(
        self,
        formula: MultiSurfaceFormula,
        output_folder: str,
        system_prompt_template: "str | Template | None" = None,
        task_message_template: "str | Template | None" = None,
        objective_weights: Optional[dict] = None,
        objective_definitions: Optional[dict[str, str]] = None,
        agent_tools: Optional[list[str]] = None,
        window_size: Optional[int] = 120,
        pin_first: Optional[int] = 2,
        compression_threshold: Optional[float] = 0.7,
        retry_attempts: int = 3,
        retry_backoff_s: float = 20.0,
        **kwargs,
    ):
        if not isinstance(formula, MultiSurfaceFormula):
            raise TypeError(
                "MultiSurfaceOptimizer requires a MultiSurfaceFormula (prompt + skills + "
                f"tool descriptions); got {type(formula).__name__}."
            )
        # The base suffix instructs the agent to call submit_optimized_params, a
        # tool this optimizer does not provide: it writes files instead.
        explicit_suffix = kwargs.pop("system_prompt_suffix", None)
        super().__init__(formula, **kwargs)
        self.system_prompt_suffix = explicit_suffix if explicit_suffix is not None else ""
        self.model_config.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

        self.output_folder = output_folder
        self.objective_weights = dict(objective_weights) if objective_weights else None
        self._weights = normalize_weights(objective_weights)
        self.objective_definitions = dict(objective_definitions or {})
        self.agent_tools = list(agent_tools) if agent_tools else None
        self.window_size = window_size
        self.pin_first = pin_first
        self.compression_threshold = compression_threshold
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_s = float(retry_backoff_s)

        self._system_prompt_template = self._as_template(
            system_prompt_template, BUILTIN_SYSTEM_PROMPT
        )
        self._task_message_template = self._as_template(task_message_template, BUILTIN_TASK_MESSAGE)

        self._current_step_dir: Optional[str] = None
        self.last_step_dir: Optional[str] = None
        self.last_applied: list[str] = []
        self.last_dropped: dict[str, list[str]] = {}
        self.last_findings: Optional[dict] = None

        logger.info(
            "Initialized MultiSurfaceOptimizer (model=%s, n_sample_traces=%s, "
            "output_folder=%s, objective=%s)",
            self.model_config.get("model_id", "default"),
            self.n_sample_traces,
            output_folder,
            self._weights,
        )

    @staticmethod
    def _as_template(source, builtin_path: str):
        if source is None:
            return load_builtin_template(builtin_path)
        if isinstance(source, str):
            return create_template(source)
        return source

    # ── agent construction ───────────────────────────────────────────────────
    def _get_tools(self, submit_optimized_params) -> list:
        """``shell`` and ``editor``; no submit tool, this agent writes files."""
        from strands_tools import shell

        try:
            from strands_tools import editor
        except ImportError as e:  # pragma: no cover - depends on the installed extras
            raise ImportError(
                "MultiSurfaceOptimizer needs the `editor` tool from strands-agents-tools. "
                "Install or upgrade strands-agents-tools."
            ) from e
        return [shell, editor]

    def _conversation_manager(self):
        """Sliding window sized for a long shell/editor session, task message pinned.

        Strands' default (window 40, nothing pinned) trims oldest-first, and the
        task message is ``messages[0]``. Knobs this strands version lacks are
        dropped one at a time with a warning rather than failing the run.
        """
        from strands.agent.conversation_manager import SlidingWindowConversationManager

        kwargs: dict = {}
        if self.window_size is not None:
            kwargs["window_size"] = self.window_size
        if self.pin_first is not None:
            kwargs["pin_first"] = self.pin_first
        if self.compression_threshold is not None:
            kwargs["proactive_compression"] = {"compression_threshold": self.compression_threshold}
        while True:
            try:
                return SlidingWindowConversationManager(**kwargs)
            except TypeError:
                for k in ("proactive_compression", "pin_first"):
                    if k in kwargs:
                        del kwargs[k]
                        logger.warning(
                            "this strands version's SlidingWindowConversationManager lacks "
                            "%r; dropped it",
                            k,
                        )
                        break
                else:
                    raise

    # ── the step ─────────────────────────────────────────────────────────────
    def step(self) -> None:
        """Run the reflector on a sample of the accumulated rollouts and apply its output."""
        if not self._rollouts:
            logger.warning("No rollouts accumulated, skipping step")
            return

        if len(self._rewards) != len(self._rollouts):
            # Not optional here, weights or no weights: the objective table, the
            # census ordering and the trace `objective` block are all built from
            # rewards, and a fabricated 0.0 would present unscored episodes as
            # failures. Refuse before sampling so nothing is paid for.
            raise ValueError(
                f"{len(self._rollouts)} rollout(s) but {len(self._rewards)} reward(s); "
                "MultiSurfaceOptimizer needs a Reward for every rollout. Call "
                "add_rewards() with one Reward per rollout before step()."
            )
        # Fail on a missing objective term BEFORE paying for the agent -- and before
        # sampling, which classifies episodes by the same totals.
        for r in self._rewards:
            objective_terms(r, self._weights)

        step_dir = self._allocate_step_dir()
        self._current_step_dir = step_dir
        try:
            indices = self._sample_traces()
            rewards = [self._rewards[i] for i in indices]
            names = [f"{self._trace_filename(self._rollouts[i], i)}.json" for i in indices]

            traces_folder = self._write_traces_to_temp(indices)
            self._write_current(step_dir)
            system_prompt, task_message = self._render(step_dir, traces_folder, rewards, names)
            self._run_agent(step_dir, system_prompt, task_message)

            params, applied, dropped, attempted = self._collect(step_dir)
            findings, findings_problem = renderers.read_findings(step_dir)

            if attempted and not applied:
                raise RuntimeError(
                    "the reflector wrote artifacts for "
                    f"{attempted} but none is usable, so this step would apply nothing "
                    "while looking like a change:\n  "
                    + "\n  ".join(f"{s}: {'; '.join(p)}" for s, p in dropped.items())
                    + f"\n  Artifacts kept at {step_dir}"
                )
            if not attempted:
                if findings is None:
                    raise RuntimeError(
                        f"the reflector wrote neither an artifact nor a usable "
                        f"{renderers.FINDINGS_FILE} ({findings_problem}), so nothing "
                        "records what it examined: an empty pass and a crashed agent are "
                        f"indistinguishable. Output kept at {step_dir}"
                    )
                logger.info(
                    "Step %d: no surface edited; %d finding(s) recorded in %s",
                    self._step_count + 1,
                    len(findings["findings"]),
                    renderers.FINDINGS_FILE,
                )
            else:
                if findings is None:
                    logger.warning("%s -- applying the artifacts anyway", findings_problem)
                for surface, problems in dropped.items():
                    logger.warning("dropping %s: %s", surface, "; ".join(problems))

            # Apply and materialize as one unit. A failure part-way (a strict formula
            # rejecting an edit, an OSError while copying skills) must not leave the
            # prompt changed while the step is recorded as failed.
            snap = self.formula.snapshot()
            try:
                if attempted:
                    self.formula.update_params(params)
                if self.formula.skills.members:
                    self.formula.skills.materialize(os.path.join(step_dir, SKILL_SET_DIR))
            except Exception:
                self.formula.restore(snap)
                raise

            self._step_count += 1
            self._prompt_history.append(
                {
                    "step_dir": step_dir,
                    "applied": list(applied),
                    "dropped": sorted(dropped),
                    "skills": self.formula.skills.skill_names,
                    "tool_overrides": sorted(self.formula.tool_descriptions.overrides),
                }
            )
            self.last_step_dir = step_dir
            self.last_applied = list(applied)
            self.last_dropped = dict(dropped)
            self.last_findings = findings
            logger.info(
                "Step %d: applied %s; dropped %s; %d finding(s)",
                self._step_count,
                applied or "nothing",
                sorted(dropped) or "nothing",
                len(findings["findings"]) if findings else 0,
            )
        except BaseException as e:
            self._mark_failed(step_dir, e)
            raise
        finally:
            self._cleanup_temp()
            self._current_step_dir = None

    # ── step directory ───────────────────────────────────────────────────────
    def _allocate_step_dir(self) -> str:
        """The lowest ``step_NNNN`` not on disk, starting from the next step number."""
        os.makedirs(self.output_folder, exist_ok=True)
        n = self._step_count + 1
        while os.path.exists(os.path.join(self.output_folder, f"{STEP_PREFIX}{n:04d}")):
            n += 1
        path = os.path.join(self.output_folder, f"{STEP_PREFIX}{n:04d}")
        os.makedirs(path)
        return path

    @staticmethod
    def _mark_failed(step_dir: str, error: BaseException) -> None:
        try:
            with open(os.path.join(step_dir, FAILED_FILE), "w") as f:
                f.write(f"{type(error).__name__}: {error}\n")
        except OSError:  # pragma: no cover - best effort
            pass

    @staticmethod
    def _log_attempt(step_dir: str, attempt: int, note: str) -> None:
        try:
            with open(os.path.join(step_dir, ATTEMPTS_FILE), "a") as f:
                f.write(f"attempt {attempt}: {note}\n")
        except OSError:  # pragma: no cover - best effort
            pass

    @staticmethod
    def _clear_agent_paths(step_dir: str) -> None:
        for rel in AGENT_OWNED_PATHS:
            p = os.path.join(step_dir, rel)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)

    # ── inputs ───────────────────────────────────────────────────────────────
    def _objective_total(self, reward: Reward) -> float:
        return weighted_total(objective_terms(reward, self._weights), self._weights)

    def _sample_reward(self, index: int) -> float:
        """Stratify on the configured objective total, not the raw Reward.reward.

        The task prompt tells the agent that the total weighted reward orders and
        samples episodes; with a weighted objective the two can disagree, and the
        sample must follow what the agent is told.
        """
        return self._objective_total(self._rewards[index])

    def _sample_threshold(self, all_indices: list[int]) -> float:
        """``success_threshold``, or the pool median when it separates nothing.

        A continuous objective (a weighted mean of judge scores) can put every
        episode on one side of 0.5. Splitting at the median then keeps a low/high
        contrast in the sample; when the objective is constant there is none to
        keep, and that is logged.
        """
        import statistics

        totals = [self._sample_reward(i) for i in all_indices]
        if not totals:
            return self.success_threshold
        high = [t for t in totals if t >= self.success_threshold]
        if high and len(high) < len(totals):
            return self.success_threshold
        mid = statistics.median(totals)
        lo = [t for t in totals if t < mid]
        hi = [t for t in totals if t >= mid]
        if lo and hi:
            logger.info(
                "objective does not separate at %.2f (%d higher / %d lower); splitting "
                "at the pool median %.3f instead -> %d higher / %d lower",
                self.success_threshold,
                len(high),
                len(totals) - len(high),
                mid,
                len(hi),
                len(lo),
            )
            return mid
        logger.warning(
            "objective is constant at %.3f across all %d rollouts -- no contrast is "
            "available in this pool",
            mid,
            len(totals),
        )
        return self.success_threshold

    def _trace_extra(self, rollout, reward, index: int) -> dict:
        """The ``objective`` block, and ``reward`` as the configured total."""
        if reward is None:  # pragma: no cover - step() refuses missing rewards first
            raise RuntimeError(f"rollout {index} has no Reward")
        terms = objective_terms(reward, self._weights)
        total = weighted_total(terms, self._weights)
        return {
            "reward": total,
            "objective": {"reward": total, "terms": terms, "weights": dict(self._weights)},
        }

    def _resolved_agent_tools(self) -> list[str]:
        base_names = sorted(self.formula.tool_descriptions.base)
        if self.agent_tools:
            if base_names and set(self.agent_tools) != set(base_names):
                logger.warning(
                    "agent_tools %s disagree with the tool-description base %s "
                    "(difference: %s); using agent_tools",
                    sorted(self.agent_tools),
                    base_names,
                    sorted(set(self.agent_tools) ^ set(base_names)),
                )
            return list(self.agent_tools)
        return base_names

    def _write_current(self, step_dir: str) -> None:
        """Write the prompt and tool descriptions the agent starts from.

        The task prompt tells the agent to COPY the prompt file and edit the copy,
        so the shipped text is byte-identical to the deployed one apart from its
        edits. That needs the file to exist, in the same YAML shape it writes back.
        """
        cur = os.path.join(step_dir, CURRENT_DIR)
        os.makedirs(cur, exist_ok=True)
        prompt_text = self.formula.system_prompt.system_prompt or ""
        with open(os.path.join(cur, CURRENT_PROMPT_FILE), "w") as f:
            f.write(_yaml_block("system_prompt", prompt_text))
        with open(os.path.join(cur, CURRENT_TOOLS_FILE), "w") as f:
            f.write(self.formula.tool_descriptions.render_effective_yaml())

    def _render(
        self, step_dir: str, traces_folder: str, rewards: list[Reward], names: list[str]
    ) -> tuple[str, str]:
        cur = os.path.join(step_dir, CURRENT_DIR)
        prompt_text = self.formula.system_prompt.system_prompt or ""
        summaries = renderers.load_trace_summaries(traces_folder)
        agent_tools = self._resolved_agent_tools()
        effective_tools = self.formula.tool_descriptions.effective()
        template_vars = {
            "traces_folder": os.path.abspath(traces_folder),
            "trace_count": len(summaries),
            "census": renderers.render_census(summaries, agent_tools, self.success_threshold),
            "objective_report": renderers.render_objective_report(
                rewards, self._weights, names, self.objective_definitions
            ),
            "prompt_path": os.path.abspath(os.path.join(cur, CURRENT_PROMPT_FILE)),
            "prompt_chars": len(prompt_text),
            "prompt_body": prompt_text or "(the current prompt is empty)",
            "existing_skills": renderers.render_deployed_skills(self.formula.skills.members),
            "agent_tools": ", ".join(agent_tools) or "(none declared)",
            "tool_desc_path": os.path.abspath(os.path.join(cur, CURRENT_TOOLS_FILE)),
            "tool_desc_body": (
                self.formula.tool_descriptions.render_effective_yaml()
                if effective_tools
                else "(no tool descriptions on file)"
            ),
            "output_folder": os.path.abspath(step_dir),
        }
        system_prompt = self._system_prompt_template.render(**template_vars)
        task_message = self._task_message_template.render(**template_vars)
        # Kept for inspection: what the reflector was actually shown.
        with open(os.path.join(cur, "rendered_system_prompt.txt"), "w") as f:
            f.write(system_prompt)
        with open(os.path.join(cur, "rendered_task_message.txt"), "w") as f:
            f.write(task_message)
        return system_prompt, task_message

    # ── the agent ────────────────────────────────────────────────────────────
    def _run_agent(self, step_dir: str, system_prompt: str, task_message: str) -> None:
        """Run the reflector, rebuilding and re-running it on a transient failure.

        A failed turn leaves the agent's history holding a partial exchange that
        cannot be resumed, so each attempt is a fresh agent. Agent-owned paths in
        the step directory are cleared first, so a half-written artifact from the
        failed attempt cannot be mistaken for this one's output.
        """
        for attempt in range(1, self.retry_attempts + 1):
            if attempt > 1:
                self._clear_agent_paths(step_dir)
            try:
                agent = self._create_agent(system_prompt)
                self._invoke_agent(agent, task_message)
                self._log_attempt(step_dir, attempt, "ok")
                return
            except Exception as e:  # noqa: BLE001 - classified below
                msg = str(e)
                transient = any(t in msg for t in TRANSIENT_ERRORS)
                self._log_attempt(
                    step_dir, attempt, f"{'transient' if transient else 'fatal'}: {msg[:200]}"
                )
                if not transient or attempt == self.retry_attempts:
                    raise
                wait = self.retry_backoff_s * attempt
                logger.warning(
                    "transient failure on attempt %d/%d (%s); rebuilding the agent and "
                    "retrying in %.0fs",
                    attempt,
                    self.retry_attempts,
                    msg[:120],
                    wait,
                )
                if wait > 0:
                    time.sleep(wait)

    # ── outputs ──────────────────────────────────────────────────────────────
    def _collect(self, step_dir: str) -> tuple[dict, list[str], dict[str, list[str]], list[str]]:
        """Read the agent's artifacts into one params dict.

        Returns ``(params, applied, dropped, attempted)``. Presence on disk decides
        ``attempted``; structural validation decides ``applied`` vs ``dropped``.
        """
        import yaml

        params: dict = {}
        applied: list[str] = []
        dropped: dict[str, list[str]] = {}
        attempted: list[str] = []
        for surface in renderers.SURFACES:
            if not renderers.surface_attempted(step_dir, surface):
                continue
            attempted.append(surface)
            problems = renderers.validate_surface(step_dir, surface)
            if problems:
                dropped[surface] = problems
                continue
            if surface == "system_prompt":
                with open(os.path.join(step_dir, renderers.PROMPT_FILE)) as f:
                    params["system_prompt"] = str(yaml.safe_load(f)["system_prompt"])
            elif surface == "skills":
                params["decisions_dir"] = os.path.join(step_dir, renderers.SKILLS_DIR)
            else:
                with open(os.path.join(step_dir, renderers.TOOL_DESC_FILE)) as f:
                    edits = dict(yaml.safe_load(f)["tool_descriptions"])
                # Structure is fine; ask the formula whether any entry would LAND.
                # A file whose every entry names an unknown tool or overruns the
                # limit changes nothing, and must not be recorded as applied.
                problems = self.formula.tool_descriptions.validate_edits(edits)
                rejected = {k: v for k, v in problems.items() if v}
                if len(rejected) == len(edits):
                    dropped[surface] = [f"{k}: {v}" for k, v in rejected.items()] or ["no entries"]
                    continue
                if rejected:
                    logger.warning(
                        "tool_descriptions: %d of %d entr%s will be rejected: %s",
                        len(rejected),
                        len(edits),
                        "y" if len(rejected) == 1 else "ies",
                        "; ".join(f"{k}: {v}" for k, v in rejected.items()),
                    )
                params[TOOL_DESCRIPTIONS_KEY] = edits
            applied.append(surface)
        return params, applied, dropped, attempted

    # ── checkpointing ────────────────────────────────────────────────────────
    _SETTINGS = (
        "output_folder",
        "objective_weights",
        "objective_definitions",
        "agent_tools",
        "window_size",
        "pin_first",
        "compression_threshold",
        "retry_attempts",
        "retry_backoff_s",
    )

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({k: getattr(self, k) for k in self._SETTINGS})
        state.update(
            {
                "last_step_dir": self.last_step_dir,
                "last_applied": list(self.last_applied),
                "last_dropped": dict(self.last_dropped),
                "formula_params": self.formula.get_tunable_params(),
            }
        )
        return state

    def load_state(self, state: dict) -> None:
        """Restore from :meth:`get_state`. Formula state is REPLACED, not merged.

        Saved tool overrides go through ``replace_overrides`` so an override the
        checkpoint does not contain is removed; the skill set is reloaded from the
        saved ``skill_dir`` (empty means cold); the prompt is set outright.
        """
        super().load_state(state)
        for k in self._SETTINGS:
            if k in state:
                setattr(self, k, state[k])
        self._weights = normalize_weights(self.objective_weights)
        if "last_step_dir" in state:
            self.last_step_dir = state["last_step_dir"]
        if "last_applied" in state:
            self.last_applied = list(state["last_applied"] or [])
        if "last_dropped" in state:
            self.last_dropped = dict(state["last_dropped"] or {})
        params = state.get("formula_params")
        if params is not None:
            if "system_prompt" in params:
                self.formula.system_prompt.update_params({"system_prompt": params["system_prompt"]})
            if "skill_dir" in params:
                self.formula.skills.update_params({"skill_dir": params["skill_dir"] or ""})
            if TOOL_DESCRIPTIONS_KEY in params:
                self.formula.tool_descriptions.replace_overrides(params[TOOL_DESCRIPTIONS_KEY])


def _yaml_block(key: str, text: str) -> str:
    """``key: |`` followed by the text as a literal block that round-trips EXACTLY.

    The chomping indicator is chosen from the text's own trailing newlines: ``|-``
    for none, ``|`` for exactly one, ``|+`` for more, so ``yaml.safe_load`` returns
    the prompt byte for byte. That matters because the agent copies this file and
    the shipped prompt must differ from the deployed one only by its edits. If the
    text cannot be expressed as a literal block at all, fall back to the dumper's
    own quoting, which is exact but less readable.
    """
    import yaml

    stripped = text.rstrip("\n")
    trailing = len(text) - len(stripped)
    indicator = "|-" if trailing == 0 else ("|" if trailing == 1 else "|+")
    body = "".join(f"  {ln}\n" if ln else "\n" for ln in stripped.split("\n"))
    if trailing > 1:
        body += "\n" * (trailing - 1)
    block = f"{key}: {indicator}\n{body}"
    try:
        if yaml.safe_load(block).get(key) == text:
            return block
    except yaml.YAMLError:
        pass
    return yaml.safe_dump({key: text}, allow_unicode=True, width=100000, sort_keys=False)
