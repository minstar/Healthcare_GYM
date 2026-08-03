"""Reflexion-style self-critique-and-retry driver for the Healthcare AI GYM.

WHAT THIS IS
------------
A *training-free* baseline: it wraps the existing :class:`AgentRunner` through
its public surface (``load_model`` / ``run_task`` / ``run_all_tasks`` /
``generate`` / ``_compute_qa_accuracy`` / ``_compute_action_score``) and adds a
verbal self-reflection retry loop on top.  ``agent_runner.py`` is NOT modified
and NOT subclassed: reflections are injected by wrapping the *environment*
(:class:`ReflexionEnvProxy`), which prepends a memory block to the observation
that ``run_task`` turns into the first user message.

It exists to settle one question directly: *do the reported gains come from the
method, or from spending more inference on better prompting?*  This
module upper-bounds what prompting alone buys in this GYM, at zero training
cost, and it measures exactly how many extra model calls that upper bound
consumed.  It is deliberately built to be strong, not to be a strawman.

METHOD / CITATION
-----------------
Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik
Narasimhan, Shunyu Yao. "Reflexion: Language Agents with Verbal Reinforcement
Learning." NeurIPS 2023.  arXiv:2303.11366.  Official code:
https://github.com/noahshinn/reflexion

This is a **reimplementation, not a port**.  The official repository is pinned
to ``OPENAI_API_KEY``, a 2023-era ``langchain``/``AnyOpenAILLM`` stack and
``tiktoken``-based truncation, and its agents are hardwired to HotPotQA /
ALFWorld / WebShop action grammars.  None of that runs against a locally served
Qwen/Solar checkpoint driving 171 ``@is_tool`` medical tools.  What is
reproduced faithfully is the part that matters for a legible ablation: the
four-level strategy ladder (``ReflexionStrategy``), the reflection/last-trial
prompt headers, the "reflect only after a failed trial" control flow, and the
per-strategy memory update rules (including the reference implementation's
choice to *reset* rather than accumulate reflections in the
``LAST_ATTEMPT_AND_REFLEXION`` branch — see ``_build_memory``).
Every deviation is listed in ``DEVIATIONS_FROM_PUBLISHED_REFLEXION`` below.

SAME-MODEL CONSTRAINT (enforced, not merely documented)
-------------------------------------------------------
The reflection and the retry are produced by the *same* model, through the same
backend, as the policy under test.  A stronger external critic would silently
turn this baseline into distillation from that critic and make the comparison
uninterpretable.  There is no separate LLM handle in this module: reflections go
through ``self.runner.generate(...)``.  On top of that structural guarantee,
:meth:`ReflexionRunner._check_same_model` re-validates identity *and*
``model_name_or_path`` / ``backend`` before every reflection call and raises
:class:`SameModelViolation` on drift.

FAILURE SIGNAL, AND WHAT HAPPENS WHEN THERE IS NONE
----------------------------------------------------
"Did this attempt fail?" is answered with the environment's own scoring, in
this priority order (see :meth:`ReflexionRunner._score_attempt`):

1. ``qa_accuracy``   — task carries ``answer``/``correct_answer``; uses
   ``AgentRunner._compute_qa_accuracy`` (threshold ``qa_success_threshold``).
2. ``action_score``  — task carries a non-empty ``evaluation_criteria.actions``;
   uses ``AgentRunner._compute_action_score`` (threshold
   ``action_success_threshold``).  The non-empty check is load-bearing:
   ``_compute_action_score`` returns **1.0** when no actions are expected, so
   trusting it blindly would mark every criteria-less task "solved" and silently
   collapse the whole ladder to strategy ``NONE``.
3. ``composite_reward`` — only if ``reward_success_threshold`` is explicitly set
   (default ``None`` = off, because any threshold on a shaped reward is a
   judgement call that should be made by the experimenter, not defaulted).
4. ``ungraded``      — none of the above is available.

Case 4 does **not** silently behave like ``NONE``.  It follows
``ReflexionConfig.ungraded_policy``, one of:

* ``RETRY_ALL`` (default) — treat the attempt as failed and use the full attempt
  budget, reflecting on an unverified trajectory.  ``AttemptRecord.success``
  stays ``None`` and ``assumed_failure=True``, so ungraded tasks can be split
  out of any reported number.  This is the honest upper bound: maximum prompting
  effort, maximum measured cost.
* ``SELF_JUDGE`` — the same model (never a stronger one) judges its own
  trajectory; the verdict becomes the failure signal and each judgement is
  counted as an extra model call under ``model_calls_judge``.
* ``STOP_AFTER_FIRST`` — one attempt only.  This *is* equivalent to ``NONE``,
  which is why it is recorded loudly: ``degraded_to_single_attempt=True`` on the
  task result and ``n_degraded_to_single_attempt`` in the run summary.

COST ACCOUNTING
---------------
A prompting baseline that wins by spending 5x the inference budget is a finding,
not a footnote.  Every ``runner.generate`` call made during a task is counted by
phase (``attempt`` / ``reflection`` / ``judge``) via an instance-level meter that
is installed and removed around the run (:func:`_meter_model_calls`; it patches
the *instance* attribute only, never the class, and restores it in ``finally``).
``ReflexionTaskResult`` reports ``model_calls_total``,
``baseline_model_calls`` (= attempt 1 alone, i.e. what plain ``AgentRunner``
would have spent), ``extra_model_calls`` and ``call_overhead_ratio``.

USAGE
-----
    from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
    from bioagents.evaluation.reflexion_runner import (
        ReflexionRunner, ReflexionConfig, ReflexionStrategy,
    )

    runner = AgentRunner(RunConfig(model_name_or_path=..., backend="sglang",
                                   domain="medical_qa"))
    runner.load_model()
    rx = ReflexionRunner(
        runner,
        ReflexionConfig(strategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
                        max_attempts=3, output_dir="logs/reflexion/medqa_larx"),
    )
    results = rx.run_all_tasks()
    print(rx.summarize(results))

Run the ladder by sweeping ``strategy`` over the four levels with everything
else held fixed.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger

__all__ = [
    "ReflexionStrategy",
    "SuccessSignal",
    "UngradedPolicy",
    "ReflexionConfig",
    "AttemptRecord",
    "ReflexionTaskResult",
    "ReflexionRunner",
    "ReflexionEnvProxy",
    "SameModelViolation",
    "GoldAnswerLeak",
    "render_scratchpad",
    "truncate_scratchpad",
    "format_reflections",
    "format_last_attempt",
    "REFLECTION_HEADER",
    "REFLECTION_AFTER_LAST_TRIAL_HEADER",
    "LAST_TRIAL_HEADER",
    "REFLECT_INSTRUCTION",
    "SELF_JUDGE_INSTRUCTION",
    "MEMORY_BLOCK_OPEN",
    "MEMORY_BLOCK_CLOSE",
    "DEVIATIONS_FROM_PUBLISHED_REFLEXION",
]


# ══════════════════════════════════════════════════════════════════════
#  Prompt fragments
#
#  The three headers are verbatim from Shinn et al.'s
#  ``hotpotqa_runs/prompts.py`` so that the reflection scaffold a reader of that
#  paper recognises is the one actually used.  REFLECT_INSTRUCTION is re-worded for
#  a tool-using clinical agent (the original names the HotPotQA Docstore API
#  and the ``Finish[...]`` grammar, neither of which exists here).
# ══════════════════════════════════════════════════════════════════════

REFLECTION_HEADER = (
    "You have attempted to answer following question before and failed. The "
    "following reflection(s) give a plan to avoid failing to answer the question "
    "in the same way you did previously. Use them to improve your strategy of "
    "correctly answering the given question.\n"
)
REFLECTION_AFTER_LAST_TRIAL_HEADER = (
    "The following reflection(s) give a plan to avoid failing to answer the "
    "question in the same way you did previously. Use them to improve your "
    "strategy of correctly answering the given question.\n"
)
LAST_TRIAL_HEADER = (
    "You have attempted to answer the following question before and failed. "
    "Below is the last trial you attempted to answer the question.\n"
)

REFLECT_INSTRUCTION = """You are an advanced clinical reasoning agent that can improve based on self reflection. You will be given a previous trial in which you acted in a medical environment with access to clinical tools, and a task you had to solve. You were unsuccessful, either because you submitted a wrong answer, called the wrong tools, or used up your allotted number of turns. In a few sentences, diagnose a possible reason for the failure and devise a new, concise, high-level plan that aims to mitigate the same failure. Be specific about which tools to call and in what order. Do not restate the task and do not guess an answer here — write only the diagnosis and the plan. Use complete sentences.

Previous trial:
Task: {question}
{scratchpad}

Reflection:"""

SELF_JUDGE_INSTRUCTION = """You are reviewing your own previous attempt at a clinical task. No answer key is available. Judge strictly whether the attempt actually solved the task: did it gather the necessary evidence with the appropriate tools and commit to a complete, well-supported answer? Reply with exactly one word, SOLVED or UNSOLVED, followed by one short sentence of justification.

Task: {question}
{scratchpad}

Verdict:"""

MEMORY_BLOCK_OPEN = "=== REFLEXION MEMORY (your own previous attempt(s) at THIS task) ==="
MEMORY_BLOCK_CLOSE = "=== END REFLEXION MEMORY ==="


DEVIATIONS_FROM_PUBLISHED_REFLEXION = [
    "Reimplementation, not a port: the official repo is hardwired to "
    "OPENAI_API_KEY + langchain AnyOpenAILLM + gpt-3.5-turbo and to the "
    "HotPotQA/ALFWorld/WebShop action grammars; this driver runs against the "
    "locally served model under test through AgentRunner.generate.",
    "Episode granularity: the reference agents re-run a hand-rolled "
    "Thought/Action/Observation loop; here each attempt is one unmodified "
    "AgentRunner.run_task episode over the GYM's real 171-tool surface, so "
    "turn limits, tool parsing, repetition nudges and scoring are the paper's, "
    "not this module's.",
    "Memory injection point: the reference formats reflections into a "
    "PromptTemplate slot; here the memory block is prepended to the "
    "environment observation (the first user message) by ReflexionEnvProxy, "
    "because run_task builds its own system prompt and must stay unmodified.",
    "No few-shot reflection exemplars: the reference passes REFLECTIONS/"
    "COT_REFLECT few-shots into the reflect prompt. Those are HotPotQA-specific "
    "and would inject an out-of-domain answer format, so the reflect prompt is "
    "zero-shot. This is the main capability-relevant deviation; if the ladder "
    "underperforms, hand-written clinical reflection exemplars are the first "
    "thing to add.",
    "Scratchpad truncation is char-budgeted (largest observations shrunk "
    "first, mirroring truncate_scratchpad's strategy) instead of tiktoken "
    "token-budgeted, to avoid a tokenizer dependency that does not match the "
    "model under test.",
    "ReflexionStrategy.NONE runs a single attempt by default "
    "(retry_on_none=False), i.e. exactly the un-augmented AgentRunner baseline. "
    "The reference loop re-runs every strategy for n trials, so NONE there is "
    "budget-matched independent resampling; set retry_on_none=True to reproduce "
    "that (only meaningful at temperature > 0).",
    "Reflection length: the reference caps the reflection LLM at 250 tokens; "
    "here reflection_max_new_tokens (default 256) temporarily overrides "
    "runner.config.max_new_tokens for the reflection call only, restoring it "
    "afterwards.",
    "format_step's newline-stripping is not applied: reflections keep their "
    "line structure because this environment's prompts are chat-formatted, not "
    "a single flat scratchpad string.",
    "Failure signal: the reference calls is_correct() (EM against the gold key) "
    "for HotPotQA and the env reward for ALFWorld/WebShop. Here the signal is "
    "resolved per task in _score_attempt, with an explicit, documented "
    "degradation path (UngradedPolicy) for tasks that carry no gold signal — "
    "the reference has no such case.",
    "ORACLE-IN-THE-LOOP (inherited, not introduced): like published Reflexion, "
    "the decision to retry uses ground truth, and the memory block therefore "
    "leaks ~1 bit per attempt ('your previous attempt was judged incorrect') "
    "to the retry. This makes the ladder an upper bound rather than a "
    "deployable system, and any comparison against a method that never sees "
    "the label must say so. UngradedPolicy.SELF_JUDGE is the oracle-free "
    "variant.",
    "The gold answer itself is never placed in the reflection or retry prompt; "
    "_assert_no_gold_leak enforces this for free-text answers.",
]


# ══════════════════════════════════════════════════════════════════════
#  Errors
# ══════════════════════════════════════════════════════════════════════


class SameModelViolation(RuntimeError):
    """Raised when reflection would be produced by anything but the model under test."""


class GoldAnswerLeak(RuntimeError):
    """Raised when a gold answer would be exposed to the reflection/retry prompt."""


# ══════════════════════════════════════════════════════════════════════
#  Strategy ladder (Shinn et al., 2023 — hotpotqa_runs/agents.py:23)
# ══════════════════════════════════════════════════════════════════════


class ReflexionStrategy(Enum):
    """The four-level ladder, values verbatim from the reference implementation.

    NONE
        No reflection.  Single attempt (or, with ``retry_on_none=True``,
        independent resamples with no memory).  This is the control rung: any
        gain above it is what the ladder actually bought.
    LAST_ATTEMPT
        The previous trajectory itself is placed in context.  No extra model
        call — this rung isolates "more context" from "verbal self-criticism".
    REFLEXION
        The model writes a verbal self-reflection on the failed trajectory;
        reflections *accumulate* across attempts.
    LAST_ATTEMPT_AND_REFLEXION
        Previous trajectory *and* a fresh reflection on it.
    """

    NONE = "base"
    LAST_ATTEMPT = "last_trial"
    REFLEXION = "reflexion"
    LAST_ATTEMPT_AND_REFLEXION = "last_trial_and_reflexion"

    @property
    def uses_verbal_reflection(self) -> bool:
        """True iff this rung spends an extra model call to write a reflection."""
        return self in (
            ReflexionStrategy.REFLEXION,
            ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
        )

    @property
    def uses_last_trial(self) -> bool:
        """True iff this rung puts the raw previous trajectory in context."""
        return self in (
            ReflexionStrategy.LAST_ATTEMPT,
            ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
        )

    @classmethod
    def from_str(cls, value: "str | ReflexionStrategy") -> "ReflexionStrategy":
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if text in (member.value, member.name.lower()):
                return member
        raise ValueError(
            f"Unknown reflexion strategy {value!r}. "
            f"Expected one of {[m.value for m in cls]} (or their names)."
        )


class SuccessSignal(str, Enum):
    """Where the pass/fail verdict for an attempt came from."""

    QA_ACCURACY = "qa_accuracy"
    ACTION_SCORE = "action_score"
    COMPOSITE_REWARD = "composite_reward"
    SELF_JUDGE = "self_judge"
    UNGRADED = "ungraded"
    ERROR = "error"


class UngradedPolicy(str, Enum):
    """Defined behaviour when a task carries no eval-time correctness signal."""

    RETRY_ALL = "retry_all"
    SELF_JUDGE = "self_judge"
    STOP_AFTER_FIRST = "stop_after_first"


# ══════════════════════════════════════════════════════════════════════
#  Config & records
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ReflexionConfig:
    """Configuration for a Reflexion ladder run."""

    strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
    max_attempts: int = 3

    # NONE normally means "one shot".  True reproduces the reference loop's
    # budget-matched independent resampling (needs temperature > 0 to differ).
    retry_on_none: bool = False

    # Success thresholds against the GYM's own scorers.
    qa_success_threshold: float = 1.0
    action_success_threshold: float = 1.0
    # None = do NOT use the shaped composite reward as a pass/fail signal.
    reward_success_threshold: Optional[float] = None

    ungraded_policy: UngradedPolicy = UngradedPolicy.RETRY_ALL

    # Scratchpad rendering budgets (chars).
    max_scratchpad_chars: int = 6000
    max_observation_chars: int = 600
    # Cap on the memory block that gets prepended to the observation.
    max_memory_chars: int = 8000

    # Reflection decoding: mirrors the reference's 250-token cap.
    reflection_max_new_tokens: Optional[int] = 256

    # Persistence.
    output_dir: Optional[str] = None
    save_records: bool = True

    def __post_init__(self) -> None:
        self.strategy = ReflexionStrategy.from_str(self.strategy)
        if not isinstance(self.ungraded_policy, UngradedPolicy):
            self.ungraded_policy = UngradedPolicy(str(self.ungraded_policy))
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


@dataclass
class AttemptRecord:
    """One attempt at one task — the unit the rebuttal reports over."""

    task_id: str
    domain: str
    strategy: str
    attempt_idx: int  # 0-based; 0 is the plain, un-augmented AgentRunner run
    success: Optional[bool]  # None == ungraded (never silently False)
    success_signal: str
    score: float
    assumed_failure: bool = False  # ungraded + RETRY_ALL: treated as failed
    qa_accuracy: Optional[float] = None
    action_score: Optional[float] = None
    final_reward: float = 0.0
    total_turns: int = 0
    # Cost.
    model_calls_attempt: int = 0
    model_calls_reflection: int = 0
    model_calls_judge: int = 0
    # Memory that was in context for THIS attempt (produced after the previous).
    memory_injected: bool = False
    memory_chars: int = 0
    memory_preview: str = ""
    # Reflection produced AFTER this attempt, for the next one.
    reflection_text: Optional[str] = None
    reflection_model_id: Optional[str] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReflexionTaskResult:
    """Ladder outcome for a single task."""

    task_id: str
    domain: str
    strategy: str
    model_id: str
    backend: Optional[str] = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)

    success: bool = False
    attempts_used: int = 0
    attempts_to_success: Optional[int] = None  # 1-based; None if never solved
    best_score: float = 0.0
    best_attempt_idx: int = 0
    graded: bool = True
    degraded_to_single_attempt: bool = False
    stop_reason: str = ""

    model_calls_total: int = 0
    model_calls_attempt: int = 0
    model_calls_reflection: int = 0
    model_calls_judge: int = 0
    baseline_model_calls: int = 0  # attempt 0 only == plain AgentRunner cost
    extra_model_calls: int = 0
    call_overhead_ratio: float = 1.0

    total_latency: float = 0.0
    # The underlying AgentRunner TaskResult of the final and best attempts.
    final_task_result: Any = None
    best_task_result: Any = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "strategy": self.strategy,
            "model_id": self.model_id,
            "backend": self.backend,
            "success": self.success,
            "attempts_used": self.attempts_used,
            "attempts_to_success": self.attempts_to_success,
            "best_score": self.best_score,
            "best_attempt_idx": self.best_attempt_idx,
            "graded": self.graded,
            "degraded_to_single_attempt": self.degraded_to_single_attempt,
            "stop_reason": self.stop_reason,
            "model_calls_total": self.model_calls_total,
            "model_calls_attempt": self.model_calls_attempt,
            "model_calls_reflection": self.model_calls_reflection,
            "model_calls_judge": self.model_calls_judge,
            "baseline_model_calls": self.baseline_model_calls,
            "extra_model_calls": self.extra_model_calls,
            "call_overhead_ratio": self.call_overhead_ratio,
            "total_latency": self.total_latency,
            "reflections": list(self.reflections),
            "attempts": [a.to_dict() for a in self.attempts],
        }


# ══════════════════════════════════════════════════════════════════════
#  Trajectory rendering  (Shinn et al. scratchpad analogue)
# ══════════════════════════════════════════════════════════════════════


def truncate_scratchpad(scratchpad: str, max_chars: int = 6000) -> str:
    """Shrink a rendered trajectory to ``max_chars``.

    Mirrors the reference ``truncate_scratchpad``: repeatedly replace the
    *largest* observation line with a placeholder until the budget is met,
    so the reasoning/action structure survives and only bulky tool output is
    lost.  Char-budgeted rather than tiktoken-budgeted (see DEVIATIONS).
    """
    if len(scratchpad) <= max_chars:
        return scratchpad

    lines = scratchpad.split("\n")
    obs_idx = [i for i, ln in enumerate(lines) if ln.startswith("Observation")]
    obs_idx.sort(key=lambda i: len(lines[i]), reverse=True)

    for i in obs_idx:
        if len("\n".join(lines)) <= max_chars:
            break
        head = lines[i].split(":", 1)[0]
        lines[i] = f"{head}: [truncated tool output]"

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 20].rstrip() + "\n... [truncated]"
    return out


def render_scratchpad(
    task_result: Any,
    max_observation_chars: int = 600,
    max_chars: int = 6000,
) -> str:
    """Render an ``AgentRunner.TaskResult`` as a Reflexion-style scratchpad."""
    lines: list[str] = []
    for turn in getattr(task_result, "turns", []) or []:
        idx = getattr(turn, "turn_idx", len(lines))
        call = getattr(turn, "parsed_tool_call", None)
        raw = (getattr(turn, "raw_output", "") or "").strip()
        if call:
            try:
                rendered = json.dumps(call, ensure_ascii=False)
            except Exception:
                rendered = str(call)
            lines.append(f"Action {idx}: {rendered}")
            obs = getattr(turn, "tool_response", None)
            if obs is not None:
                obs = str(obs).replace("\n", " ")
                if len(obs) > max_observation_chars:
                    obs = obs[:max_observation_chars] + " ...[truncated]"
                lines.append(f"Observation {idx}: {obs}")
        else:
            if len(raw) > max_observation_chars * 2:
                raw = raw[: max_observation_chars * 2] + " ...[truncated]"
            label = "Final answer" if getattr(turn, "is_final_answer", False) else "Thought"
            lines.append(f"{label} {idx}: {raw}")

    err = getattr(task_result, "error", None)
    if err:
        lines.append(f"Error: {str(err).splitlines()[0][:300]}")
    if not lines:
        lines.append("(the previous attempt produced no usable turns)")

    return truncate_scratchpad("\n".join(lines), max_chars=max_chars)


def format_reflections(
    reflections: list[str],
    header: str = REFLECTION_HEADER,
) -> str:
    """Reference ``format_reflections``."""
    if not reflections:
        return ""
    return header + "Reflections:\n- " + "\n- ".join(r.strip() for r in reflections)


def format_last_attempt(
    question: str,
    scratchpad: str,
    header: str = LAST_TRIAL_HEADER,
) -> str:
    """Reference ``format_last_attempt``."""
    return (
        header
        + f"Task: {question}\n"
        + scratchpad.strip("\n").strip()
        + "\n(END PREVIOUS TRIAL)\n"
    )


# ══════════════════════════════════════════════════════════════════════
#  Environment proxy — the only injection point
# ══════════════════════════════════════════════════════════════════════


class ReflexionEnvProxy:
    """Transparent wrapper that prepends a memory block to ``reset``'s observation.

    ``AgentRunner.run_task`` does ``obs, info = env.reset(...)`` and makes ``obs``
    the first user message.  Wrapping the env is therefore enough to place
    Reflexion memory in context without touching ``agent_runner.py``.  Every
    other attribute (``step``, ``get_trajectory``, ``_tool_call_log``,
    ``_task_map`` ...) is delegated untouched.
    """

    def __init__(self, env: Any):
        self._env = env
        self._memory_text: str = ""
        self.last_observation: str = ""

    # -- memory control -------------------------------------------------
    def set_memory(self, memory_text: str) -> None:
        self._memory_text = memory_text or ""

    @property
    def memory_text(self) -> str:
        return self._memory_text

    @property
    def unwrapped_env(self) -> Any:
        return self._env

    # -- gym surface ----------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        if self._memory_text:
            obs = f"{self._memory_text}\n\n{obs}"
        self.last_observation = obs
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def __getattr__(self, name: str):
        # Only reached for names not found on the proxy itself.
        return getattr(self._env, name)


# ══════════════════════════════════════════════════════════════════════
#  Model-call meter
# ══════════════════════════════════════════════════════════════════════


class _CallMeter:
    """Counts ``runner.generate`` calls, bucketed by phase."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {"attempt": 0, "reflection": 0, "judge": 0}
        self.phase: str = "attempt"

    def bump(self) -> None:
        self.counts[self.phase] = self.counts.get(self.phase, 0) + 1

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


_MISSING = object()


@contextmanager
def _meter_model_calls(runner: Any):
    """Temporarily wrap ``runner.generate`` (instance attribute only) with a counter."""
    meter = _CallMeter()
    previous = runner.__dict__.get("generate", _MISSING)
    inner = runner.generate  # bound method (or a previously-set instance attr)

    def counting_generate(*args, **kwargs):
        meter.bump()
        return inner(*args, **kwargs)

    counting_generate._reflexion_meter = meter  # type: ignore[attr-defined]
    runner.generate = counting_generate
    try:
        yield meter
    finally:
        if previous is _MISSING:
            runner.__dict__.pop("generate", None)
        else:
            runner.generate = previous


@contextmanager
def _temporary_max_new_tokens(runner: Any, value: Optional[int]):
    """Cap decoding length for the reflection call only, then restore."""
    cfg = getattr(runner, "config", None)
    if value is None or cfg is None or not hasattr(cfg, "max_new_tokens"):
        yield
        return
    original = cfg.max_new_tokens
    cfg.max_new_tokens = value
    try:
        yield
    finally:
        cfg.max_new_tokens = original


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════


class ReflexionRunner:
    """Reflexion self-critique-and-retry driver wrapping an ``AgentRunner``."""

    def __init__(
        self,
        runner: Any,
        config: Optional[ReflexionConfig] = None,
        *,
        reflector: Any = None,
    ):
        """
        Args:
            runner: a loaded ``AgentRunner`` (or any object exposing the same
                public surface: ``config``, ``generate``, ``run_task``).
            config: :class:`ReflexionConfig`.
            reflector: the object that writes reflections.  Must be ``runner``
                itself (or ``None``).  Anything else raises
                :class:`SameModelViolation` — a stronger critic would make this
                baseline uninterpretable and would leak that critic's ability
                into a "prompting only" number.
        """
        self.runner = runner
        self.config = config or ReflexionConfig()

        self._reflector = runner if reflector is None else reflector
        if self._reflector is not self.runner:
            raise SameModelViolation(
                "Reflexion reflections must be generated by the model under test. "
                f"Got reflector={type(self._reflector).__name__} which is not the "
                "AgentRunner passed in. Using a different (typically stronger) "
                "model turns this baseline into distillation and invalidates the "
                "'prompting only, no training' claim."
            )

        rcfg = getattr(runner, "config", None)
        self.model_id: str = str(getattr(rcfg, "model_name_or_path", "<unknown-model>"))
        self.backend: Optional[str] = getattr(rcfg, "backend", None)

        out = self.config.output_dir
        if out is None:
            base = getattr(runner, "log_path", None)
            out = str(Path(base) / "reflexion") if base is not None else None
        self.output_dir: Optional[Path] = Path(out) if out is not None else None
        if self.output_dir is not None and self.config.save_records:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "reflexion_config.json").write_text(
                json.dumps(
                    {
                        "strategy": self.config.strategy.value,
                        "max_attempts": self.config.max_attempts,
                        "retry_on_none": self.config.retry_on_none,
                        "qa_success_threshold": self.config.qa_success_threshold,
                        "action_success_threshold": self.config.action_success_threshold,
                        "reward_success_threshold": self.config.reward_success_threshold,
                        "ungraded_policy": self.config.ungraded_policy.value,
                        "reflection_max_new_tokens": self.config.reflection_max_new_tokens,
                        "model_id": self.model_id,
                        "backend": self.backend,
                        "reflection_model_id": self.model_id,
                        "same_model_reflection": True,
                        "created": datetime.now().isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # ── same-model enforcement ───────────────────────────────────────
    def _check_same_model(self) -> None:
        """Re-validate the same-model constraint before every reflection call."""
        if self._reflector is not self.runner:
            raise SameModelViolation(
                "Reflector is no longer the runner under test — reflection aborted."
            )
        current = str(
            getattr(getattr(self._reflector, "config", None), "model_name_or_path", "")
        )
        if current != self.model_id:
            raise SameModelViolation(
                "Model identity drifted during the run: reflection would be produced "
                f"by {current!r} but the policy under test is {self.model_id!r}."
            )
        current_backend = getattr(getattr(self._reflector, "config", None), "backend", None)
        if current_backend != self.backend:
            raise SameModelViolation(
                "Backend drifted during the run: reflection would be produced via "
                f"{current_backend!r} but the policy under test uses {self.backend!r}."
            )

    # ── failure signal ───────────────────────────────────────────────
    @staticmethod
    def _gold_answer(task: dict) -> str:
        return str(task.get("answer", task.get("correct_answer", "")) or "").strip()

    @staticmethod
    def _expected_actions(task: dict) -> list:
        crit = task.get("evaluation_criteria") or {}
        return list(crit.get("actions") or [])

    def _score_attempt(
        self, task: dict, env: Any, task_result: Any
    ) -> tuple[Optional[bool], SuccessSignal, float, dict]:
        """Resolve the pass/fail verdict from the environment's own scoring.

        Returns ``(success, signal, score, details)`` where ``success is None``
        means *ungraded* — never silently ``False``.
        """
        details: dict[str, Any] = {"qa_accuracy": None, "action_score": None}

        if getattr(task_result, "error", None):
            return False, SuccessSignal.ERROR, 0.0, details

        tool_log = list(getattr(env, "_tool_call_log", []) or [])
        trajectory = getattr(task_result, "trajectory", None) or {}

        gold = self._gold_answer(task)
        if gold:
            qa = trajectory.get("qa_accuracy") if isinstance(trajectory, dict) else None
            if qa is None and hasattr(self.runner, "_compute_qa_accuracy"):
                qa = self.runner._compute_qa_accuracy(task, tool_log)
            qa = float(qa or 0.0)
            details["qa_accuracy"] = qa
            return qa >= self.config.qa_success_threshold, SuccessSignal.QA_ACCURACY, qa, details

        # NOTE: _compute_action_score returns 1.0 when nothing is expected, so the
        # non-empty guard here is what stops every criteria-less task from being
        # scored "solved" and collapsing the ladder to strategy NONE.
        if self._expected_actions(task) and hasattr(self.runner, "_compute_action_score"):
            act = float(self.runner._compute_action_score(task, tool_log))
            details["action_score"] = act
            return (
                act >= self.config.action_success_threshold,
                SuccessSignal.ACTION_SCORE,
                act,
                details,
            )

        if self.config.reward_success_threshold is not None:
            reward = float(getattr(task_result, "final_reward", 0.0) or 0.0)
            return (
                reward >= self.config.reward_success_threshold,
                SuccessSignal.COMPOSITE_REWARD,
                reward,
                details,
            )

        return None, SuccessSignal.UNGRADED, float(getattr(task_result, "final_reward", 0.0) or 0.0), details

    # ── prompts ──────────────────────────────────────────────────────
    @staticmethod
    def _question_text(task: dict) -> str:
        """The task statement, with no gold answer in it."""
        ticket = task.get("ticket") or ""
        if not ticket:
            desc = task.get("description") or {}
            if isinstance(desc, dict):
                ticket = desc.get("purpose", "") or ""
            else:
                ticket = str(desc)
        return str(ticket).strip() or f"(task {task.get('id', '?')})"

    def _assert_no_gold_leak(self, task: dict, prompt: str) -> None:
        """Refuse to send the gold answer into a reflection/judge prompt.

        Only checkable for free-text golds (a single-letter MCQA key trivially
        occurs in any English text), so the guard is length-gated.
        """
        gold = self._gold_answer(task)
        if len(gold) < 12:
            return
        ticket = self._question_text(task)
        if gold.lower() in prompt.lower() and gold.lower() not in ticket.lower():
            raise GoldAnswerLeak(
                f"Gold answer for task {task.get('id')!r} would be exposed to the "
                "reflection prompt; refusing to generate."
            )

    def _generate_reflection(self, task: dict, scratchpad: str) -> str:
        """Write a verbal self-reflection with the SAME model under test."""
        self._check_same_model()
        prompt = REFLECT_INSTRUCTION.format(
            question=self._question_text(task), scratchpad=scratchpad
        )
        self._assert_no_gold_leak(task, prompt)
        messages = [
            {
                "role": "system",
                "content": "You are a clinical AI agent performing self-reflection on your own failed attempt.",
            },
            {"role": "user", "content": prompt},
        ]
        with _temporary_max_new_tokens(self.runner, self.config.reflection_max_new_tokens):
            text = self.runner.generate(messages, tools=None)
        return (text or "").strip()

    def _self_judge(self, task: dict, scratchpad: str) -> bool:
        """Same-model verdict used only under ``UngradedPolicy.SELF_JUDGE``."""
        self._check_same_model()
        prompt = SELF_JUDGE_INSTRUCTION.format(
            question=self._question_text(task), scratchpad=scratchpad
        )
        self._assert_no_gold_leak(task, prompt)
        messages = [
            {
                "role": "system",
                "content": "You are strictly reviewing your own previous attempt. No answer key is available.",
            },
            {"role": "user", "content": prompt},
        ]
        with _temporary_max_new_tokens(self.runner, self.config.reflection_max_new_tokens):
            verdict = (self.runner.generate(messages, tools=None) or "").strip()
        head = verdict.upper()
        # "UNSOLVED" contains "SOLVED": test the negative first.
        return not ("UNSOLVED" in head or "NOT SOLVED" in head) and "SOLVED" in head

    def _build_memory(
        self,
        question: str,
        scratchpad: str,
        reflections: list[str],
        new_reflection: Optional[str],
    ) -> tuple[str, list[str]]:
        """Per-strategy memory update — reference ``reflect()`` semantics."""
        strategy = self.config.strategy

        if strategy is ReflexionStrategy.NONE:
            return "", reflections

        if strategy is ReflexionStrategy.LAST_ATTEMPT:
            reflections = [scratchpad]
            memory = format_last_attempt(question, scratchpad)

        elif strategy is ReflexionStrategy.REFLEXION:
            if new_reflection:
                reflections = reflections + [new_reflection]
            memory = format_reflections(reflections)

        elif strategy is ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            memory = format_last_attempt(question, scratchpad)
            # Faithful to the reference: this branch REPLACES the reflection
            # list rather than accumulating (agents.py:307-309).
            reflections = [new_reflection] if new_reflection else []
            memory += "\n" + format_reflections(
                reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER
            )
        else:  # pragma: no cover - enum is closed
            raise NotImplementedError(f"Unknown reflection strategy: {strategy}")

        if len(memory) > self.config.max_memory_chars:
            memory = memory[: self.config.max_memory_chars].rstrip() + "\n... [memory truncated]"

        block = (
            f"{MEMORY_BLOCK_OPEN}\n"
            f"{memory.strip()}\n"
            f"{MEMORY_BLOCK_CLOSE}"
        )
        return block, reflections

    # ── main loop ────────────────────────────────────────────────────
    def run_task(self, task: dict, env: Any) -> ReflexionTaskResult:
        """Run the ladder on one task.  ``env`` is any BioAgentGymEnv-like object."""
        cfg = self.config
        proxy = env if isinstance(env, ReflexionEnvProxy) else ReflexionEnvProxy(env)
        question = self._question_text(task)

        out = ReflexionTaskResult(
            task_id=str(task.get("id", "?")),
            domain=str(getattr(getattr(self.runner, "config", None), "domain", "") or task.get("domain", "")),
            strategy=cfg.strategy.value,
            model_id=self.model_id,
            backend=self.backend,
        )

        reflections: list[str] = []
        memory_block = ""
        graded_seen = False

        with _meter_model_calls(self.runner) as meter:
            for attempt_idx in range(cfg.max_attempts):
                proxy.set_memory(memory_block)

                meter.phase = "attempt"
                before = meter.snapshot()
                t0 = time.time()
                task_result = self.runner.run_task(task, proxy)
                latency = time.time() - t0
                after = meter.snapshot()
                attempt_calls = after["attempt"] - before["attempt"]

                success, signal, score, details = self._score_attempt(task, proxy, task_result)
                scratchpad = render_scratchpad(
                    task_result,
                    max_observation_chars=cfg.max_observation_chars,
                    max_chars=cfg.max_scratchpad_chars,
                )

                judge_calls = 0
                assumed_failure = False
                if success is None:
                    if cfg.ungraded_policy is UngradedPolicy.SELF_JUDGE:
                        meter.phase = "judge"
                        before_j = meter.snapshot()
                        success = self._self_judge(task, scratchpad)
                        judge_calls = meter.snapshot()["judge"] - before_j["judge"]
                        signal = SuccessSignal.SELF_JUDGE
                    elif cfg.ungraded_policy is UngradedPolicy.RETRY_ALL:
                        assumed_failure = True
                else:
                    graded_seen = True

                record = AttemptRecord(
                    task_id=out.task_id,
                    domain=out.domain,
                    strategy=cfg.strategy.value,
                    attempt_idx=attempt_idx,
                    success=success,
                    success_signal=signal.value,
                    score=float(score),
                    assumed_failure=assumed_failure,
                    qa_accuracy=details.get("qa_accuracy"),
                    action_score=details.get("action_score"),
                    final_reward=float(getattr(task_result, "final_reward", 0.0) or 0.0),
                    total_turns=int(getattr(task_result, "total_turns", 0) or 0),
                    model_calls_attempt=attempt_calls,
                    model_calls_judge=judge_calls,
                    memory_injected=bool(memory_block),
                    memory_chars=len(memory_block),
                    memory_preview=memory_block[:400],
                    latency_seconds=latency,
                    error=getattr(task_result, "error", None),
                )
                out.attempts.append(record)
                out.final_task_result = task_result
                if score > out.best_score or attempt_idx == 0:
                    out.best_score = float(score)
                    out.best_attempt_idx = attempt_idx
                    out.best_task_result = task_result

                if success is True:
                    out.success = True
                    out.attempts_to_success = attempt_idx + 1
                    out.stop_reason = "solved"
                    break

                if attempt_idx == cfg.max_attempts - 1:
                    out.stop_reason = "attempt_budget_exhausted"
                    break

                if cfg.strategy is ReflexionStrategy.NONE and not cfg.retry_on_none:
                    out.stop_reason = "strategy_none_single_attempt"
                    break

                if success is None and cfg.ungraded_policy is UngradedPolicy.STOP_AFTER_FIRST:
                    out.degraded_to_single_attempt = True
                    out.stop_reason = "ungraded_stop_after_first"
                    logger.warning(
                        f"[reflexion] task {out.task_id}: no eval-time correctness signal; "
                        "UngradedPolicy.STOP_AFTER_FIRST => behaving as strategy NONE "
                        "(recorded as degraded_to_single_attempt)."
                    )
                    break

                # ── failed: build memory for the next attempt ──
                new_reflection = None
                if cfg.strategy.uses_verbal_reflection:
                    meter.phase = "reflection"
                    before_r = meter.snapshot()
                    try:
                        new_reflection = self._generate_reflection(task, scratchpad)
                    except SameModelViolation:
                        raise
                    except GoldAnswerLeak:
                        raise
                    except Exception as exc:  # a flaky backend must not kill the run
                        logger.warning(f"[reflexion] reflection failed on {out.task_id}: {exc}")
                        new_reflection = None
                    record.model_calls_reflection = (
                        meter.snapshot()["reflection"] - before_r["reflection"]
                    )
                    if new_reflection:
                        record.reflection_text = new_reflection
                        record.reflection_model_id = self.model_id
                        out.reflections.append(new_reflection)

                memory_block, reflections = self._build_memory(
                    question, scratchpad, reflections, new_reflection
                )
                out.stop_reason = "retrying"

            counts = meter.snapshot()

        out.attempts_used = len(out.attempts)
        out.graded = graded_seen
        out.model_calls_attempt = counts.get("attempt", 0)
        out.model_calls_reflection = counts.get("reflection", 0)
        out.model_calls_judge = counts.get("judge", 0)
        out.model_calls_total = sum(counts.values())
        out.baseline_model_calls = out.attempts[0].model_calls_attempt if out.attempts else 0
        out.extra_model_calls = out.model_calls_total - out.baseline_model_calls
        out.call_overhead_ratio = (
            out.model_calls_total / out.baseline_model_calls if out.baseline_model_calls else 1.0
        )
        out.total_latency = sum(a.latency_seconds for a in out.attempts)

        self._persist(out)
        logger.info(
            f"[reflexion:{cfg.strategy.value}] task={out.task_id} "
            f"success={out.success} attempts={out.attempts_used} "
            f"calls={out.model_calls_total} (+{out.extra_model_calls}, "
            f"{out.call_overhead_ratio:.2f}x) stop={out.stop_reason}"
        )
        return out

    # ── batch ────────────────────────────────────────────────────────
    def run_all_tasks(
        self,
        tasks: Optional[Iterable[dict]] = None,
        env: Any = None,
        task_ids: Optional[list[str]] = None,
    ) -> list[ReflexionTaskResult]:
        """Run the ladder over a task set, mirroring ``AgentRunner.run_all_tasks``."""
        if env is None:
            from bioagents.gym.agent_env import BioAgentGymEnv  # local: heavy import

            rcfg = self.runner.config
            env = BioAgentGymEnv(
                domain=rcfg.domain,
                task_split=getattr(rcfg, "task_split", None),
                max_turns=rcfg.max_turns,
            )

        proxy = env if isinstance(env, ReflexionEnvProxy) else ReflexionEnvProxy(env)

        if tasks is None:
            task_map = getattr(proxy, "_task_map", {}) or {}
            ids = task_ids or getattr(self.runner.config, "task_ids", None)
            if ids:
                tasks = [task_map[i] for i in ids]
            else:
                tasks = list(getattr(proxy, "_tasks", []) or list(task_map.values()))
        tasks = list(tasks)

        logger.info(
            f"[reflexion] {len(tasks)} tasks | strategy={self.config.strategy.value} "
            f"| max_attempts={self.config.max_attempts} | model={self.model_id}"
        )
        results = [self.run_task(t, proxy) for t in tasks]

        summary = self.summarize(results)
        if self.output_dir is not None and self.config.save_records:
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return results

    # ── reporting ────────────────────────────────────────────────────
    def summarize(self, results: list[ReflexionTaskResult]) -> dict:
        """Aggregate the numbers the rebuttal needs, including the cost column."""
        n = len(results)
        graded = [r for r in results if r.graded]
        solved = [r for r in results if r.success]
        base_calls = sum(r.baseline_model_calls for r in results)
        total_calls = sum(r.model_calls_total for r in results)
        return {
            "strategy": self.config.strategy.value,
            "model_id": self.model_id,
            "backend": self.backend,
            "reflection_model_id": self.model_id,
            "same_model_reflection": True,
            "max_attempts": self.config.max_attempts,
            "ungraded_policy": self.config.ungraded_policy.value,
            "n_tasks": n,
            "n_graded": len(graded),
            "n_ungraded": n - len(graded),
            "n_degraded_to_single_attempt": sum(
                1 for r in results if r.degraded_to_single_attempt
            ),
            "success_rate_all": (len(solved) / n) if n else 0.0,
            "success_rate_graded": (
                sum(1 for r in graded if r.success) / len(graded) if graded else None
            ),
            "mean_attempts": (sum(r.attempts_used for r in results) / n) if n else 0.0,
            "mean_attempts_to_success": (
                sum(r.attempts_to_success for r in solved) / len(solved) if solved else None
            ),
            "attempts_to_success_histogram": {
                str(k): sum(1 for r in solved if r.attempts_to_success == k)
                for k in range(1, self.config.max_attempts + 1)
            },
            "model_calls_total": total_calls,
            "model_calls_attempt": sum(r.model_calls_attempt for r in results),
            "model_calls_reflection": sum(r.model_calls_reflection for r in results),
            "model_calls_judge": sum(r.model_calls_judge for r in results),
            "baseline_model_calls": base_calls,
            "extra_model_calls": sum(r.extra_model_calls for r in results),
            "call_overhead_ratio": (total_calls / base_calls) if base_calls else 1.0,
            "total_latency": sum(r.total_latency for r in results),
        }

    # ── persistence ──────────────────────────────────────────────────
    def _persist(self, result: ReflexionTaskResult) -> None:
        if self.output_dir is None or not self.config.save_records:
            return
        try:
            with (self.output_dir / "attempts.jsonl").open("a", encoding="utf-8") as fh:
                for rec in result.attempts:
                    fh.write(json.dumps(rec.to_dict(), ensure_ascii=False, default=str) + "\n")
            with (self.output_dir / "tasks.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result.to_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # persistence must never kill an eval run
            logger.warning(f"[reflexion] failed to persist records: {exc}")
