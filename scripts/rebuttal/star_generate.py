"""STaR rejection-sampling generation pass for the Healthcare AI GYM rebuttal.

REIMPLEMENTATION NOTICE
=======================
This is an INDEPENDENT REIMPLEMENTATION of the STaR algorithm (Zelikman, Wu,
Mu & Goodman, "STaR: Bootstrapping Reasoning With Reasoning", NeurIPS 2022,
arXiv:2203.14465).  The official release, ``github.com/ezelikman/STaR``, was
NOT run and could not be: it is JAX 0.2.12 / mesh-transformer-jax / GPT-J-6B on
TPU pods driven by Ray, none of which exists on this cluster.  Nothing produced
by this file may be described as output of the authors' code.  What is
reproduced is the ALGORITHM — sample, filter on correctness, fine-tune, repeat
— against this paper's own environment, scorers and task pool.  Every deviation
from the paper is enumerated in ``baselines/STAR_PORT.md``.

WHAT THIS PASS DOES
===================
One STaR *generation* half-iteration:

  1. roll the current policy over the training pool through the GYM
     (``BioAgentGymEnv`` + ``AgentRunner.run_task`` — the same loop every other
     condition in the rebuttal is evaluated with),
  2. score every rollout with the environment's own scorers,
  3. keep the rollouts that pass the acceptance test,
  4. write them in the two shapes ``bioagents/training/sft_trainer.py`` already
     consumes, and
  5. report the acceptance rate and the full score distribution.

``scripts/rebuttal/star_iterate.py`` is the outer loop that calls this.

ACCEPTANCE SIGNAL — THE DESIGN DECISION THAT MATTERS
====================================================
The GYM's composite reward weights six dimensions (accuracy .25, format .10,
process .20, safety .20, coherence .10, assertion .15 — see
``bioagents/evaluation/rewards.py::compute_composite_reward``).  Accepting on
that composite would select trajectories for being well-FORMATTED, and a
baseline built that way would exhibit the paper's bias instead of testing for it
— it would adapt to the reward's surface form, the very confound this arm has to
rule out rather than reproduce.  It
is not a hypothetical: a fully correct rollout measured against a real task
here scored ``accuracy=1.0`` but ``total=0.570``, while a wrong-but-fluent
rollout with the same tool usage scores ABOVE it.  ``tests/test_star_baseline.py``
asserts that inversion against the live scorer.

So the default acceptance signal is ``qa_accuracy`` — correctness alone, from
the environment's own ``AgentRunner._compute_qa_accuracy``.  This also matches
the sibling baseline already in this repo: ``ReflexionConfig`` sets
``qa_success_threshold = 1.0`` and leaves ``reward_success_threshold = None``
with the comment "do NOT use the shaped composite reward as a pass/fail
signal".

Every signal is nevertheless COMPUTED AND RECORDED for every rollout, and
``--refilter-from`` re-runs the filter under a different signal with no new
rollouts, so the rebuttal can report the composite-accepted variant for the
price of a file read rather than a second 10k-rollout run.

Signals (``--accept-on``):
  qa_accuracy      env's own submit_answer grader (exact letter for MC).  DEFAULT.
  accuracy         the composite's accuracy DIMENSION alone (accuracy_reward_soft):
                   exact match for A-E, normalised match for short answers,
                   token-F1 for long ones.  Accepts correct answers that never
                   reached submit_answer, so it is the more permissive
                   accuracy-only signal.
  verl_acc         reward_fn.compute_score(...)["acc"] — the binary readout the
                   RL arms are compared on, recomputed here on a reconstructed
                   verl-style transcript.  Use when the rebuttal needs STaR and
                   the RL arms judged by literally the same scorer.
  composite_reward the full 6D composite.  Provided so the format-adaptation
                   variant can be reported; NOT the default, for the reason above.
  action_score     expected-vs-actual tool-call completeness.  Recorded for
                   diagnosis; a poor acceptance signal on its own (it is 1.0 by
                   definition for any task with no expected actions).
  verl_score       reward_fn's SHAPED score.  Recorded only — it is arm-specific
                   by construction (reward_fn's own docstring says it is "NOT
                   comparable across arms"), so accepting on it is a mistake.

Usage
-----
    # generation pass against a served policy
    python scripts/rebuttal/star_generate.py \
        --model /path/to/ckpt --backend sglang --server-url http://127.0.0.1:31000 \
        --pool full_4modality_clean --split train \
        --samples-per-task 3 --temperature 1.0 --max-turns 5 \
        --out-dir /path/to/iter_00/gen

    # re-filter an existing pass under a different signal, no rollouts
    python scripts/rebuttal/star_generate.py --refilter-from /path/to/iter_00/gen \
        --accept-on composite_reward --out-dir /path/to/iter_00/gen_composite

    # no-GPU smoke test
    python scripts/rebuttal/star_generate.py --backend mock --limit 8 --out-dir /tmp/star_smoke
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import statistics
import sys
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

HCGYM_ROOT = Path(__file__).resolve().parents[2]
if str(HCGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(HCGYM_ROOT))

from loguru import logger  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
#  Pool → gym domain
# ══════════════════════════════════════════════════════════════════════
#
# The 4-modality pools carry a ``_source_domain`` per task, which is a DATA
# domain name and not always a registered gym domain.  The mapping below is a
# copy of the project's single source of truth,
# ``scripts/verl/bioagents_tool.py::DATA_TO_BIO_DOMAIN`` — the map the RL arms'
# tool server itself uses, so a STaR rollout gets the same tool set the RL arms
# get for the same task.  It is copied rather than imported because that module
# imports verl at module scope and verl is not installed in this venv.
#
# tests/test_star_baseline.py re-parses that file with ``ast`` and asserts this
# copy is identical, so the duplication cannot silently drift.
DATA_TO_BIO_DOMAIN = {
    "text_qa": "medical_qa",
    "multimodal_vqa": "visual_diagnosis",
    "clinical_diagnosis": "clinical_diagnosis",
    "drug_interaction": "drug_interaction",
    "ehr_management": "ehr_management",
    "triage_emergency": "triage_emergency",
    "psychiatry": "psychiatry",
    "obstetrics": "obstetrics",
    "radiology_report": "radiology_report",
}
FALLBACK_DOMAIN = "medical_qa"


def read_data_to_bio_domain_literal(path: Path) -> dict:
    """Extract ``DATA_TO_BIO_DOMAIN`` from bioagents_tool.py WITHOUT importing it.

    bioagents_tool.py imports verl at module scope, so a plain import fails
    outside a verl environment.  The map is a plain dict literal, so ``ast``
    can read it exactly.  Used by the drift test.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "DATA_TO_BIO_DOMAIN":
                    return ast.literal_eval(node.value)
    raise KeyError(f"DATA_TO_BIO_DOMAIN not found in {path}")


# ══════════════════════════════════════════════════════════════════════
#  Acceptance signals
# ══════════════════════════════════════════════════════════════════════

# Signals a rollout may be accepted on, with the threshold used when
# --accept-threshold is not given.  A rollout is accepted when
# score >= threshold.
DEFAULT_THRESHOLDS = {
    # Binary by construction: 1.0 == the env's own grader called it correct.
    "qa_accuracy": 1.0,
    # 1.0/0.0 for multiple-choice and short answers; token-F1 for long ones.
    # 0.5 mirrors reward_fn's own open-ended correctness cut (`overlap > 0.5`);
    # note reward_fn uses strict `>` and this uses `>=`, which differ only for a
    # rollout whose F1 is exactly 0.5.
    "accuracy": 0.5,
    # Binary.
    "verl_acc": 1.0,
    # The composite's realistic ceiling is well below 1.0 (a perfectly correct
    # rollout measured on a real task scored 0.570), so a "high" threshold here
    # is much lower than it looks.  See STAR_PORT.md.
    "composite_reward": 0.5,
    "action_score": 1.0,
    # Recorded, not recommended: shaped and therefore arm-specific.
    "verl_score": 0.5,
}

# Signals that select on correctness only.  Anything outside this set mixes
# format/process/safety terms into the acceptance decision and must be reported
# as such.
ACCURACY_ONLY_SIGNALS = frozenset({"qa_accuracy", "accuracy", "verl_acc"})

SIGNAL_NAMES = tuple(DEFAULT_THRESHOLDS)


# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════


@dataclass
class GenConfig:
    """One STaR generation pass."""

    # Policy
    model: str = "mock-policy"
    backend: str = "sglang"          # sglang | transformers | vllm | mock
    server_url: Optional[str] = None
    temperature: float = 1.0          # STaR samples; greedy would give k identical rollouts
    top_p: float = 0.95
    max_new_tokens: int = 2048
    no_think: bool = False
    prompt_mode: str = "default"

    # Task pool
    pool: str = "full_4modality_clean"
    split: str = "train"
    limit: int = 0                    # 0 = whole split
    seed: int = 42

    # Sampling
    samples_per_task: int = 3
    max_turns: int = 5
    workers: int = 8

    # Acceptance
    accept_on: str = "qa_accuracy"
    accept_threshold: Optional[float] = None
    max_accepted_per_task: int = 1    # STaR keeps ONE rationale per problem; 0 = keep all

    # Threshold sanity.  An acceptance rate at either extreme makes the arm
    # uninformative and must never pass silently.
    min_informative_rate: float = 0.02
    max_informative_rate: float = 0.98
    allow_uninformative: bool = False

    # Output
    out_dir: str = ""
    eval_only: bool = False           # score only; write no training data
    keep_rejected: bool = True        # needed for --refilter-from

    def resolved_threshold(self) -> float:
        if self.accept_threshold is not None:
            return float(self.accept_threshold)
        return DEFAULT_THRESHOLDS[self.accept_on]


# ══════════════════════════════════════════════════════════════════════
#  Pool loading
# ══════════════════════════════════════════════════════════════════════


def load_pool(pool: str, split: str, limit: int = 0, seed: int = 42,
              root: Optional[Path] = None) -> list[dict]:
    """Load one split of a decontaminated task pool.

    ``pool`` is a directory under ``data/domains/`` holding ``tasks.json`` and
    ``split_tasks.json``; the RL arms consume the parquet built from exactly
    these two files, so the STaR pool is the RL pool by construction.
    """
    root = root or HCGYM_ROOT
    base = Path(root) / "data" / "domains" / pool
    tasks = json.loads((base / "tasks.json").read_text(encoding="utf-8"))
    splits = json.loads((base / "split_tasks.json").read_text(encoding="utf-8"))
    if split not in splits:
        raise KeyError(f"split '{split}' not in {base/'split_tasks.json'} (have {list(splits)})")
    by_id = {t["id"]: t for t in tasks}
    out = [by_id[i] for i in splits[split] if i in by_id]
    if limit and limit < len(out):
        rng = random.Random(seed)
        out = sorted(rng.sample(out, limit), key=lambda t: t["id"])
    return out


def gym_domain_for(task: dict) -> str:
    return DATA_TO_BIO_DOMAIN.get(task.get("_source_domain", ""), FALLBACK_DOMAIN)


# ══════════════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════════════


def render_verl_solution(turns: list) -> str:
    """Rebuild a verl-style transcript from an AgentRunner trajectory.

    ``scripts/verl/reward_fn.py`` scores the flat rollout string verl produces,
    in which each tool call appears as ``<function=NAME>`` and each result as
    ``<tool_response>``.  AgentRunner keeps the same information structurally
    (``parsed_tool_call`` / ``tool_response``) but not as that markup, so the
    markup is re-emitted here.  This is a RECONSTRUCTION and is labelled as one
    everywhere it is used: it makes ``verl_acc`` comparable to the RL arms'
    readout, it is not a byte-identical replay of a verl rollout.

    A turn whose raw output already carries ``<function=`` is passed through
    unchanged, so a model that natively emits the qwen3_coder format is not
    double-counted.
    """
    parts: list[str] = []
    for t in turns:
        raw = t.get("raw_output") or ""
        call = t.get("parsed_tool_call")
        if raw:
            parts.append(raw)
        if call and "<function=" not in raw:
            name = call.get("name", "tool")
            args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            parts.append(f"<function={name}>\n{args}\n</function>")
        resp = t.get("tool_response")
        if resp:
            parts.append(f"<tool_response>\n{resp}\n</tool_response>")
    return "\n".join(parts)


def verl_ground_truth(task: dict) -> tuple[str, bool]:
    """The (ground_truth, has_options) pair the RL arms' parquet carries.

    Replicates ``scripts/verl/convert_tasks_to_parquet.py`` lines 65-68 and 79
    exactly.  ``tests/test_star_baseline.py`` checks this against the ACTUAL
    parquet the RL arms train on, row by row, so a divergence is caught rather
    than argued about.
    """
    has_options = bool(task.get("options"))
    correct = task.get("correct_answer", "")
    raw_answer = task.get("raw_answer", "")
    return (correct if has_options else raw_answer), has_options


_reward_fn_mod = None
_reward_fn_lock = threading.Lock()


def _reward_fn():
    """Import scripts/verl/reward_fn.py read-only (stdlib-only module, no verl)."""
    global _reward_fn_mod
    with _reward_fn_lock:
        if _reward_fn_mod is None:
            verl_dir = str(HCGYM_ROOT / "scripts" / "verl")
            if verl_dir not in sys.path:
                sys.path.insert(0, verl_dir)
            import reward_fn as _m  # noqa: PLC0415
            _reward_fn_mod = _m
    return _reward_fn_mod


def score_rollout(task: dict, result, turn_dicts: list) -> dict[str, float]:
    """Compute every acceptance signal for one finished rollout.

    All of them come from the environment's own scorers; nothing is invented
    here.  ``result`` is an ``AgentRunner.TaskResult``.
    """
    traj = result.trajectory if isinstance(result.trajectory, dict) else {}
    details = traj.get("reward_details", {}) or {}

    scores: dict[str, float] = {
        # AgentRunner puts this on the trajectory for any task carrying a gold
        # answer; absent (ungraded task) it stays None and the rollout is
        # dropped from acceptance rather than counted as a pass.
        "qa_accuracy": float(traj["qa_accuracy"]) if "qa_accuracy" in traj else float("nan"),
        "accuracy": float(details.get("accuracy", 0.0)),
        "composite_reward": float(result.final_reward or 0.0),
        "action_score": float(result.action_score or 0.0),
    }

    gt, has_options = verl_ground_truth(task)
    solution = render_verl_solution(turn_dicts)
    try:
        payload = _reward_fn().compute_score(
            data_source="bioagents_medical",
            solution_str=solution,
            ground_truth=gt,
            extra_info={
                "has_options": has_options,
                "correct_answer": task.get("correct_answer", ""),
                "raw_answer": task.get("raw_answer", ""),
                "domain": task.get("_source_domain", "unknown"),
                "task_id": task.get("id", ""),
                "options": task.get("options", {}) or {},
            },
        )
        scores["verl_acc"] = float(payload["acc"])
        scores["verl_score"] = float(payload["score"])
        scores["verl_answer_found"] = float(payload["answer_found"])
        scores["verl_degenerate"] = float(payload["degenerate"])
        scores["verl_n_tool_calls"] = float(payload["n_tool_calls"])
        scores["verl_n_invalid_tool_calls"] = float(payload["n_invalid_tool_calls"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"reward_fn scoring failed for {task.get('id')}: {exc}")
        for k in ("verl_acc", "verl_score", "verl_answer_found", "verl_degenerate",
                  "verl_n_tool_calls", "verl_n_invalid_tool_calls"):
            scores[k] = float("nan")

    # Dimensions of the composite, kept so the note can show WHICH dimension a
    # composite-accepted rollout was actually selected by.
    for dim in ("format", "process", "safety", "coherence", "assertion"):
        scores[f"dim_{dim}"] = float(details.get(dim, 0.0))
    return scores


def is_accepted(scores: dict, accept_on: str, threshold: float) -> bool:
    v = scores.get(accept_on)
    if v is None:
        return False
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    if v != v:  # NaN — ungraded task, no correctness signal to accept on
        return False
    return v >= threshold


# ══════════════════════════════════════════════════════════════════════
#  Rollout
# ══════════════════════════════════════════════════════════════════════


class _MessageSpy:
    """Capture the exact message list the policy was shown, per thread.

    The SFT rendering must be the conversation the policy actually saw, not a
    re-derivation of it: re-deriving would silently drift the moment the prompt
    builder in agent_runner.py changes (and that file is owned elsewhere).  The
    same wrapping trick is already used in the repo by
    ``eval_benchmark_multiturn._run_single_task_multiturn_debug``.
    """

    def __init__(self, inner: Callable):
        self._inner = inner
        self._local = threading.local()

    def reset(self):
        self._local.seen = []

    def last(self) -> list:
        """The most recent message list this thread's policy was shown."""
        seen = getattr(self._local, "seen", None) or []
        return seen[-1] if seen else []

    def __call__(self, messages, tools=None):
        seen = getattr(self._local, "seen", None)
        if seen is None:
            seen = self._local.seen = []
        seen.append(json.loads(json.dumps(messages, default=str)))
        return self._inner(messages, tools=tools)


def turn_dicts_from(result) -> list[dict]:
    """AgentRunner turns → the dict shape ``_save_task_result`` writes.

    Identical to ``AgentRunner._save_task_result`` EXCEPT that the tool response
    is not truncated to 500 characters.  The truncation there is a log-size
    measure; keeping the full text matters here because these dicts are the
    training data, and a trajectory whose evidence has been cut mid-sentence
    teaches the model to answer from a cut-off observation.
    """
    return [
        {
            "turn_idx": t.turn_idx,
            "raw_output": t.raw_output,
            "parsed_tool_call": t.parsed_tool_call,
            "tool_call_format": getattr(t, "tool_call_format", ""),
            "tool_response": t.tool_response,
            "is_final_answer": t.is_final_answer,
            "latency_seconds": t.latency_seconds,
        }
        for t in result.turns
    ]


def build_sft_messages(seen_messages: list, turn_dicts: list, prompt_mode: str = "default") -> list:
    """The full chat the policy saw, plus its final assistant turn.

    ``seen_messages`` is the LAST message list handed to ``generate``, i.e. the
    conversation up to (not including) the final assistant turn.  Appending
    that turn — rendered exactly the way ``run_task`` would have replayed it —
    yields the complete trajectory, system prompt and task ticket included.

    This is why the ``sft_path`` shape is preferred over ``trajectory_dir``:
    ``sft_generator.trajectory_to_sft`` rebuilds messages from turns alone and
    therefore emits a training example with NO user question in it (see
    STAR_PORT.md § "What sft_trainer does to a trajectory").
    """
    from bioagents.evaluation.agent_runner import format_assistant_turn  # noqa: PLC0415

    msgs = deepcopy(seen_messages) if seen_messages else []
    if not turn_dicts:
        return msgs
    last = turn_dicts[-1]
    call = last.get("parsed_tool_call")
    raw = last.get("raw_output") or ""
    content = format_assistant_turn(raw, call, prompt_mode) if call else raw
    msgs.append({"role": "assistant", "content": content})
    return msgs


def trajectory_record(task: dict, result, turn_dicts: list, scores: dict,
                      sample_idx: int, cfg: GenConfig, accepted: bool) -> dict:
    """The on-disk trajectory, in the shape ``sft_trainer`` already consumes.

    ``final_reward`` / ``action_score`` / ``turns`` / ``task_id`` are exactly
    the keys ``sft_generator.trajectory_to_sft`` reads, carrying their genuine
    values — the acceptance decision is NOT written back into them.  Downstream
    the SFT config sets ``min_reward: 0.0`` so the trainer's own filter is a
    pass-through over an already-filtered directory; overwriting the recorded
    rewards to force that filter would have falsified the record.
    """
    return {
        "task_id": task["id"],
        "domain": gym_domain_for(task),
        "source_domain": task.get("_source_domain", "unknown"),
        "model_name": Path(cfg.model).name,
        "total_turns": result.total_turns,
        "action_score": result.action_score,
        "final_reward": result.final_reward,
        "completed": result.completed,
        "error": result.error,
        "total_latency": result.total_latency,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "prompt_mode": cfg.prompt_mode,
        "format_adherence": result.format_adherence,
        "turns": turn_dicts,
        # STaR bookkeeping.  Namespaced so it cannot collide with any key the
        # existing loaders read.
        "star": {
            "sample_idx": sample_idx,
            "accepted": accepted,
            "accept_on": cfg.accept_on,
            "accept_threshold": cfg.resolved_threshold(),
            "temperature": cfg.temperature,
            "max_turns": cfg.max_turns,
            "scores": scores,
            "rationalized": False,
        },
    }


def _make_runner(cfg: GenConfig, log_dir: Path):
    """Build a real AgentRunner (or a scripted stand-in for --backend mock)."""
    from bioagents.evaluation.agent_runner import AgentRunner, RunConfig  # noqa: PLC0415

    backend = "transformers" if cfg.backend == "mock" else cfg.backend
    run_cfg = RunConfig(
        model_name_or_path=cfg.model,
        backend=backend,
        server_url=cfg.server_url,
        domain=FALLBACK_DOMAIN,       # overridden per task below
        max_turns=cfg.max_turns,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        log_dir=str(log_dir),
        seed=cfg.seed,
        no_think=cfg.no_think,
        prompt_mode=cfg.prompt_mode,
    )
    runner = AgentRunner(run_cfg)
    if cfg.backend == "mock":
        runner.generate = _mock_policy
    else:
        runner.load_model()
    return runner


def _mock_policy(messages, tools=None) -> str:
    """A deliberately mediocre scripted policy for no-GPU smoke tests.

    Searches once, then submits the first option letter it can see.  It is
    right roughly by chance, which is what makes a smoke run exercise BOTH
    branches of the acceptance filter.
    """
    text = "\n".join(str(m.get("content", "")) for m in messages)
    if "Tool result" not in text and "Observation:" not in text:
        return json.dumps({"name": "think", "arguments": {"thought": "Consider the options."}})
    letter = "A"
    for cand in ("Option A", "A:"):
        if cand in text:
            break
    return json.dumps({
        "name": "submit_answer",
        "arguments": {"answer": letter, "reasoning": f"Best supported option. Answer: {letter}"},
    })


class _EnvCache:
    """One gym env per (thread, domain).

    ``BioAgentGymEnv.__init__`` loads that domain's whole task file, so
    constructing one per rollout would dominate the run.  ``reset()`` rebuilds
    the inner environment and clears the tool log, so reuse is safe — but the
    instance carries per-episode state, so it must not be shared ACROSS threads.
    """

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self._local = threading.local()

    def get(self, domain: str):
        from bioagents.gym.agent_env import BioAgentGymEnv  # noqa: PLC0415
        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = self._local.cache = {}
        if domain not in cache:
            cache[domain] = BioAgentGymEnv(domain=domain, max_turns=self.max_turns)
        return cache[domain]


def rollout_one(runner, spy: _MessageSpy, envs: _EnvCache, task: dict,
                sample_idx: int, cfg: GenConfig) -> dict:
    """Roll the policy once over one task and score it.

    Returns a record carrying the trajectory, the scores and the SFT messages.

    The caller is responsible for having set ``runner.config.domain`` to this
    task's gym domain — ``run_generation`` runs one domain at a time precisely
    so that a shared runner can be driven from several threads without the
    system prompt of one rollout leaking into another.
    """
    domain = gym_domain_for(task)
    env = envs.get(domain)

    # Our pool's version of a task always wins: a registered domain's own
    # tasks.json can carry the same id, and silently scoring against the wrong
    # gold answer would be undetectable downstream.
    injected = getattr(env, "_star_injected", None)
    if injected is None:
        injected = env._star_injected = set()
    env._task_map[task["id"]] = task
    if task["id"] not in injected:
        env._tasks.append(task)
        injected.add(task["id"])

    spy.reset()
    result = runner.run_task(task, env)

    turn_dicts = turn_dicts_from(result)
    scores = score_rollout(task, result, turn_dicts)
    accepted = is_accepted(scores, cfg.accept_on, cfg.resolved_threshold())
    rec = trajectory_record(task, result, turn_dicts, scores, sample_idx, cfg, accepted)
    rec["_sft_messages"] = build_sft_messages(spy.last(), turn_dicts, cfg.prompt_mode)
    return rec


# ══════════════════════════════════════════════════════════════════════
#  Stats
# ══════════════════════════════════════════════════════════════════════


def _quantiles(vals: list[float]) -> dict:
    vals = sorted(v for v in vals if v == v)   # drop NaN
    if not vals:
        return {}
    def q(p):
        if len(vals) == 1:
            return vals[0]
        i = p * (len(vals) - 1)
        lo, hi = int(i), min(int(i) + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (i - lo)
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": vals[0], "p10": q(0.10), "p25": q(0.25), "p50": q(0.50),
        "p75": q(0.75), "p90": q(0.90), "max": vals[-1],
    }


def _histogram(vals: list[float], bins: int = 20) -> dict:
    vals = [v for v in vals if v == v]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {"edges": [lo, hi], "counts": [len(vals)]}
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return {"edges": [lo + i * width for i in range(bins + 1)], "counts": counts}


def summarize(records: list[dict], cfg: GenConfig, n_tasks: int,
              iterations_planned: int = 1) -> dict:
    """Acceptance rate, coverage, score distributions and a threshold verdict.

    The verdict exists because an acceptance rate at either extreme makes this
    baseline uninformative — at ~0 there is nothing to fine-tune on, at ~1 the
    filter is not filtering and STaR degenerates into plain self-imitation —
    and either case must be visible rather than silently trained on.
    """
    n_roll = len(records)
    threshold = cfg.resolved_threshold()
    accepted = [r for r in records if r["star"]["accepted"]]

    per_signal_vals: dict[str, list[float]] = {}
    for r in records:
        for k, v in r["star"]["scores"].items():
            per_signal_vals.setdefault(k, []).append(v)

    covered = {r["task_id"] for r in accepted}
    by_src: dict[str, dict] = {}
    for r in records:
        d = by_src.setdefault(r["source_domain"], {"n_rollouts": 0, "n_accepted": 0,
                                                   "acc_sum": 0.0, "n_tasks": set()})
        d["n_rollouts"] += 1
        d["n_accepted"] += int(r["star"]["accepted"])
        a = r["star"]["scores"].get("accuracy", 0.0)
        d["acc_sum"] += a if a == a else 0.0
        d["n_tasks"].add(r["task_id"])
    for d in by_src.values():
        d["n_tasks"] = len(d["n_tasks"])
        d["acceptance_rate"] = d["n_accepted"] / max(d["n_rollouts"], 1)
        d["mean_accuracy"] = d.pop("acc_sum") / max(d["n_rollouts"], 1)

    rate = len(accepted) / max(n_roll, 1)
    if n_roll == 0:
        verdict = "empty"
    elif not accepted:
        verdict = "uninformative_zero"
    elif rate < cfg.min_informative_rate:
        verdict = "uninformative_low"
    elif rate > cfg.max_informative_rate:
        verdict = "uninformative_high"
    else:
        verdict = "ok"

    # Counterfactual acceptance under every other signal at its own default
    # threshold.  This is the number that tells the rebuttal whether the
    # accuracy-accepted and composite-accepted variants would even differ.
    counterfactual = {}
    for sig in SIGNAL_NAMES:
        thr = DEFAULT_THRESHOLDS[sig]
        n = sum(1 for r in records if is_accepted(r["star"]["scores"], sig, thr))
        counterfactual[sig] = {"threshold": thr, "n_accepted": n,
                               "acceptance_rate": n / max(n_roll, 1)}

    return {
        "generated_at": datetime.now().isoformat(),
        "config": asdict(cfg),
        "accept_on": cfg.accept_on,
        "accept_threshold": threshold,
        "accept_signal_is_accuracy_only": cfg.accept_on in ACCURACY_ONLY_SIGNALS,
        "n_tasks": n_tasks,
        "n_rollouts": n_roll,
        "n_accepted": len(accepted),
        "acceptance_rate": rate,
        # Coverage: the fraction of TASKS with at least one accepted rollout.
        # STaR's dataset size is bounded by this, not by the rollout-level rate.
        "task_coverage": len(covered) / max(n_tasks, 1),
        "n_tasks_covered": len(covered),
        "threshold_verdict": verdict,
        "score_distribution": {k: _quantiles(v) for k, v in sorted(per_signal_vals.items())},
        "score_histogram": {k: _histogram(v) for k, v in sorted(per_signal_vals.items())
                            if k in SIGNAL_NAMES},
        "acceptance_if_accepted_on": counterfactual,
        "by_source_domain": by_src,
        "n_errors": sum(1 for r in records if r.get("error")),
        # Rollouts the ENVIRONMENT's graders produced no correctness signal for.
        # These are not wrong answers, they are unscored ones, and they are
        # counted in the acceptance denominator on purpose — an unscored rollout
        # contributes no training data either way, so hiding it would inflate
        # the reported acceptance rate.  Known cause on this pool:
        # agent_runner._compute_qa_accuracy calls .get on task["options"], which
        # is null (not {}) for 142/3390 train and 20/850 test tasks — all
        # multimodal_vqa with a <=2-character gold answer.  It returns early on
        # a MATCH, so the AttributeError fires only on INCORRECT rollouts;
        # run_task catches it, so the trajectory survives but reward_details is
        # never computed.  Accuracy-based acceptance is therefore unaffected
        # (those rollouts are wrong and would be rejected anyway); the composite
        # and dimension means lose them.  See baselines/STAR_PORT.md
        # § "Upstream defects".  --accept-on verl_acc is immune: that signal is
        # computed here, from the transcript.
        "n_unscored_by_env_grader": sum(
            1 for r in records
            if not r["star"]["scores"]
            or r["star"]["scores"].get("qa_accuracy", float("nan"))
            != r["star"]["scores"].get("qa_accuracy", float("nan"))),
        "budget": budget_report(n_tasks, cfg.samples_per_task, iterations_planned),
        "reward_fn_env": {k: os.environ.get(k, "")
                          for k in ("COSINE_REWARD", "DEGENERATE_FILTER", "DEGENERATE_EXCLUDE",
                                    "DEGENERATE_GIBBERISH", "DEGENERATE_NGRAM_THRESHOLD")},
    }


def budget_report(n_tasks: int, samples_per_task: int, iterations: int) -> dict:
    """Rollout budget, next to the RL arms' rollout budget.

    The RL arms in ``runs/train_hcgym.slurm`` see
    ``|pool| x rollout.n x total_epochs`` rollouts.  A STaR arm at
    ``samples_per_task = rollout.n`` and ``iterations = total_epochs`` sees
    exactly the same number, which is why those are the defaults.  Both figures
    are the CONFIGURED budgets; a preempted RL run consumes fewer.
    """
    rl_rollout_n = int(os.environ.get("ROLLOUT_N", "3"))
    rl_epochs = int(os.environ.get("TOTAL_EPOCHS", "3"))
    star_total = n_tasks * samples_per_task * iterations
    rl_total = n_tasks * rl_rollout_n * rl_epochs
    return {
        "star_rollouts_per_iteration": n_tasks * samples_per_task,
        "star_iterations": iterations,
        "star_rollouts_total": star_total,
        "rl_reference": {
            "source": "runs/train_hcgym.slurm (ROLLOUT_N x TOTAL_EPOCHS over the same pool)",
            "rollout_n": rl_rollout_n,
            "total_epochs": rl_epochs,
            "rollouts_total": rl_total,
        },
        "ratio_star_over_rl": (star_total / rl_total) if rl_total else None,
        "matched": star_total == rl_total,
    }


# ══════════════════════════════════════════════════════════════════════
#  Selection & writing
# ══════════════════════════════════════════════════════════════════════


def select_accepted(records: list[dict], cfg: GenConfig) -> list[dict]:
    """Accepted rollouts, capped per task.

    STaR generates ONE rationale per problem per iteration.  ``k`` samples are
    drawn here only to match the RL arms' rollout budget, so keeping all of
    them would additionally re-weight the training set toward whichever tasks
    happen to be easy (an easy task contributes k copies, a hard one at most
    1).  ``--max-accepted-per-task 0`` keeps everything for the ablation.
    """
    accepted = [r for r in records if r["star"]["accepted"]]
    if cfg.max_accepted_per_task <= 0:
        return sorted(accepted, key=lambda r: (r["task_id"], r["star"]["sample_idx"]))
    by_task: dict[str, list[dict]] = {}
    for r in accepted:
        by_task.setdefault(r["task_id"], []).append(r)
    out = []
    for tid in sorted(by_task):
        ranked = sorted(
            by_task[tid],
            key=lambda r: (-_safe(r["star"]["scores"].get(cfg.accept_on)),
                           -_safe(r["star"]["scores"].get("composite_reward")),
                           r["star"]["sample_idx"]),
        )
        out.extend(ranked[: cfg.max_accepted_per_task])
    return out


def _safe(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return -1e9
    return f if f == f else -1e9


def write_outputs(records: list[dict], selected: list[dict], stats: dict,
                  out_dir: Path, cfg: GenConfig) -> dict:
    """Write the trajectory dir, the SFT jsonl, the score log and the stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectory_dir"
    paths = {"out_dir": str(out_dir), "stats": str(out_dir / "gen_stats.json")}

    # Per-rollout score log — every rollout, accepted or not.  This is what
    # makes --refilter-from possible without new rollouts.
    with (out_dir / "all_scores.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda r: (r["task_id"], r["star"]["sample_idx"])):
            f.write(json.dumps({
                "task_id": r["task_id"],
                "sample_idx": r["star"]["sample_idx"],
                "source_domain": r["source_domain"],
                "accepted": r["star"]["accepted"],
                "total_turns": r["total_turns"],
                "error": bool(r.get("error")),
                "scores": r["star"]["scores"],
            }, ensure_ascii=False) + "\n")
    paths["all_scores"] = str(out_dir / "all_scores.jsonl")

    if cfg.keep_rejected:
        raw_dir = out_dir / "rollouts"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for r in records:
            body = {k: v for k, v in r.items() if k != "_sft_messages"}
            body["_sft_messages"] = r.get("_sft_messages", [])
            (raw_dir / f"task_{r['task_id']}__k{r['star']['sample_idx']}.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        paths["rollouts"] = str(raw_dir)

    if not cfg.eval_only:
        if traj_dir.exists():
            shutil.rmtree(traj_dir)
        traj_dir.mkdir(parents=True, exist_ok=True)
        for r in selected:
            body = {k: v for k, v in r.items() if k != "_sft_messages"}
            (traj_dir / f"task_{r['task_id']}__k{r['star']['sample_idx']}.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        paths["trajectory_dir"] = str(traj_dir)

        sft_path = out_dir / "star_sft.jsonl"
        with sft_path.open("w", encoding="utf-8") as f:
            for r in selected:
                msgs = r.get("_sft_messages") or []
                if len(msgs) < 2:
                    continue
                f.write(json.dumps({
                    "messages": msgs,
                    "metadata": {
                        "source": "star",
                        "task_id": r["task_id"],
                        "domain": r["domain"],
                        "source_domain": r["source_domain"],
                        "accept_on": r["star"]["accept_on"],
                        "accept_threshold": r["star"]["accept_threshold"],
                        "scores": r["star"]["scores"],
                        "rationalized": r["star"].get("rationalized", False),
                    },
                }, ensure_ascii=False) + "\n")
        paths["sft_path"] = str(sft_path)

    stats = dict(stats)
    stats["n_selected"] = len(selected)
    stats["paths"] = paths
    (out_dir / "gen_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return stats


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════


def run_generation(cfg: GenConfig, tasks: Optional[list[dict]] = None,
                   generate_fn: Optional[Callable] = None,
                   iterations_planned: int = 1) -> dict:
    """One full generation pass.  Returns the stats dict.

    ``generate_fn`` injects a policy directly and is how the no-GPU tests drive
    the REAL ``AgentRunner.run_task`` loop, the REAL parser and the REAL
    scorers with a scripted model.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = load_pool(cfg.pool, cfg.split, cfg.limit, cfg.seed)
    n_tasks = len(tasks)

    logger.info(f"[star] pool={cfg.pool}/{cfg.split} tasks={n_tasks} "
                f"k={cfg.samples_per_task} T={cfg.temperature} max_turns={cfg.max_turns}")
    bud = budget_report(n_tasks, cfg.samples_per_task, iterations_planned)
    logger.info(f"[star] budget: {bud['star_rollouts_total']} STaR rollouts vs "
                f"{bud['rl_reference']['rollouts_total']} RL rollouts "
                f"(matched={bud['matched']})")

    runner = _make_runner(cfg, out_dir / "runner_logs")
    if generate_fn is not None:
        runner.generate = generate_fn
    spy = _MessageSpy(runner.generate)
    runner.generate = spy
    envs = _EnvCache(cfg.max_turns)

    def work(job):
        task, k = job
        try:
            return rollout_one(runner, spy, envs, task, k, cfg)
        except Exception as exc:  # a single bad task must not kill the pass
            logger.error(f"[star] rollout failed task={task.get('id')} k={k}: {exc}")
            return {
                "task_id": task.get("id", "?"), "domain": gym_domain_for(task),
                "source_domain": task.get("_source_domain", "unknown"),
                "total_turns": 0, "action_score": 0.0, "final_reward": 0.0,
                "completed": False, "error": str(exc), "turns": [],
                "format_adherence": {}, "_sft_messages": [],
                "star": {"sample_idx": k, "accepted": False, "accept_on": cfg.accept_on,
                         "accept_threshold": cfg.resolved_threshold(),
                         "temperature": cfg.temperature, "max_turns": cfg.max_turns,
                         "scores": {}, "rationalized": False},
            }

    # One gym domain at a time.  ``run_task`` reads ``runner.config.domain`` to
    # build the system prompt, and the runner is shared across worker threads,
    # so mutating it per rollout would let one task's domain prompt leak into
    # another's.  Grouping by domain makes it a per-PHASE constant instead.
    by_domain: dict[str, list[dict]] = {}
    for t in tasks:
        by_domain.setdefault(gym_domain_for(t), []).append(t)

    records: list[dict] = []
    for domain in sorted(by_domain):
        jobs = [(t, k) for t in by_domain[domain] for k in range(cfg.samples_per_task)]
        logger.info(f"[star] domain={domain}: {len(by_domain[domain])} tasks -> {len(jobs)} rollouts")
        runner.config.domain = domain
        if cfg.workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                records.extend(pool.map(work, jobs))
        else:
            records.extend(work(job) for job in jobs)

    stats = summarize(records, cfg, n_tasks, iterations_planned)
    selected = [] if cfg.eval_only else select_accepted(records, cfg)
    stats = write_outputs(records, selected, stats, out_dir, cfg)
    report(stats)
    return stats


def refilter(cfg: GenConfig, source_dir: Path) -> dict:
    """Re-run the acceptance filter over a finished pass — no new rollouts.

    Reads the cached rollouts, re-decides acceptance under the current
    ``--accept-on`` / ``--accept-threshold`` and writes a fresh output dir.
    This is what makes "report both signals" cost a file read.
    """
    raw_dir = Path(source_dir) / "rollouts"
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_dir} not found — the source pass must have been run with "
            "--keep-rejected (the default) for re-filtering to be possible.")
    records = []
    for p in sorted(raw_dir.glob("task_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["star"]["accepted"] = is_accepted(r["star"].get("scores", {}),
                                            cfg.accept_on, cfg.resolved_threshold())
        r["star"]["accept_on"] = cfg.accept_on
        r["star"]["accept_threshold"] = cfg.resolved_threshold()
        records.append(r)
    n_tasks = len({r["task_id"] for r in records})
    stats = summarize(records, cfg, n_tasks, 1)
    stats["refiltered_from"] = str(source_dir)
    selected = [] if cfg.eval_only else select_accepted(records, cfg)
    cfg2 = GenConfig(**{**asdict(cfg), "keep_rejected": False})
    stats = write_outputs(records, selected, stats, Path(cfg.out_dir), cfg2)
    report(stats)
    return stats


def report(stats: dict) -> None:
    """Print the numbers the rebuttal needs, loudly."""
    print("=" * 78)
    print(f"  STaR generation — accept_on={stats['accept_on']} "
          f"threshold={stats['accept_threshold']}"
          f"{'' if stats['accept_signal_is_accuracy_only'] else '   [NOT accuracy-only]'}")
    print("=" * 78)
    print(f"  tasks              {stats['n_tasks']}")
    print(f"  rollouts           {stats['n_rollouts']}")
    print(f"  accepted           {stats['n_accepted']}  ({stats['acceptance_rate']:.1%})")
    print(f"  selected for SFT   {stats.get('n_selected', 0)}")
    print(f"  task coverage      {stats['n_tasks_covered']}/{stats['n_tasks']} "
          f"({stats['task_coverage']:.1%})")
    print(f"  errors             {stats['n_errors']}")
    unscored = stats.get("n_unscored_by_env_grader", 0)
    if unscored:
        print(f"  UNSCORED by env    {unscored}  "
              f"({unscored / max(stats['n_rollouts'], 1):.1%} of rollouts had no "
              f"correctness signal from the GYM graders;")
        print("                     they stay in the acceptance denominator. "
              "--accept-on verl_acc is immune.)")
    print("-" * 78)
    print("  score distribution (mean / p50 / max)")
    for sig in SIGNAL_NAMES:
        d = stats["score_distribution"].get(sig)
        if d:
            print(f"    {sig:<18} {d['mean']:>8.3f} {d['p50']:>8.3f} {d['max']:>8.3f}")
    print("-" * 78)
    print("  acceptance if accepted on ...")
    for sig, d in stats["acceptance_if_accepted_on"].items():
        mark = "  <- used" if sig == stats["accept_on"] else ""
        print(f"    {sig:<18} thr={d['threshold']:<5} "
              f"{d['n_accepted']:>6} ({d['acceptance_rate']:.1%}){mark}")
    print("-" * 78)
    b = stats["budget"]
    print(f"  rollout budget     STaR {b['star_rollouts_total']} vs "
          f"RL {b['rl_reference']['rollouts_total']}  matched={b['matched']}")
    v = stats["threshold_verdict"]
    if v != "ok":
        print("!" * 78)
        print(f"!! UNINFORMATIVE THRESHOLD ({v}): acceptance rate "
              f"{stats['acceptance_rate']:.3%} is at an extreme.")
        print("!! Near 0 there is nothing to fine-tune on; near 1 the filter is not")
        print("!! filtering and STaR degenerates into plain self-imitation.")
        print("!! Re-choose --accept-on / --accept-threshold before reporting this arm.")
        print("!" * 78)
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="STaR generation pass (reimplementation)")
    p.add_argument("--model", default="mock-policy")
    p.add_argument("--backend", default="mock",
                   choices=["sglang", "transformers", "vllm", "mock"])
    p.add_argument("--server-url", default=None)
    p.add_argument("--pool", default="full_4modality_clean")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-task", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--prompt-mode", default="default",
                   choices=["default", "strong_tool", "react"])
    p.add_argument("--accept-on", default="qa_accuracy", choices=list(SIGNAL_NAMES))
    p.add_argument("--accept-threshold", type=float, default=None)
    p.add_argument("--max-accepted-per-task", type=int, default=1)
    p.add_argument("--min-informative-rate", type=float, default=0.02)
    p.add_argument("--max-informative-rate", type=float, default=0.98)
    p.add_argument("--allow-uninformative", action="store_true",
                   help="exit 0 even when the acceptance rate is degenerate")
    p.add_argument("--iterations-planned", type=int, default=1,
                   help="only affects the printed budget accounting")
    p.add_argument("--eval-only", action="store_true",
                   help="score a split and write stats; write no training data")
    p.add_argument("--no-keep-rejected", action="store_true")
    p.add_argument("--refilter-from", default=None,
                   help="re-filter a finished pass instead of rolling out")
    p.add_argument("--out-dir", required=True)
    return p


def cfg_from_args(a) -> GenConfig:
    return GenConfig(
        model=a.model, backend=a.backend, server_url=a.server_url,
        temperature=a.temperature, top_p=a.top_p, max_new_tokens=a.max_new_tokens,
        no_think=a.no_think, prompt_mode=a.prompt_mode,
        pool=a.pool, split=a.split, limit=a.limit, seed=a.seed,
        samples_per_task=a.samples_per_task, max_turns=a.max_turns, workers=a.workers,
        accept_on=a.accept_on, accept_threshold=a.accept_threshold,
        max_accepted_per_task=a.max_accepted_per_task,
        min_informative_rate=a.min_informative_rate,
        max_informative_rate=a.max_informative_rate,
        allow_uninformative=a.allow_uninformative,
        out_dir=a.out_dir, eval_only=a.eval_only, keep_rejected=not a.no_keep_rejected,
    )


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = cfg_from_args(a)
    if a.refilter_from:
        stats = refilter(cfg, Path(a.refilter_from))
    else:
        stats = run_generation(cfg, iterations_planned=a.iterations_planned)
    if stats["threshold_verdict"] != "ok" and not cfg.allow_uninformative and not cfg.eval_only:
        # Exit non-zero: an uninformative threshold must stop a pipeline, not
        # scroll past in a log.
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
