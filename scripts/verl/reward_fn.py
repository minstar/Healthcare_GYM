"""Reward function for BIOAgents medical GRPO on veRL.

Supports optional cosine length-scaling reward (Yeo et al., 2025,
"Demystifying Long Chain-of-Thought Reasoning in LLMs", arXiv:2502.03373).
Enable via environment variable COSINE_REWARD=1.

`compute_score` returns a dict, not a scalar.  ``score`` is the shaped reward
that drives the GRPO advantage; every other key is a per-sample diagnostic that
verl files into ``reward_extra_info`` and logs.  The point of the extra keys is
that ``score`` is NOT comparable across arms — the cosine arms shape it by
response length, so an identical correct answer scores 1.1000 without cosine and
anywhere in 0.87-1.20 with it.  ``acc`` is the arm-comparable readout.

Stdlib imports only: verl loads this file standalone via
`verl.utils.import_utils.load_extern_object`, outside any package.
"""
import json
import math
import os
import re
from typing import Any, Optional

_DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
_debug_count = 0

# ── Degenerate response detection ────────────────────────────────────
# Detect repetitive/degenerate responses and assign penalty reward.
# Enable via DEGENERATE_FILTER=1 (default: disabled for backwards compat)
DEGENERATE_FILTER_ENABLED = os.environ.get("DEGENERATE_FILTER", "") == "1"
DEGENERATE_REWARD = float(os.environ.get("DEGENERATE_REWARD", "-1.0"))
DEGENERATE_NGRAM_THRESHOLD = float(os.environ.get("DEGENERATE_NGRAM_THRESHOLD", "0.3"))
DEGENERATE_MIN_LENGTH = int(os.environ.get("DEGENERATE_MIN_LENGTH", "200"))
# DEGENERATE_EXCLUDE=1: use sentinel reward (-999.0) to completely exclude
# degenerate responses from GRPO advantage computation (no gradient at all).
# Without this, degenerate responses with -1.0 reward still produce large
# gradients that cause grad_norm explosion after step 50.
DEGENERATE_EXCLUDE = os.environ.get("DEGENERATE_EXCLUDE", "") == "1"
DEGENERATE_SENTINEL = -999.0  # Must match value in core_algos.py
# DEGENERATE_GIBBERISH=1: enhanced gibberish detection that catches non-repetitive
# garbage (random multilingual tokens) which bypasses the n-gram filter.
# v28 showed model can collapse into gibberish that has unique n-grams.
DEGENERATE_GIBBERISH = os.environ.get("DEGENERATE_GIBBERISH", "") == "1"


def _strip_tool_markup(text: str) -> str:
    """Strip tool/turn markup to isolate unique model-generated content."""
    # Remove tool responses (environment output, not model text)
    cleaned = re.sub(r"<tool_response>.*?</tool_response>", " ", text, flags=re.DOTALL)
    # Remove tool call XML tags
    cleaned = re.sub(r"</?(?:tool_call|function=[^>]*)>", " ", cleaned)
    # Remove JSON arguments in tool calls
    cleaned = re.sub(r"\{[^}]{0,500}\}", " ", cleaned)
    # Remove turn markers that repeat in every multi-turn exchange
    cleaned = re.sub(r"</?think>", " ", cleaned)
    cleaned = re.sub(r"\b(user|assistant|system)\b", " ", cleaned)
    # Remove common filler phrases that naturally repeat across turns
    cleaned = re.sub(r"I'm here and ready to help[.!]?", " ", cleaned)
    cleaned = re.sub(r"What would you like to ask or discuss\?", " ", cleaned)
    cleaned = re.sub(r"please (?:try )?typ(?:e|ing) (?:it )?(?:out )?again", " ", cleaned, flags=re.IGNORECASE)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_degenerate_response(text: str) -> bool:
    """Detect degenerate/repetitive responses.

    Strips tool markup first to avoid false positives from repeated XML
    structure in multi-turn tool-use responses.

    Checks for:
    1. High ratio of repeated n-grams (n=4) in model-generated text
    2. Exact phrase repetition loops
    3. Very short responses with no tool calls (model gave up)
    4. Multi-turn degenerate: excessive 'assistant' token repetition
    5. Gibberish: high Unicode diversity (multilingual garbage tokens)
    """
    if len(text) < DEGENERATE_MIN_LENGTH:
        if "<function=" not in text and "<tool_call>" not in text:
            return True
        return False

    # Check 4: Multi-turn assistant token repetition
    # Pattern: model generates correct answer then fills with "assistant\nassistant\n..."
    assistant_count = text.count("\nassistant\n")
    if assistant_count > 20:
        return True

    # Check 5: Gibberish detection — high ratio of non-ASCII or high script diversity
    latter_half = text[len(text) // 2:]
    if len(latter_half) > 500:
        non_ascii = sum(1 for c in latter_half if ord(c) > 127)
        non_ascii_ratio = non_ascii / len(latter_half)
        if non_ascii_ratio > 0.08:
            return True
        # Also check: many unique scripts in latter half = gibberish (random multilingual tokens)
        words = latter_half.split()
        if len(words) > 100:
            # Sample 200 words, count how many have non-ASCII characters
            sample = words[::max(1, len(words) // 200)]
            mixed_script = sum(1 for w in sample if any(ord(c) > 127 for c in w))
            if mixed_script / len(sample) > 0.15:
                return True

    # Check 6 (v29): Enhanced gibberish detection for DEGENERATE_GIBBERISH mode
    # v28 showed model can produce ASCII word salad that passes n-gram check
    # (unique random words = high n-gram diversity, not repetitive)
    # Detection: very long response + no coherent structure + low tool usage
    if DEGENERATE_GIBBERISH and len(text) > 30000:
        # Long response with no Answer pattern and no submit_answer = likely gibberish
        has_answer = bool(re.search(r"Answer:\s*\S", text))
        has_submit = "submit_answer" in text
        has_tool_response = "</tool_response>" in text
        # Check word length distribution — gibberish has many short random fragments
        last_quarter = text[3 * len(text) // 4:]
        lq_words = last_quarter.split()
        if len(lq_words) > 200:
            # Gibberish signal: high ratio of very short words (1-3 chars)
            short_words = sum(1 for w in lq_words[:500] if len(w) <= 3)
            short_ratio = short_words / min(500, len(lq_words))
            # Also check: no medical/English structure (no common words)
            common_medical = {"the", "is", "of", "and", "in", "to", "a", "for", "with", "that", "this", "patient", "diagnosis"}
            common_count = sum(1 for w in lq_words[:500] if w.lower() in common_medical)
            common_ratio = common_count / min(500, len(lq_words))
            # Gibberish: many short words + few common English words + no answer/tool
            if short_ratio > 0.4 and common_ratio < 0.05 and not has_answer and not has_submit:
                return True
            # Also: non-ASCII in last quarter even at lower threshold
            non_ascii_lq = sum(1 for c in last_quarter[:2000] if ord(c) > 127)
            if non_ascii_lq / min(2000, len(last_quarter)) > 0.04 and not has_tool_response:
                return True

    # Strip tool markup before checking repetition
    cleaned = _strip_tool_markup(text)

    # Check the latter half for repetition
    half = cleaned[len(cleaned) // 2:]
    words = half.lower().split()

    if len(words) < 30:
        return False

    # 4-gram repetition ratio on cleaned text
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return False
    unique_ratio = len(set(ngrams)) / len(ngrams)
    # Low unique ratio = high repetition (threshold: <30% unique = degenerate)
    # Raised from 0.15 to catch more subtle repetition patterns
    if unique_ratio < 0.30:
        return True

    # Check for exact phrase loops (same phrase repeated many times)
    for phrase_len in [5, 10, 20]:
        if len(words) < phrase_len * 6:
            continue
        phrases = [tuple(words[i:i+phrase_len]) for i in range(0, len(words) - phrase_len, phrase_len)]
        if phrases:
            most_common = max(set(phrases), key=phrases.count)
            if phrases.count(most_common) >= len(phrases) * 0.6:
                return True

    return False

# ── Multi-dimensional clinical reward (composite) ────────────────────
# The paper describes the environment's reward as multi-dimensional --
# accuracy, format, process, safety, coherence, assertion -- and
# `bioagents.evaluation.rewards.compute_composite_reward` implements exactly
# that.  Until this knob existed nothing in the RL path called it: every
# published run optimized `accuracy + 0.1*format_bonus - 0.2*n_invalid`, so the
# composite was an evaluation-time quantity only.  This makes it selectable as
# a TRAINING objective so its dimensions can be ablated by intervention.
#
# Set HCGYM_REWARD_WEIGHTS to a JSON object, e.g.
#   HCGYM_REWARD_WEIGHTS='{"accuracy":0.25,"format":0.10,"process":0.20,
#                          "safety":0.20,"coherence":0.10,"assertion":0.15}'
# Omitted dimensions get weight 0.0 and contribute nothing (compute_composite_
# reward falls back symmetrically), so a leave-one-out arm is expressed by
# dropping a key.
#
# What this replaces and what it does NOT: the composite substitutes for
# `base_reward` at exactly the point the cosine reward substitutes for it.  The
# invalid-tool penalty and the degenerate filter stay in force for every arm,
# because they are shared safety rails rather than reward dimensions -- leaving
# them on is what makes the arms differ ONLY in reward composition.  `acc` and
# `acc_partial` are captured before this and remain unshaped, so cross-arm
# comparison is still done on a quantity no arm's weights can move.
_COMPOSITE_DIMS = ("accuracy", "format", "process", "safety", "coherence", "assertion")


def _parse_reward_weights():
    raw = os.environ.get("HCGYM_REWARD_WEIGHTS", "").strip()
    if not raw:
        return None
    try:
        w = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"HCGYM_REWARD_WEIGHTS is not valid JSON: {e}") from e
    if not isinstance(w, dict) or not w:
        raise ValueError("HCGYM_REWARD_WEIGHTS must be a non-empty JSON object")
    # A typo in a slurm script would otherwise silently define a DIFFERENT arm
    # than the one its name claims, and the run would look successful.
    unknown = sorted(set(w) - set(_COMPOSITE_DIMS))
    if unknown:
        raise ValueError(f"HCGYM_REWARD_WEIGHTS has unknown dimensions {unknown}; known: {list(_COMPOSITE_DIMS)}")
    bad = {k: v for k, v in w.items() if not isinstance(v, (int, float)) or isinstance(v, bool)}
    if bad:
        raise ValueError(f"HCGYM_REWARD_WEIGHTS values must be numbers, got {bad}")
    return {k: float(v) for k, v in w.items()}


COMPOSITE_WEIGHTS = _parse_reward_weights()
COMPOSITE_REWARD_ENABLED = COMPOSITE_WEIGHTS is not None

if COMPOSITE_REWARD_ENABLED:
    # Resolve the import NOW rather than on the first scored rollout. The
    # composite lives in the `bioagents` package, which reaches the verl workers
    # only through PYTHONPATH; if that is wrong the lazy import would not fail
    # until step 1, i.e. after the whole cluster job has spent its setup time,
    # and it would surface as a ray ActorDiedError rather than an ImportError.
    from bioagents.evaluation.rewards import compute_composite_reward as _  # noqa: F401


def _composite_base_reward(solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """The composite total, for use in place of base_reward.

    Scored once over the whole trajectory, which is the only thing verl hands
    the reward function.  `expected_actions` and `nl_assertions` are absent from
    the training parquet -- its extra_info carries only correct_answer, domain,
    has_options, index, options, raw_answer, split, task_id -- so those inputs
    are empty here.  That is the real condition inside training, not a
    simplification, and it is why the assertion dimension sits at its neutral
    value and the tool half of the process dimension has nothing to compare
    against.  Both facts are measured in
    scripts/rebuttal/decompose_reward_signal.py.
    """
    from bioagents.evaluation.rewards import compute_composite_reward

    result = compute_composite_reward(
        response=solution_str,
        correct_answer=ground_truth or "",
        reference_text=extra_info.get("raw_answer", "") or "",
        tool_call_log=[],
        expected_actions=[],
        nl_assertions=None,
        turn_idx=0,
        is_final=True,
        weights=COMPOSITE_WEIGHTS,
        task_domain=extra_info.get("domain", "") or "",
    )
    return float(result["total"])


# ── Cosine length-scaling reward (arXiv:2502.03373) ──────────────────
# CosFn(t, T, η_min, η_max) = η_min + ½(η_max − η_min)(1 + cos(tπ/T))
# Goes from η_max at t=0 to η_min at t=T.
COSINE_REWARD_ENABLED = os.environ.get("COSINE_REWARD", "") == "1"

# Both of these REPLACE base_reward.  Whichever ran second would silently win
# and the arm would not be the one its name claims, so refuse the combination
# at import time rather than resolve it by precedence.
if COSINE_REWARD_ENABLED and COMPOSITE_REWARD_ENABLED:
    raise ValueError(
        "COSINE_REWARD=1 and HCGYM_REWARD_WEIGHTS are mutually exclusive: both replace "
        "the base reward. Pick one per arm."
    )

# Max response length in tokens (must match data.max_response_length)
COSINE_L_MAX = int(os.environ.get("COSINE_L_MAX", "12288"))
# Chars-per-token estimate for converting solution_str length to tokens
COSINE_CHARS_PER_TOKEN = float(os.environ.get("COSINE_CHARS_PER_TOKEN", "5.0"))

# Reward at L_gen=0 / L_gen=L_max for correct answers
COSINE_R0_CORRECT = float(os.environ.get("COSINE_R0_CORRECT", "1.1"))
COSINE_RL_CORRECT = float(os.environ.get("COSINE_RL_CORRECT", "0.7"))
# Reward at L_gen=0 / L_gen=L_max for wrong answers
COSINE_R0_WRONG = float(os.environ.get("COSINE_R0_WRONG", "0.0"))
COSINE_RL_WRONG = float(os.environ.get("COSINE_RL_WRONG", "-0.3"))
# Penalty when response hits max length (clipped)
COSINE_R_EXCEED = float(os.environ.get("COSINE_R_EXCEED", "-0.5"))


def _cosine_fn(t: float, T: float, eta_min: float, eta_max: float) -> float:
    """Cosine annealing: eta_max at t=0, eta_min at t=T."""
    t = max(0.0, min(t, T))
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(t * math.pi / T))


def cosine_length_reward(is_correct: bool, response_len_chars: int) -> float:
    """Compute cosine length-scaling reward.

    Adapts Yeo et al. (2025) for multi-turn agentic setting:
    - Correct short → highest reward (1.1)
    - Correct long → reduced reward (0.7)
    - Wrong short → zero (0.0)
    - Wrong long → penalty (-0.3)
    - Exceeded max length → strong penalty (-0.5)
    """
    est_tokens = response_len_chars / COSINE_CHARS_PER_TOKEN

    if est_tokens >= COSINE_L_MAX:
        return COSINE_R_EXCEED

    if is_correct:
        return _cosine_fn(est_tokens, COSINE_L_MAX, COSINE_RL_CORRECT, COSINE_R0_CORRECT)
    else:
        return _cosine_fn(est_tokens, COSINE_L_MAX, COSINE_RL_WRONG, COSINE_R0_WRONG)

# Valid tool names from tool_config_consolidated.yaml (25 tools)
VALID_TOOLS = frozenset({
    "think", "submit_answer",
    "search_pubmed", "search_medical_wiki", "search_medical_literature",
    "retrieve_evidence",
    "analyze_answer_options", "get_differential_diagnosis",
    "check_diagnostic_criteria", "compare_treatments",
    "get_drug_info", "check_interaction", "check_medication_safety",
    "get_patient_info", "get_patient_records",
    "get_vital_signs", "get_lab_results", "order_test",
    "analyze_medical_image", "get_image_report", "search_imaging_knowledge",
    "calculate_clinical_score", "search_clinical_guidelines",
    "record_clinical_action", "perform_assessment",
    "assess_obstetric_status", "get_ed_status",
})

# Penalty per invalid tool call (tool name not in VALID_TOOLS)
INVALID_TOOL_PENALTY = 0.2


def tool_call_names(solution_str: str) -> list[str]:
    """Return the (stripped) name of every tool call in the response."""
    return [name.strip() for name in re.findall(r"<function=([^>]+)>", solution_str)]


def count_invalid_tool_calls(solution_str: str) -> int:
    """Count tool calls to names not in VALID_TOOLS."""
    return sum(1 for name in tool_call_names(solution_str) if name not in VALID_TOOLS)


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract answer letter from model response.

    Uses the LAST match in multi-turn responses to avoid picking up
    intermediate tool responses or earlier reasoning attempts.
    """
    # Pattern 1: "Answer: X" — use last match (critical for multi-turn)
    matches = re.findall(r"Answer:\s*([A-E])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Pattern 2: "the answer is X" — use last match
    matches = re.findall(r"the answer is\s*([A-E])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Pattern 3: PARENTHESIZED letter at end, e.g. "(D)" or "(D)." — this is an
    # unambiguous answer format. A bare trailing letter is intentionally NOT matched:
    # "Vitamin D" / "type A" are indistinguishable from a gold "D"/"A" and matching
    # them flipped the binary training reward.
    match = re.search(r"\(([A-E])\)[.)]?\s*$", text.strip())
    if match:
        return match.group(1).upper()

    return None


def _extract_final_answer_span_ex(text: str, window: int = 600) -> tuple[str, bool]:
    """`_extract_final_answer_span`, plus whether the span came from a marker.

    The bool is False when the span is the tail-window fallback, i.e. the model
    never emitted a locatable final answer and the score is being computed off
    whatever happened to land at the end of the transcript.  Reported as the
    `answer_found` metric.
    """
    if not text:
        return "", False
    # After the last submit_answer(...) payload, if present.
    m = list(re.finditer(r'submit_answer[^\{]*\{(.*?)\}', text, re.DOTALL | re.IGNORECASE))
    if m:
        return m[-1].group(1), True
    # After the last explicit answer marker.
    m = list(re.finditer(r'(?:final\s+answer|answer)\s*[:\-]\s*(.+)', text, re.IGNORECASE))
    if m:
        return m[-1].group(1), True
    # Fallback: tail window (final reasoning usually lands here).
    return text[-window:], False


def _extract_final_answer_span(text: str, window: int = 600) -> str:
    """Return the model's final answer text for open-ended scoring.

    Prefers the content after the last submit_answer tool call or an explicit
    'Answer:'/'Final Answer:' marker; otherwise falls back to the tail of the
    response. This keeps verbosity/tool-echo out of the overlap score.
    """
    return _extract_final_answer_span_ex(text, window)[0]


# ── Metric payload ───────────────────────────────────────────────────
# verl unpacks a dict return as: score -> the scalar that drives the advantage,
# every other key -> reward_extra_info, which is logged and dumped per sample.
# See verl/experimental/reward_loop/reward_manager/naive.py::run_single.
#
# TWO HARD CONSTRAINTS, both learned from reading the consumers:
#
#  1. EVERY return point must carry the SAME keys.  reward_loop.py:358 takes the
#     key list from the FIRST sample of the batch and then indexes every other
#     sample with it, so a key that is present on sample 0 but missing on
#     sample 7 is a KeyError that kills training mid-step.  Build the payload in
#     one place (_result) so no return path can drift.
#
#  2. EVERY value must be a Python float.  The values are round-tripped through
#     np.array(...) (reward_loop.py:361) and then json.dumps() in
#     ray_trainer._dump_generations.  np.int64 is not JSON-serialisable, so an
#     int-valued metric crashes the rollout dump; np.float64 subclasses float
#     and is fine.  Hence n_tool_calls et al. are floats, not ints.
#
# All keys except `score` are arm-invariant by construction: none of them read
# COSINE_* and none of them depend on response length.

def _result(
    score: float,
    *,
    acc: float,
    acc_partial: float,
    has_options: bool,
    answer_found: bool,
    n_tool_calls: int,
    n_invalid_tool_calls: int,
    degenerate: bool,
) -> dict[str, float]:
    """Assemble the reward payload.  Sole construction site — see notes above.

    score                 shaped reward; the ONLY value that feeds the advantage.
                          Scale differs per arm (cosine); do not compare across arms.
    acc                   binary correctness, 1.0/0.0.  Never length-shaped, never
                          penalised, never touched by COSINE_*.  THIS is the
                          cross-arm readout, and it is the key verl picks as
                          `core_var` for val metrics (ray_trainer._val_metrics_update).
    acc_partial           unshaped base credit before any bonus/penalty: identical
                          to acc for multiple-choice, token-F1 in [0,1] for
                          open-ended.  Lower-variance version of acc for curves.
    has_options           1.0 multiple-choice / 0.0 open-ended, so acc can be
                          sliced by task type (the eval split is 33% MC / 67% open).
    answer_found          1.0 if the scorer found anything to score (a parseable
                          letter for MC, a non-empty answer span for open-ended).
                          Separates "wrong" from "never answered".
    n_tool_calls          total <function=...> invocations.
    n_invalid_tool_calls  invocations naming a tool outside VALID_TOOLS.  The
                          paper's hallucinated-tool rate is sum(invalid)/sum(total)
                          over a run — emit both counts so that ratio is exact
                          rather than a mean of per-sample ratios.
    has_invalid_tool_call 1.0 if this response hallucinated at least one tool.
    degenerate            1.0 if the degenerate filter FIRED on this response.
                          Necessarily 0.0 when DEGENERATE_FILTER is off, because
                          the detector is not run in that configuration.
    """
    return {
        "score": float(score),
        "acc": float(acc),
        "acc_partial": float(acc_partial),
        "has_options": 1.0 if has_options else 0.0,
        "answer_found": 1.0 if answer_found else 0.0,
        "n_tool_calls": float(n_tool_calls),
        "n_invalid_tool_calls": float(n_invalid_tool_calls),
        "has_invalid_tool_call": 1.0 if n_invalid_tool_calls > 0 else 0.0,
        "degenerate": 1.0 if degenerate else 0.0,
    }


# Keys every payload carries.  Constraint 1 above; asserted by the tests.
METRIC_KEYS = tuple(
    _result(
        0.0, acc=0.0, acc_partial=0.0, has_options=False, answer_found=False,
        n_tool_calls=0, n_invalid_tool_calls=0, degenerate=False,
    ).keys()
)


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, float]:
    """Compute reward for medical QA tasks.

    For MCQA: binary reward (1.0 correct, 0.0 wrong) + format bonus.
    For open-ended: partial credit based on keyword overlap.

    Returns the dict described in `_result`.  `score` reproduces exactly what
    this function returned as a bare float before the dict was introduced.
    """
    if extra_info is None:
        extra_info = {}

    # Parse extra_info if it's a string
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)

    # NOTE — there is deliberately no train/validation branch here.
    #
    # This function used to read `is_validate = extra_info.get("validate", False)`
    # and use it to skip cosine shaping and the degenerate filter during
    # validation.  That flag was ALWAYS False and the branch never once executed
    # in any published run.  verl sets "validate" only in DataProto.meta_info
    # (ray_trainer.py:732,763); the reward manager passes the reward fn the
    # PER-ITEM non_tensor_batch["extra_info"] (reward_manager/naive.py), whose
    # keys come from the dataset parquet and are exactly:
    #   correct_answer, domain, has_options, index, options, raw_answer,
    #   split, task_id
    # The only other contributor is tool_extra_fields, which carries
    # turn_scores / tool_rewards / max_global_steps.  Nothing anywhere sets
    # "validate" on a per-item basis.  The branch was dead code that looked live.
    #
    # It is not resurrected, on purpose.  Validation is scored with exactly the
    # same shaping as training, and cross-arm comparison is done on `acc`, which
    # is unshaped by construction.  Reasons, in order of weight:
    #
    #   * Switching validation to unshaped reward would redefine the metric
    #     verl logs mid-study, so curves from runs already on disk would not be
    #     comparable to new ones.
    #   * It would change the metric for the GRPO baseline too, not just the
    #     cosine arms: turning off `is_validate` shaping also turns off the
    #     degenerate filter, and with DEGENERATE_EXCLUDE=1 a degenerate rollout
    #     currently contributes -999.0 to the validation mean (core_algos only
    #     excludes the sentinel from the training advantage, not from val
    #     aggregation).  "Fix" the flag and every arm's val reward moves.
    #   * `acc` is a cleaner quantity than "reward with shaping conditionally
    #     disabled" would have been anyway: the latter still carries the +0.1
    #     format bonus and the -0.2/call hallucinated-tool penalty, so two arms
    #     with equal accuracy but different tool hygiene would still diverge.
    #
    # For the record, reviving it would NOT have required touching verl:
    # extra_info["split"] is already "train"/"test" per row.  The flag stays
    # dead because of the metric-stability argument above, not for lack of a
    # cheap implementation.

    # Degenerate response filter: applied AFTER normal scoring below
    # (we need to know if the answer was correct first)

    # One regex pass, reused by both branches.
    _tool_names = tool_call_names(solution_str)
    n_tool_calls = len(_tool_names)
    n_invalid = sum(1 for name in _tool_names if name not in VALID_TOOLS)

    # Normalize ground_truth: extract answer letter from formats like "ANSWER: (D)", "(D)", "D"
    if ground_truth:
        gt_stripped = ground_truth.strip()
        # Try explicit patterns first: "ANSWER: (X)" or "(X)" at end
        gt_match = re.search(r"(?:ANSWER:\s*)?[\(]([A-E])[\)]", gt_stripped, re.IGNORECASE)
        if gt_match:
            ground_truth = gt_match.group(1).upper()
        elif len(gt_stripped) == 1 and gt_stripped.upper() in "ABCDE":
            ground_truth = gt_stripped.upper()

    has_options = extra_info.get("has_options", bool(ground_truth and len(ground_truth) == 1))

    if has_options:
        # MCQA scoring
        predicted = extract_answer_letter(solution_str)
        correct = ground_truth.strip().upper() if ground_truth else ""

        # Debug logging — first 20 samples
        global _debug_count
        if _DEBUG_LOG and _debug_count < 20:
            _debug_count += 1
            # Show first 200 + last 200 chars to understand format
            head = repr(solution_str[:200])
            tail = repr(solution_str[-200:]) if len(solution_str) > 200 else ""
            has_tool_call = "<tool_call>" in solution_str
            has_think = "<think>" in solution_str
            print(f"[REWARD #{_debug_count}] pred={predicted} gt={correct} len={len(solution_str)} "
                  f"tool_call={has_tool_call} think={has_think}")
            print(f"  HEAD: {head}")
            if tail:
                print(f"  TAIL: {tail}")

        if predicted is None:
            is_correct = False
            base_reward = 0.0
        else:
            is_correct = predicted == correct
            accuracy = 1.0 if is_correct else 0.0
            # Format bonus: only for correct answers to avoid rewarding wrong-but-formatted
            format_bonus = 0.1 if (accuracy > 0 and re.search(r"Answer:\s*[A-E]", solution_str)) else 0.0
            base_reward = accuracy + format_bonus

        # Unshaped metrics, captured before any shaping touches base_reward.
        acc = 1.0 if is_correct else 0.0
        answer_found = predicted is not None

        # Multi-dimensional composite (replaces base_reward when enabled).
        # No format bonus is re-added on top: `format` is one of the composite's
        # own dimensions, and adding the flat +0.1 as well would double-count it
        # and make the leave-one-out format arm not actually leave it out.
        if COMPOSITE_REWARD_ENABLED:
            base_reward = _composite_base_reward(solution_str, ground_truth, extra_info)

        # Cosine length-scaling reward (replaces base_reward when enabled)
        if COSINE_REWARD_ENABLED:
            base_reward = cosine_length_reward(is_correct, len(solution_str))
            # Still add format bonus on top for correct answers
            if is_correct and re.search(r"Answer:\s*[A-E]", solution_str):
                base_reward += 0.1

        # Penalty for calling tools not in the provided tool list
        penalty = n_invalid * INVALID_TOOL_PENALTY
        reward = base_reward - penalty

        # Degenerate filter: repetitive responses get hard penalty regardless of correctness
        # Critical: must apply to CORRECT answers too, otherwise RL rewards degenerate-but-correct
        # responses (model answers correctly early, then fills with repetition)
        degenerate = DEGENERATE_FILTER_ENABLED and is_degenerate_response(solution_str)

        metrics = dict(
            acc=acc,
            acc_partial=acc,  # MC has no partial credit; acc_partial == acc by definition
            has_options=True,
            answer_found=answer_found,
            n_tool_calls=n_tool_calls,
            n_invalid_tool_calls=n_invalid,
            degenerate=degenerate,
        )

        if degenerate:
            if DEGENERATE_EXCLUDE:
                # Sentinel, not a reward: core_algos.py drops any rollout scoring
                # below DEGENERATE_SENTINEL + 1.0 from the GRPO group statistics
                # and zeroes its advantage. It must reach the reward tensor
                # unmodified, which is why it is returned as `score` verbatim.
                return _result(DEGENERATE_SENTINEL, **metrics)
            return _result(DEGENERATE_REWARD, **metrics)
        return _result(reward, **metrics)
    else:
        # Open-ended: simple keyword overlap scoring.
        # Score the FINAL answer span (not the whole transcript) with F1.
        # Recall-only over the full multi-turn solution_str rewarded verbosity
        # (any rollout that mentions every gold token anywhere scored 1.0),
        # which is exactly opposite to the cosine length reward.
        # Hoisted out of the scoring branch only so `answer_found` is reported
        # even when there is no ground truth; the span itself is unchanged.
        answer_span, answer_found = _extract_final_answer_span_ex(solution_str)
        if not ground_truth or not solution_str:
            base_reward = 0.0
            is_correct = False
        else:
            gt_words = set(ground_truth.lower().split())
            pred_words = set(answer_span.lower().split())

            if not gt_words:
                base_reward = 0.0
                is_correct = False
            else:
                inter = len(gt_words & pred_words)
                recall = inter / len(gt_words)
                precision = inter / len(pred_words) if pred_words else 0.0
                overlap = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                base_reward = min(overlap, 1.0)
                is_correct = overlap > 0.5

        # Unshaped metrics, captured before any shaping touches base_reward.
        # base_reward is still the raw token-F1 here, so acc_partial is exact.
        acc = 1.0 if is_correct else 0.0
        acc_partial = base_reward

        # Cosine length-scaling reward
        # For open-ended questions: only apply cosine scaling when answer is
        # meaningfully correct (overlap > 0.5).  When incorrect, keep the raw
        # overlap score (0.0-0.5) instead of the cosine wrong-answer schedule,
        # which produces persistent negative signal on ~27% of data and makes
        # the model overly cautious / non-answering.
        # Composite applies to the open-ended branch on the same terms as the
        # MCQA one: it replaces the base credit (here the token-F1 overlap) and
        # leaves acc / acc_partial, the penalty and the degenerate filter alone.
        if COMPOSITE_REWARD_ENABLED:
            base_reward = _composite_base_reward(solution_str, ground_truth, extra_info)

        if COSINE_REWARD_ENABLED:
            if is_correct:
                # Good answer → reward with length efficiency bonus
                base_reward = cosine_length_reward(True, len(solution_str))
            # else: keep base_reward = overlap (0.0–0.5), no cosine penalty

        # Penalty for calling tools not in the provided tool list
        penalty = n_invalid * INVALID_TOOL_PENALTY
        reward = base_reward - penalty

        # Degenerate filter: repetitive responses get hard penalty regardless of correctness
        degenerate = DEGENERATE_FILTER_ENABLED and is_degenerate_response(solution_str)

        metrics = dict(
            acc=acc,
            acc_partial=acc_partial,
            has_options=False,
            answer_found=answer_found,
            n_tool_calls=n_tool_calls,
            n_invalid_tool_calls=n_invalid,
            degenerate=degenerate,
        )

        if degenerate:
            # NB: the open-ended branch has never honoured DEGENERATE_EXCLUDE —
            # it returns the soft penalty even when the sentinel is configured.
            # Preserved verbatim; changing it would change training.
            return _result(DEGENERATE_REWARD, **metrics)
        return _result(reward, **metrics)
