"""Reward function for BIOAgents medical GRPO on veRL.

Supports optional cosine length-scaling reward (Yeo et al., 2025,
"Demystifying Long Chain-of-Thought Reasoning in LLMs", arXiv:2502.03373).
Enable via environment variable COSINE_REWARD=1.
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

# ── Cosine length-scaling reward (arXiv:2502.03373) ──────────────────
# CosFn(t, T, η_min, η_max) = η_min + ½(η_max − η_min)(1 + cos(tπ/T))
# Goes from η_max at t=0 to η_min at t=T.
COSINE_REWARD_ENABLED = os.environ.get("COSINE_REWARD", "") == "1"

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


def count_invalid_tool_calls(solution_str: str) -> int:
    """Count tool calls to names not in VALID_TOOLS."""
    tool_calls = re.findall(r"<function=([^>]+)>", solution_str)
    return sum(1 for name in tool_calls if name.strip() not in VALID_TOOLS)


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


def _extract_final_answer_span(text: str, window: int = 600) -> str:
    """Return the model's final answer text for open-ended scoring.

    Prefers the content after the last submit_answer tool call or an explicit
    'Answer:'/'Final Answer:' marker; otherwise falls back to the tail of the
    response. This keeps verbosity/tool-echo out of the overlap score.
    """
    if not text:
        return ""
    # After the last submit_answer(...) payload, if present.
    m = list(re.finditer(r'submit_answer[^\{]*\{(.*?)\}', text, re.DOTALL | re.IGNORECASE))
    if m:
        return m[-1].group(1)
    # After the last explicit answer marker.
    m = list(re.finditer(r'(?:final\s+answer|answer)\s*[:\-]\s*(.+)', text, re.IGNORECASE))
    if m:
        return m[-1].group(1)
    # Fallback: tail window (final reasoning usually lands here).
    return text[-window:]


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    """Compute reward for medical QA tasks.

    For MCQA: binary reward (1.0 correct, 0.0 wrong) + format bonus.
    For open-ended: partial credit based on keyword overlap.
    """
    if extra_info is None:
        extra_info = {}

    # Parse extra_info if it's a string
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)

    # During validation, use binary accuracy (no cosine scaling)
    is_validate = extra_info.get("validate", False)

    # Degenerate response filter: applied AFTER normal scoring below
    # (we need to know if the answer was correct first)

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

        # Cosine length-scaling reward (replaces base_reward when enabled, training only)
        if COSINE_REWARD_ENABLED and not is_validate:
            base_reward = cosine_length_reward(is_correct, len(solution_str))
            # Still add format bonus on top for correct answers
            if is_correct and re.search(r"Answer:\s*[A-E]", solution_str):
                base_reward += 0.1

        # Penalty for calling tools not in the provided tool list
        n_invalid = count_invalid_tool_calls(solution_str)
        penalty = n_invalid * INVALID_TOOL_PENALTY
        reward = base_reward - penalty

        # Degenerate filter: repetitive responses get hard penalty regardless of correctness
        # Critical: must apply to CORRECT answers too, otherwise RL rewards degenerate-but-correct
        # responses (model answers correctly early, then fills with repetition)
        if DEGENERATE_FILTER_ENABLED and not is_validate:
            if is_degenerate_response(solution_str):
                if DEGENERATE_EXCLUDE:
                    return DEGENERATE_SENTINEL  # Excluded from batch in core_algos
                return DEGENERATE_REWARD
        return reward
    else:
        # Open-ended: simple keyword overlap scoring
        if not ground_truth or not solution_str:
            base_reward = 0.0
            is_correct = False
        else:
            # Score the FINAL answer span (not the whole transcript) with F1.
            # Recall-only over the full multi-turn solution_str rewarded verbosity
            # (any rollout that mentions every gold token anywhere scored 1.0),
            # which is exactly opposite to the cosine length reward.
            answer_span = _extract_final_answer_span(solution_str)
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

        # Cosine length-scaling reward (training only)
        # For open-ended questions: only apply cosine scaling when answer is
        # meaningfully correct (overlap > 0.5).  When incorrect, keep the raw
        # overlap score (0.0-0.5) instead of the cosine wrong-answer schedule,
        # which produces persistent negative signal on ~27% of data and makes
        # the model overly cautious / non-answering.
        if COSINE_REWARD_ENABLED and not is_validate:
            if is_correct:
                # Good answer → reward with length efficiency bonus
                base_reward = cosine_length_reward(True, len(solution_str))
            # else: keep base_reward = overlap (0.0–0.5), no cosine penalty

        # Penalty for calling tools not in the provided tool list
        n_invalid = count_invalid_tool_calls(solution_str)
        penalty = n_invalid * INVALID_TOOL_PENALTY
        reward = base_reward - penalty

        # Degenerate filter: repetitive responses get hard penalty regardless of correctness
        if DEGENERATE_FILTER_ENABLED and not is_validate:
            if is_degenerate_response(solution_str):
                return DEGENERATE_REWARD
        return reward
