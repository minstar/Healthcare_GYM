"""Reward functions for BIOAgents RL training.

Implements multi-signal reward computation following the MRPO framework:
1. Accuracy Reward: Correctness of the answer (exact match + soft metrics)
2. Format Reward: Adherence to expected output format (tool calls, answer format)
3. Process Reward: Quality of reasoning and tool usage (LLM-as-judge or heuristic)

These rewards are designed to be used with GRPO/PPO training pipelines.

Reference: MRPO (grpo_vqa_Qwen3_token_shaping.py)
"""

import json
import re
from typing import Any, Callable, Optional

from loguru import logger


# ============================================================
# 1. Accuracy Reward
# ============================================================


def accuracy_reward_exact_match(
    response: str,
    correct_answer: str,
    **kwargs,
) -> float:
    """Exact match accuracy reward.

    For multiple-choice questions, checks if the submitted answer
    matches the correct answer label.

    Args:
        response: The agent's final response or submitted answer
        correct_answer: The ground truth answer

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Try to extract answer from response
    submitted = _extract_answer_from_response(response)
    if submitted and submitted.upper() == correct_answer.strip().upper():
        return 1.0
    return 0.0


def accuracy_reward_soft(
    response: str,
    correct_answer: str,
    reference_text: str = "",
    **kwargs,
) -> float:
    """Soft accuracy reward using text overlap metrics.

    Combines ROUGE-1 and BLEU-1 scores for open-ended answers.
    For multiple-choice, falls back to exact match.

    Args:
        response: The agent's response
        correct_answer: The ground truth answer
        reference_text: Optional reference explanation text

    Returns:
        Score between 0.0 and 1.0
    """
    # If it's a single-letter answer, use exact match
    if len(correct_answer.strip()) <= 2:
        return accuracy_reward_exact_match(response, correct_answer)

    # For longer answers, use text overlap
    reference = reference_text or correct_answer
    response_clean = response.strip().lower()
    reference_clean = reference.strip().lower()

    if not response_clean or not reference_clean:
        return 0.0

    # Simple token overlap (ROUGE-1 proxy)
    response_tokens = set(response_clean.split())
    reference_tokens = set(reference_clean.split())

    if not reference_tokens:
        return 0.0

    precision = len(response_tokens & reference_tokens) / max(len(response_tokens), 1)
    recall = len(response_tokens & reference_tokens) / len(reference_tokens)

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def accuracy_reward_bertscore(
    response: str,
    correct_answer: str,
    reference_text: str = "",
    **kwargs,
) -> float:
    """BERTScore-based accuracy reward.

    Uses BERTScore for semantic similarity between response and reference.
    Falls back to soft reward if BERTScore is not available.

    Args:
        response: The agent's response
        correct_answer: The ground truth answer
        reference_text: Optional reference explanation text

    Returns:
        Score between 0.0 and 1.0
    """
    try:
        from bert_score import score as bert_score
        reference = reference_text or correct_answer
        P, R, F1 = bert_score(
            [response], [reference],
            lang="en",
            model_type="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            verbose=False,
        )
        return float(F1[0])
    except (ImportError, Exception) as e:
        logger.debug(f"BERTScore unavailable ({e}), falling back to soft reward")
        return accuracy_reward_soft(response, correct_answer, reference_text)


# ============================================================
# 2. Format Reward
# ============================================================


def format_reward_tool_call(
    response: str,
    expected_format: str = "json_tool_call",
    **kwargs,
) -> float:
    """Reward for adhering to the expected tool call format.

    Checks if the response is a valid JSON tool call with
    'name' and 'arguments' fields.

    Args:
        response: The agent's response
        expected_format: Expected format type

    Returns:
        1.0 if valid format, 0.5 for partial, 0.0 for invalid
    """
    response = response.strip()

    if expected_format == "json_tool_call":
        # Try direct JSON parse
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                has_name = "name" in parsed
                has_args = "arguments" in parsed
                if has_name and has_args:
                    return 1.0
                elif has_name:
                    return 0.5
        except json.JSONDecodeError:
            pass

        # Try code block extraction
        code_match = re.search(r'```(?:json)?\s*\n?({.*?})\s*\n?```', response, re.DOTALL)
        if code_match:
            try:
                parsed = json.loads(code_match.group(1))
                if "name" in parsed:
                    return 0.8  # Slightly lower for code block wrapping
            except json.JSONDecodeError:
                pass

        return 0.0

    elif expected_format == "answer_letter":
        # Check if response contains an answer letter A-E
        answer = _extract_answer_from_response(response)
        if answer and answer.upper() in "ABCDE":
            return 1.0
        # Also try <answer> tag extraction
        answer_tag = re.search(r"<answer>\s*([A-E])\s*</answer>", response, re.IGNORECASE)
        if answer_tag:
            return 1.0
        return 0.0

    return 0.0


def format_reward_think_answer(
    response: str,
    **kwargs,
) -> float:
    """Reward for using <think>...</think> <answer>...</answer> format.

    Args:
        response: The agent's response

    Returns:
        Score based on format adherence
    """
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))

    if has_think and has_answer:
        return 1.0
    elif has_think or has_answer:
        return 0.5
    return 0.0


def format_reward_composite(
    response: str,
    turn_idx: int = 0,
    is_final: bool = False,
    **kwargs,
) -> float:
    """Composite format reward that adapts based on turn context.

    - During interaction turns: expects valid tool calls
    - On final turn: expects a coherent answer

    Args:
        response: The agent's response
        turn_idx: Current turn index
        is_final: Whether this is the final response

    Returns:
        Format score between 0.0 and 1.0
    """
    if is_final:
        # Final answer should be non-empty text
        if len(response.strip()) > 10:
            return 1.0
        return 0.3
    else:
        # Intermediate turns should be tool calls
        return format_reward_tool_call(response)


# ============================================================
# 3. Process Reward
# ============================================================


def process_reward_tool_usage(
    tool_call_log: list[dict],
    expected_actions: list[dict],
    **kwargs,
) -> float:
    """Reward based on the quality and correctness of tool usage.

    Evaluates:
    - Were the expected tools called? (coverage)
    - Were tool calls diverse in arguments? (diversity)
    - Was tool usage efficient? (no exact-duplicate calls)

    The efficiency metric now counts *unique (name, args)* pairs instead of
    just unique names. This avoids penalising agents that call the same tool
    with different arguments (e.g. get_lab_results for creatinine vs WBC),
    which is legitimate clinical workflow.

    Args:
        tool_call_log: List of actual tool calls made
        expected_actions: List of expected tool call specifications

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected_actions:
        # If no expected actions defined, reward any non-trivial tool usage
        if len(tool_call_log) > 0:
            return 1.0
        return 0.5

    # ── Score 1: Coverage — did the agent call the expected tools? ────
    matched = 0
    for exp in expected_actions:
        exp_name = exp.get("name", "")
        compare_args = exp.get("compare_args", [])
        exp_args = exp.get("arguments", {})

        for tc in tool_call_log:
            if tc.get("tool_name") == exp_name:
                if compare_args:
                    all_match = all(
                        str(tc.get("arguments", {}).get(k, "")).lower()
                        == str(exp_args.get(k, "")).lower()
                        for k in compare_args
                        if k in exp_args
                    )
                    if all_match:
                        matched += 1
                        break
                else:
                    matched += 1
                    break

    tool_coverage = matched / len(expected_actions)

    # ── Score 2: Diversity — unique (name, sorted_args) pairs ────────
    # Calling the same tool with DIFFERENT arguments is legitimate clinical
    # workflow and should not be penalised.  Only exact-duplicate calls
    # (same name AND same arguments) count as wasteful repetition.
    seen_signatures = set()
    for tc in tool_call_log:
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        # Create a hashable signature from name + sorted argument items
        try:
            sig = (name, tuple(sorted(
                (str(k), str(v)) for k, v in args.items()
            )))
        except Exception:
            sig = (name,)
        seen_signatures.add(sig)

    total_calls = len(tool_call_log)
    unique_calls = len(seen_signatures)

    if total_calls > 0:
        # Ratio of unique signatures to total calls
        diversity = unique_calls / total_calls
    else:
        diversity = 0.0

    # ── Score 3: Thoroughness bonus ──────────────────────────────────
    # Reward agents that explore more tools (use more of the available
    # tool space), capped at a reasonable maximum.
    unique_tool_names = len({tc.get("tool_name", "") for tc in tool_call_log})
    # Bonus scales with the number of distinct tools used (max 1.0 at 5+)
    thoroughness = min(1.0, unique_tool_names / 5.0) if total_calls > 0 else 0.0

    # ── Combined score ───────────────────────────────────────────────
    # 60% coverage (did you do the right things?)
    # 20% diversity (were your calls meaningfully different?)
    # 20% thoroughness (did you use a variety of tools?)
    return 0.6 * tool_coverage + 0.2 * diversity + 0.2 * thoroughness


def process_reward_reasoning_quality(
    response: str,
    correct_answer: str = "",
    **kwargs,
) -> float:
    """Heuristic reward for reasoning quality.

    Evaluates the quality of the agent's reasoning based on:
    - Length (reasonable, not too short)
    - Mention of key medical terms
    - Structured reasoning indicators
    - Clinical specificity (drug names, lab values, scores)

    Args:
        response: The agent's reasoning/final response
        correct_answer: The correct answer (for context)

    Returns:
        Score between 0.0 and 1.0
    """
    if not response.strip():
        return 0.0

    score = 0.0
    response_lower = response.lower()

    # ── Length check (reasonable response length) ────────────────────
    word_count = len(response.split())
    if 50 <= word_count <= 800:
        score += 0.25          # ideal range for thorough clinical reasoning
    elif 20 <= word_count <= 1200:
        score += 0.15
    elif 10 <= word_count:
        score += 0.05

    # ── Medical terminology presence ─────────────────────────────────
    medical_indicators = [
        "diagnosis", "symptom", "treatment", "mechanism", "cause",
        "patient", "clinical", "drug", "disease", "condition",
        "evidence", "findings", "analysis", "pathology", "physiology",
        "because", "therefore", "suggests", "indicates", "consistent",
        "prognosis", "differential", "etiology", "management",
        "medication", "dosage", "contraindication", "interaction",
        "laboratory", "imaging", "biopsy", "histology",
    ]
    term_count = sum(1 for term in medical_indicators if term in response_lower)
    score += min(0.25, term_count * 0.03)

    # ── Structured reasoning indicators ──────────────────────────────
    reasoning_markers = [
        "first", "second", "next", "finally", "in conclusion",
        "based on", "the evidence", "differential", "rule out",
        "key finding", "most likely", "best answer", "assessment",
        "recommend", "plan", "follow-up", "monitor", "consider",
        "risk", "benefit", "indication", "guideline",
    ]
    marker_count = sum(1 for m in reasoning_markers if m in response_lower)
    score += min(0.2, marker_count * 0.04)

    # ── Clinical specificity (numbers, scores, drug names) ──────────
    import re as _re
    has_numbers = bool(_re.search(r'\d+\.?\d*\s*(?:mg|mcg|ml|mmol|mmHg|bpm|%|mg/dL|mEq/L|U/L)', response))
    has_scores = bool(_re.search(r'(?:SOFA|NEWS|GRACE|APACHE|GCS|MELD|CHA2DS2|HAS-BLED|Wells)\s*(?:score)?\s*(?:=|:|\d)', response, _re.IGNORECASE))
    has_labs = bool(_re.search(r'(?:creatinine|BNP|troponin|lactate|WBC|hemoglobin|BUN|INR|HbA1c|LDL|CRP|procalcitonin)', response, _re.IGNORECASE))

    specificity_count = sum([has_numbers, has_scores, has_labs])
    score += min(0.15, specificity_count * 0.05)

    # ── Answer presence ──────────────────────────────────────────────
    if correct_answer and correct_answer.strip().upper() in response.upper():
        score += 0.15

    return min(1.0, score)


# ============================================================
# 4. Composite Reward (for RL training)
# ============================================================


def compute_composite_reward(
    response: str,
    correct_answer: str = "",
    reference_text: str = "",
    tool_call_log: list[dict] = None,
    expected_actions: list[dict] = None,
    turn_idx: int = 0,
    is_final: bool = False,
    weights: dict = None,
    **kwargs,
) -> dict:
    """Compute the composite reward from all signal components.

    Args:
        response: The agent's response
        correct_answer: Ground truth answer
        reference_text: Reference explanation
        tool_call_log: Log of tool calls made
        expected_actions: Expected tool call specifications
        turn_idx: Current turn index
        is_final: Whether this is the final response
        weights: Reward component weights (default: accuracy=0.4, format=0.2, process=0.4)

    Returns:
        Dictionary with individual scores and weighted total
    """
    if weights is None:
        weights = {
            "accuracy": 0.30,
            "format": 0.15,
            "process": 0.25,
            "safety": 0.20,
            "coherence": 0.10,
        }

    if tool_call_log is None:
        tool_call_log = []
    if expected_actions is None:
        expected_actions = []

    # --- 1. Accuracy Reward ---
    accuracy = accuracy_reward_soft(response, correct_answer, reference_text)

    # --- 2. Format Reward ---
    format_score = format_reward_composite(
        response, turn_idx=turn_idx, is_final=is_final
    )

    # --- 3. Process Reward (tool usage + reasoning quality) ---
    tool_score = process_reward_tool_usage(tool_call_log, expected_actions)
    reasoning_score = process_reward_reasoning_quality(response, correct_answer)
    process_score = 0.5 * tool_score + 0.5 * reasoning_score

    # --- 4. Safety Reward ---
    safety_score = 1.0  # Default: safe
    if weights.get("safety", 0) > 0:
        try:
            from bioagents.evaluation.safety_eval import compute_safety_reward
            safety_result = compute_safety_reward(
                response=response,
                task_domain=kwargs.get("task_domain", ""),
                patient_allergies=kwargs.get("patient_allergies", []),
                patient_conditions=kwargs.get("patient_conditions", []),
                emergency_type=kwargs.get("emergency_type", ""),
            )
            safety_score = safety_result.get("total", 1.0)
        except Exception:
            safety_score = 1.0  # Assume safe if evaluation fails

    # --- 5. Coherence Reward ---
    # Measures response structure: logical flow, no contradictions,
    # clear final answer
    coherence_score = _compute_coherence_score(response, is_final)

    # Weighted 5D total
    total = (
        weights.get("accuracy", 0.30) * accuracy
        + weights.get("format", 0.15) * format_score
        + weights.get("process", 0.25) * process_score
        + weights.get("safety", 0.20) * safety_score
        + weights.get("coherence", 0.10) * coherence_score
    )

    return {
        "total": total,
        "accuracy": accuracy,
        "format": format_score,
        "process": process_score,
        "safety": safety_score,
        "coherence": coherence_score,
        "tool_usage": tool_score,
        "reasoning_quality": reasoning_score,
        "weights": weights,
    }


def _compute_coherence_score(response: str, is_final: bool = False) -> float:
    """Compute coherence reward for agent responses.

    Measures:
    - Logical structure (has intro, reasoning, conclusion)
    - No self-contradictions
    - Clear final answer if is_final
    - Appropriate length (not too short, not repetitive)

    Returns:
        float in [0, 1]
    """
    if not response or not response.strip():
        return 0.0

    score = 0.5  # Baseline

    # Length check: too short is bad
    words = response.split()
    if len(words) < 5:
        score -= 0.3
    elif len(words) >= 20:
        score += 0.1

    # Structure: has reasoning markers
    reasoning_markers = [
        "because", "therefore", "since", "based on",
        "considering", "given that", "analysis",
        "evidence", "indicates", "suggests",
        "이유", "따라서", "기반", "결과",
    ]
    has_reasoning = any(
        m.lower() in response.lower() for m in reasoning_markers
    )
    if has_reasoning:
        score += 0.2

    # Final answer clarity
    if is_final:
        answer_markers = [
            "final answer", "diagnosis", "recommendation",
            "conclusion", "assessment", "in summary",
            "최종", "진단", "결론",
        ]
        has_conclusion = any(
            m.lower() in response.lower() for m in answer_markers
        )
        if has_conclusion:
            score += 0.15

    # Repetition penalty: check for repeated phrases
    sentences = response.split(".")
    if len(sentences) > 3:
        unique_ratio = len(set(s.strip().lower() for s in sentences if s.strip())) / len(sentences)
        if unique_ratio < 0.5:
            score -= 0.3  # Heavy repetition

    return max(0.0, min(1.0, score))


# ============================================================
# 5. Reward Function Registry (for GRPO/PPO training)
# ============================================================


_REWARD_REGISTRY: dict[str, Callable] = {
    "accuracy_exact": accuracy_reward_exact_match,
    "accuracy_soft": accuracy_reward_soft,
    "accuracy_bertscore": accuracy_reward_bertscore,
    "format_tool_call": format_reward_tool_call,
    "format_think_answer": format_reward_think_answer,
    "format_composite": format_reward_composite,
    "process_tool_usage": process_reward_tool_usage,
    "process_reasoning": process_reward_reasoning_quality,
    "composite": lambda **kw: compute_composite_reward(**kw)["total"],
}

# Safety rewards are registered lazily to avoid circular imports.
# They become available after `bioagents.evaluation.safety_eval` is imported.
def _register_safety_rewards() -> None:
    """Register safety reward functions into the global registry."""
    try:
        from bioagents.evaluation.safety_eval import (
            compute_safety_reward,
            grpo_safety_reward,
        )
        _REWARD_REGISTRY["safety_composite"] = lambda **kw: compute_safety_reward(**kw)["total"]
        _REWARD_REGISTRY["safety_contraindication"] = lambda **kw: compute_safety_reward(**kw)["contraindication"]
        _REWARD_REGISTRY["safety_emergency"] = lambda **kw: compute_safety_reward(**kw)["emergency"]
        _REWARD_REGISTRY["safety_uncertainty"] = lambda **kw: compute_safety_reward(**kw)["uncertainty"]
    except ImportError:
        pass

_register_safety_rewards()


def get_reward_function(name: str) -> Callable:
    """Get a reward function by name.

    Args:
        name: Reward function name from the registry

    Returns:
        The reward function callable
    """
    if name not in _REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Available: {list(_REWARD_REGISTRY.keys())}"
        )
    return _REWARD_REGISTRY[name]


def register_reward_function(name: str, fn: Callable) -> None:
    """Register a custom reward function."""
    _REWARD_REGISTRY[name] = fn


# ============================================================
# Helper Functions
# ============================================================


def _extract_answer_from_response(text: str) -> str:
    """Extract an answer letter from model response text."""
    text = text.strip()

    # Check for <answer> tags first
    answer_tag = re.search(r"<answer>\s*([A-E])\s*</answer>", text, re.IGNORECASE)
    if answer_tag:
        return answer_tag.group(1).upper()

    # Check for submit_answer tool call
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("name") == "submit_answer":
            return parsed.get("arguments", {}).get("answer", "")
    except json.JSONDecodeError:
        pass

    # Look for explicit "Answer: X" patterns (order matters - more specific first)
    patterns = [
        r"(?:the\s+)?(?:correct|best|final)\s+answer\s+is\s+([A-E])\b",
        r"(?:answer|final answer|my answer|correct answer)\s+is\s+([A-E])\b",
        r"(?:answer|final answer|my answer|correct answer)[:\s]+([A-E])\b",
        r"\b([A-E])\s+is\s+(?:the\s+)?(?:correct|best|right)\b",
        r"Option\s+([A-E])\b",
        r"^([A-E])$",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Single character response
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()

    return ""
