"""Evaluation module for BIOAgents.

Provides:
- AgentRunner for running LLM agents in the gym
- Reward functions (accuracy, format, process, safety)
- Metrics computation (BERTScore, ROUGE, BLEU)
- GRPO reward wrappers for RL training
- Safety evaluation (contraindication, emergency, uncertainty, adversarial)
- EHR Benchmark evaluation (MIMIC-III, eICU real-world clinical data)
"""

from bioagents.evaluation.rewards import (
    # Individual reward functions
    accuracy_reward_exact_match,
    accuracy_reward_soft,
    accuracy_reward_bertscore,
    format_reward_tool_call,
    format_reward_think_answer,
    format_reward_composite,
    process_reward_tool_usage,
    process_reward_reasoning_quality,
    # Composite
    compute_composite_reward,
    # Registry
    get_reward_function,
    register_reward_function,
)

from bioagents.evaluation.grpo_rewards import (
    grpo_accuracy_reward,
    grpo_format_reward,
    grpo_process_reward,
    grpo_tool_use_reward,
    grpo_composite_reward,
    get_grpo_reward_functions,
    GRPO_REWARD_REGISTRY,
)

from bioagents.evaluation.safety_eval import (
    # Safety reward functions
    safety_reward_contraindication,
    safety_reward_emergency_recognition,
    safety_reward_uncertainty,
    safety_reward_scope,
    compute_safety_reward,
    # Adversarial testing
    get_adversarial_test_suite,
    evaluate_adversarial,
    run_adversarial_suite,
    # GRPO wrapper
    grpo_safety_reward,
    # Data classes
    SafetyViolation,
    AdversarialTestCase,
)

__all__ = [
    # Core reward functions
    "accuracy_reward_exact_match",
    "accuracy_reward_soft",
    "accuracy_reward_bertscore",
    "format_reward_tool_call",
    "format_reward_think_answer",
    "format_reward_composite",
    "process_reward_tool_usage",
    "process_reward_reasoning_quality",
    "compute_composite_reward",
    "get_reward_function",
    "register_reward_function",
    # GRPO-compatible wrappers
    "grpo_accuracy_reward",
    "grpo_format_reward",
    "grpo_process_reward",
    "grpo_tool_use_reward",
    "grpo_composite_reward",
    "get_grpo_reward_functions",
    "GRPO_REWARD_REGISTRY",
    # Safety evaluation
    "safety_reward_contraindication",
    "safety_reward_emergency_recognition",
    "safety_reward_uncertainty",
    "safety_reward_scope",
    "compute_safety_reward",
    "get_adversarial_test_suite",
    "evaluate_adversarial",
    "run_adversarial_suite",
    "grpo_safety_reward",
    "SafetyViolation",
    "AdversarialTestCase",
]
