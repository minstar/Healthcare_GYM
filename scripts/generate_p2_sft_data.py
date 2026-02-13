#!/usr/bin/env python3
"""P2-1: Enhanced multi-domain SFT data generation for 500+ examples.

Strategy:
  1. Extract existing Qwen3-8B-Base expert trajectories (lower threshold)
  2. Generate synthetic ideal trajectories for ALL 5 domains
  3. Use medical_qa_200 (200 tasks) for expanded QA data
  4. Augment with trajectory variations (different tool orders, reasoning depth)
  5. Domain-balanced output with quality metrics

Output: datasets/sft/p2_multidomain_sft.jsonl (target: 500+ examples)

Usage:
    python scripts/generate_p2_sft_data.py
    python scripts/generate_p2_sft_data.py --target-count 600 --min-score 0.3
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_DIR = PROJECT_ROOT / "logs" / "baseline"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "sft"

# ── Domain tool registry ────────────────────────────────────────────────────

DOMAIN_TOOLS = {
    "clinical_diagnosis": [
        "get_patient_info", "get_vital_signs", "get_lab_results",
        "get_patient_history", "get_medications", "get_clinical_notes",
        "get_vital_signs_trend", "order_lab_test", "check_drug_interaction",
        "get_differential_diagnosis", "search_clinical_guidelines",
        "search_medical_literature", "record_diagnosis",
        "prescribe_medication", "add_clinical_note",
        "transfer_to_specialist", "think",
    ],
    "medical_qa": [
        "search_pubmed", "browse_article", "search_medical_wiki",
        "browse_wiki_entry", "retrieve_evidence",
        "analyze_answer_options", "think", "submit_answer",
    ],
    "visual_diagnosis": [
        "analyze_medical_image", "get_patient_context",
        "get_image_report", "compare_with_prior",
        "search_similar_cases", "search_imaging_knowledge",
        "record_visual_diagnosis", "submit_answer", "think",
    ],
    "drug_interaction": [
        "get_patient_medications", "check_interaction",
        "check_all_interactions", "get_drug_info", "check_dosage",
        "search_alternatives", "search_drugs_by_class",
        "submit_answer", "think",
    ],
    "ehr_management": [
        "get_patient_summary", "get_lab_results", "get_lab_trend",
        "get_vital_signs", "detect_vital_alerts",
        "get_medication_orders", "get_admission_history",
        "get_clinical_scores", "get_discharge_summary",
        "get_procedures", "get_quality_indicators",
        "lookup_icd_code", "submit_answer", "think",
    ],
}

# ── Domain system prompts ───────────────────────────────────────────────────

DOMAIN_SYSTEM_PROMPTS = {
    "clinical_diagnosis": (
        "You are a clinical AI assistant. Your task is to assess patients, "
        "review their medical history, vital signs, and laboratory results, "
        "perform differential diagnosis, check drug interactions, and develop "
        "treatment plans. Use the available tools systematically.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "One tool call per response. When done, provide your clinical assessment as text."
    ),
    "medical_qa": (
        "You are a medical AI assistant that answers medical questions using "
        "evidence-based reasoning. Search for evidence, think through the "
        "options, and submit your answer with clear reasoning.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When ready, use submit_answer to submit your final answer."
    ),
    "visual_diagnosis": (
        "You are a medical imaging AI assistant. Analyze medical images "
        "(X-ray, CT, MRI, pathology, dermoscopy, fundoscopy) to identify "
        "abnormalities and provide diagnostic assessments.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When done, use submit_answer with your visual diagnosis."
    ),
    "drug_interaction": (
        "You are a clinical pharmacology AI assistant. Review patient medication "
        "profiles, identify drug-drug interactions, assess risk levels, and "
        "recommend safer alternatives when needed.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When done, use submit_answer with your interaction assessment."
    ),
    "ehr_management": (
        "You are an EHR clinical AI assistant. Review electronic health records, "
        "analyze lab trends, calculate clinical scores, assess discharge readiness, "
        "and provide comprehensive clinical assessments.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When done, use submit_answer with your clinical recommendation."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Extract existing expert trajectories (from Qwen3-8B baseline)
# ═══════════════════════════════════════════════════════════════════════


def _load_all_test_ids() -> set:
    """Load all test-split task IDs across all domains for contamination filtering."""
    test_ids = set()
    domains_dir = PROJECT_ROOT / "data" / "domains"
    if domains_dir.exists():
        for domain_dir in domains_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            split_path = domain_dir / "split_tasks.json"
            if split_path.exists():
                with open(split_path) as f:
                    splits = json.load(f)
                test_ids.update(splits.get("test", []))
    return test_ids


def extract_expert_trajectories(
    min_action_score: float = 0.3,
    model_filter: str = None,
) -> list[dict]:
    """Extract trajectories from all baseline runs with lower threshold.

    IMPORTANT: Filters out any trajectories from test-split tasks to prevent
    data contamination.
    """
    examples = []
    test_ids = _load_all_test_ids()
    filtered_count = 0

    for run_dir in sorted(BASELINE_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        parts = run_dir.name.split("_")
        domain = None
        model_parts = []
        for i, p in enumerate(parts):
            candidate = "_".join(parts[i:])
            for d in DOMAIN_SYSTEM_PROMPTS:
                if candidate.startswith(d + "_"):
                    domain = d
                    model_parts = parts[:i]
                    break
            if domain:
                break

        if not domain:
            continue

        model_name = "-".join(model_parts) if model_parts else parts[0]
        if model_filter and model_filter not in model_name:
            continue

        for task_file in sorted(run_dir.glob("task_*.json")):
            example = _trajectory_file_to_sft(task_file, domain, model_name, min_action_score)
            if example:
                # Filter out test-split trajectories to prevent contamination
                task_id = example.get("metadata", {}).get("task_id", "")
                if task_id in test_ids:
                    filtered_count += 1
                    continue
                examples.append(example)

    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} test-split trajectories to prevent contamination")
    return examples


def _trajectory_file_to_sft(
    task_file: Path,
    domain: str,
    model_name: str,
    min_action_score: float,
) -> dict | None:
    """Convert a task trajectory file into an SFT example."""
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    action_score = data.get("action_score", 0.0)
    if action_score < min_action_score:
        return None

    turns = data.get("turns", [])
    if not turns:
        return None

    messages = [{"role": "system", "content": DOMAIN_SYSTEM_PROMPTS.get(domain, "")}]

    task_id = data.get("task_id", "")
    messages.append({
        "role": "user",
        "content": f"[Task: {task_id}] Please complete this clinical task using the available tools.",
    })

    valid_turns = 0
    for turn in turns:
        raw_output = turn.get("raw_output", "")
        parsed_tool_call = turn.get("parsed_tool_call")
        tool_response = turn.get("tool_response", "")
        is_final = turn.get("is_final_answer", False)

        if parsed_tool_call:
            messages.append({
                "role": "assistant",
                "content": json.dumps(parsed_tool_call, ensure_ascii=False),
            })
            valid_turns += 1
            if tool_response:
                tool_name = parsed_tool_call.get("name", "tool")
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{tool_response[:3000]}",
                })
        elif is_final and raw_output:
            messages.append({"role": "assistant", "content": raw_output})
            valid_turns += 1

    if valid_turns < 1:
        return None

    return {
        "messages": messages,
        "metadata": {
            "source": "trajectory",
            "domain": domain,
            "task_id": task_id,
            "model": model_name,
            "action_score": action_score,
            "final_reward": data.get("final_reward", 0.0),
            "total_turns": data.get("total_turns", 0),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Synthetic ideal trajectory generation (ALL domains)
# ═══════════════════════════════════════════════════════════════════════


def _normalize_options(options) -> list[dict]:
    """Normalize options to list[dict] format."""
    if isinstance(options, dict):
        return [{"label": k, "text": str(v)} for k, v in options.items()]
    if isinstance(options, list):
        if options and isinstance(options[0], dict) and "label" in options[0]:
            return options
        if options and isinstance(options[0], str):
            labels = "ABCDEFGHIJ"
            return [{"label": labels[i], "text": o} for i, o in enumerate(options) if i < len(labels)]
    return []


def _build_qa_messages(
    system_prompt: str, ticket: str, strategy: str,
    question: str, opts_str: str, correct_answer: str,
    correct_text: str, search_query: str, options,
) -> list[dict]:
    """Build messages for a single QA SFT example given a strategy."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ticket},
    ]

    if strategy == "think_search_submit":
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "think",
            "arguments": {"thought": (
                f"Let me analyze this medical question carefully. "
                f"The question involves: {question[:200]}. Options: {opts_str}. "
                f"I should consider the key medical concepts and search for evidence."
            )}
        })})
        messages.append({"role": "user", "content": "Tool result for think:\nThought recorded."})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "retrieve_evidence", "arguments": {"query": search_query[:80]}
        })})
        messages.append({"role": "user", "content": "Tool result for retrieve_evidence:\n[Evidence retrieved. Multiple relevant passages found.]"})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "submit_answer",
            "arguments": {"answer": correct_answer, "reasoning": (
                f"Based on the evidence, the answer is {correct_answer}: {correct_text}. "
                f"The clinical presentation and evidence support this option."
            )}
        })})

    elif strategy == "search_think_submit":
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "search_pubmed", "arguments": {"query": search_query[:60]}
        })})
        messages.append({"role": "user", "content": "Tool result for search_pubmed:\n[Found 5 relevant articles.]"})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "think",
            "arguments": {"thought": (
                f"Based on the search results, the evidence supports option "
                f"{correct_answer}: {correct_text}. Let me submit this answer."
            )}
        })})
        messages.append({"role": "user", "content": "Tool result for think:\nThought recorded."})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "submit_answer",
            "arguments": {"answer": correct_answer, "reasoning": f"The answer is {correct_answer}: {correct_text}."}
        })})

    elif strategy == "think_search_analyze_submit":
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "think",
            "arguments": {"thought": (
                f"This medical question requires careful analysis. "
                f"Question: {question[:150]}. I need to search evidence and evaluate systematically."
            )}
        })})
        messages.append({"role": "user", "content": "Tool result for think:\nThought recorded."})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "retrieve_evidence", "arguments": {"query": search_query[:80]}
        })})
        messages.append({"role": "user", "content": "Tool result for retrieve_evidence:\n[Evidence passages retrieved.]"})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "analyze_answer_options",
            "arguments": {"question": question[:200], "options": json.dumps(options) if isinstance(options, dict) else str(options)}
        })})
        messages.append({"role": "user", "content": f"Tool result for analyze_answer_options:\nBest supported: {correct_answer} ({correct_text})"})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "submit_answer",
            "arguments": {"answer": correct_answer, "reasoning": f"After analyzing evidence: {correct_answer}: {correct_text}."}
        })})

    else:  # evidence_submit
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "retrieve_evidence", "arguments": {"query": search_query[:80]}
        })})
        messages.append({"role": "user", "content": "Tool result for retrieve_evidence:\n[Relevant evidence found.]"})
        messages.append({"role": "assistant", "content": json.dumps({
            "name": "submit_answer",
            "arguments": {"answer": correct_answer, "reasoning": f"Based on evidence: {correct_answer} ({correct_text})"}
        })})

    return messages


def generate_medical_qa_sft(tasks: list[dict], max_examples: int = 300) -> list[dict]:
    """Generate SFT examples from medical QA tasks with varied strategies."""
    examples = []
    system_prompt = DOMAIN_SYSTEM_PROMPTS["medical_qa"]

    strategies = [
        "think_search_submit",
        "search_think_submit",
        "think_search_analyze_submit",
        "evidence_submit",
    ]

    for task in tasks[:max_examples]:
        correct_answer = task.get("correct_answer", "")
        if not correct_answer:
            continue

        ticket = task.get("ticket", "")
        question = task.get("raw_question", ticket)
        options = task.get("options", {})
        answer_text = task.get("raw_answer", "")
        source = task.get("description", {}).get("source", "unknown")

        norm_opts = _normalize_options(options)
        opts_str = ", ".join(f"{o['label']}: {o['text'][:80]}" for o in norm_opts)

        correct_text = answer_text
        for opt in norm_opts:
            if opt["label"] == correct_answer:
                correct_text = opt["text"]
                break

        stop_words = {
            "the", "a", "an", "is", "was", "were", "are", "of", "in", "to",
            "for", "with", "which", "following", "most", "likely", "due",
            "patient", "year", "old", "man", "woman", "comes", "physician",
            "because", "history", "shows", "laboratory", "studies", "show",
        }
        words = question.split()[:30]
        search_query = " ".join(
            w.strip(".,;:()")
            for w in words
            if w.lower().strip(".,;:()") not in stop_words and len(w) > 2
        )[:100]

        # Generate 2 strategies per task for diversity
        selected_strategies = random.sample(strategies, k=min(2, len(strategies)))

        for strategy in selected_strategies:
            messages = _build_qa_messages(
                system_prompt, ticket, strategy,
                question, opts_str, correct_answer,
                correct_text, search_query, options,
            )
            examples.append({
                "messages": messages,
                "metadata": {
                    "source": source,
                    "domain": "medical_qa",
                    "task_id": task.get("id", ""),
                    "correct_answer": correct_answer,
                    "strategy": strategy,
                },
            })

    return examples


def generate_clinical_dx_sft(tasks: list[dict]) -> list[dict]:
    """Generate synthetic SFT for clinical diagnosis tasks."""
    examples = []
    system_prompt = DOMAIN_SYSTEM_PROMPTS["clinical_diagnosis"]

    # Multiple workflow patterns
    patterns = [
        # Comprehensive assessment
        [
            ("get_patient_info", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_vital_signs", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_lab_results", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_patient_history", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_differential_diagnosis", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("search_clinical_guidelines", lambda t: {"query": t.get("description", {}).get("condition", "clinical guidelines")}),
            ("record_diagnosis", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1]), "diagnosis": t.get("description", {}).get("condition", "diagnosis")}),
        ],
        # Quick assessment
        [
            ("get_patient_info", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_vital_signs", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_lab_results", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_differential_diagnosis", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
        ],
        # With medication focus
        [
            ("get_patient_info", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_vital_signs", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("get_medications", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("check_drug_interaction", lambda t: {"drug_a": "drug_a", "drug_b": "drug_b"}),
            ("get_lab_results", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1])}),
            ("record_diagnosis", lambda t: {"patient_id": t.get("user_scenario", {}).get("patient_id", t["id"].split("_")[-1]), "diagnosis": t.get("description", {}).get("condition", "diagnosis")}),
        ],
    ]

    for task in tasks:
        patient_id = "P001"
        # Try to extract patient ID from ticket
        ticket = task.get("ticket", "")
        import re
        pid_match = re.search(r'Patient\s+(P\d+)', ticket)
        if pid_match:
            patient_id = pid_match.group(1)

        for pattern_idx, pattern in enumerate(patterns):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticket},
            ]

            for tool_name, arg_fn in pattern:
                try:
                    args = arg_fn(task)
                    # Substitute actual patient_id
                    for k, v in args.items():
                        if v == task["id"].split("_")[-1]:
                            args[k] = patient_id
                except Exception:
                    args = {"patient_id": patient_id}

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"name": tool_name, "arguments": args}),
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n[{tool_name} result returned successfully]",
                })

            # Final assessment
            messages.append({
                "role": "assistant",
                "content": (
                    f"Based on my comprehensive assessment of the patient, including vital signs, "
                    f"laboratory results, and clinical history, I have completed the clinical evaluation. "
                    f"The diagnosis and treatment plan have been recorded."
                ),
            })

            examples.append({
                "messages": messages,
                "metadata": {
                    "source": "synthetic_trajectory",
                    "domain": "clinical_diagnosis",
                    "task_id": task["id"],
                    "pattern": f"pattern_{pattern_idx}",
                },
            })

    return examples


def generate_visual_dx_sft(tasks: list[dict]) -> list[dict]:
    """Generate synthetic SFT for visual diagnosis tasks."""
    examples = []
    system_prompt = DOMAIN_SYSTEM_PROMPTS["visual_diagnosis"]

    patterns = [
        # Standard analysis
        ["analyze_medical_image", "get_patient_context", "submit_answer"],
        # With similar case search
        ["analyze_medical_image", "get_patient_context", "search_similar_cases", "submit_answer"],
        # Comprehensive
        ["think", "analyze_medical_image", "get_image_report", "get_patient_context", "search_imaging_knowledge", "submit_answer"],
        # With comparison
        ["analyze_medical_image", "get_patient_context", "compare_with_prior", "submit_answer"],
    ]

    for task in tasks:
        ticket = task.get("ticket", "")
        correct_answer = task.get("correct_answer", "")
        image_id = task.get("image_id", "IMG001")
        patient_id = task.get("patient_id", "PAT001")

        for pattern_idx, pattern in enumerate(patterns):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticket},
            ]

            for tool_name in pattern:
                if tool_name == "analyze_medical_image":
                    args = {"image_id": image_id}
                    response = "[Image analysis complete. Key findings identified.]"
                elif tool_name == "get_patient_context":
                    args = {"patient_id": patient_id}
                    response = "[Patient context retrieved.]"
                elif tool_name == "search_similar_cases":
                    args = {"query": ticket[:100]}
                    response = "[Similar cases found.]"
                elif tool_name == "get_image_report":
                    args = {"image_id": image_id}
                    response = "[Radiology report retrieved.]"
                elif tool_name == "search_imaging_knowledge":
                    args = {"query": correct_answer[:80] if correct_answer else "imaging findings"}
                    response = "[Relevant imaging knowledge found.]"
                elif tool_name == "compare_with_prior":
                    args = {"image_id": image_id}
                    response = "[No prior images available for comparison.]"
                elif tool_name == "think":
                    args = {"thought": f"I need to analyze this medical image carefully. {ticket[:100]}"}
                    response = "Thought recorded."
                elif tool_name == "submit_answer":
                    args = {
                        "answer": correct_answer or "Findings consistent with clinical presentation",
                        "reasoning": f"Based on image analysis and clinical context, {correct_answer}",
                    }
                    response = "[Answer submitted.]"
                else:
                    args = {}
                    response = f"[{tool_name} completed.]"

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"name": tool_name, "arguments": args}),
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{response}",
                })

            examples.append({
                "messages": messages,
                "metadata": {
                    "source": "synthetic_trajectory",
                    "domain": "visual_diagnosis",
                    "task_id": task["id"],
                    "pattern": f"pattern_{pattern_idx}",
                },
            })

    return examples


def generate_drug_interaction_sft(tasks: list[dict]) -> list[dict]:
    """Generate synthetic SFT for drug interaction tasks."""
    examples = []
    system_prompt = DOMAIN_SYSTEM_PROMPTS["drug_interaction"]

    patterns = [
        # Standard check
        ["get_patient_medications", "check_interaction", "get_drug_info", "submit_answer"],
        # Comprehensive
        ["get_patient_medications", "check_all_interactions", "get_drug_info", "search_alternatives", "submit_answer"],
        # With thinking
        ["think", "get_patient_medications", "check_interaction", "check_dosage", "submit_answer"],
        # Alternative-focused
        ["get_patient_medications", "check_interaction", "search_alternatives", "search_drugs_by_class", "submit_answer"],
    ]

    for task in tasks:
        ticket = task.get("ticket", "")
        patient_id = task.get("patient_id", "DI_P001")
        new_drug = task.get("new_drug", "")
        correct_answer = task.get("correct_answer", "")

        import re
        pid_match = re.search(r'(DI_P\d+)', ticket)
        if pid_match:
            patient_id = pid_match.group(1)

        for pattern_idx, pattern in enumerate(patterns):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticket},
            ]

            for tool_name in pattern:
                if tool_name == "get_patient_medications":
                    args = {"patient_id": patient_id}
                    response = "[Current medications retrieved.]"
                elif tool_name == "check_interaction":
                    args = {"drug_a": new_drug or "drug_a", "drug_b": "drug_b"}
                    response = "[Interaction check complete.]"
                elif tool_name == "check_all_interactions":
                    args = {"patient_id": patient_id}
                    response = "[All interactions checked.]"
                elif tool_name == "get_drug_info":
                    args = {"drug_name": new_drug or "warfarin"}
                    response = "[Drug information retrieved.]"
                elif tool_name == "check_dosage":
                    args = {"drug_name": new_drug or "warfarin", "patient_id": patient_id}
                    response = "[Dosage check complete.]"
                elif tool_name == "search_alternatives":
                    args = {"drug_name": new_drug or "warfarin", "reason": "interaction"}
                    response = "[Alternative medications found.]"
                elif tool_name == "search_drugs_by_class":
                    args = {"drug_class": "anticoagulant"}
                    response = "[Drugs in class found.]"
                elif tool_name == "think":
                    args = {"thought": f"I need to assess drug interactions for {patient_id}. {ticket[:100]}"}
                    response = "Thought recorded."
                elif tool_name == "submit_answer":
                    args = {
                        "answer": correct_answer or "Drug interaction assessment complete",
                        "reasoning": "Based on medication review and interaction analysis",
                    }
                    response = "[Answer submitted.]"
                else:
                    args = {}
                    response = f"[{tool_name} completed.]"

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"name": tool_name, "arguments": args}),
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{response}",
                })

            examples.append({
                "messages": messages,
                "metadata": {
                    "source": "synthetic_trajectory",
                    "domain": "drug_interaction",
                    "task_id": task["id"],
                    "pattern": f"pattern_{pattern_idx}",
                },
            })

    return examples


def generate_ehr_sft(tasks: list[dict]) -> list[dict]:
    """Generate synthetic SFT for EHR management tasks."""
    examples = []
    system_prompt = DOMAIN_SYSTEM_PROMPTS["ehr_management"]

    patterns = [
        # Standard review
        ["get_patient_summary", "get_lab_results", "get_vital_signs", "get_clinical_scores", "submit_answer"],
        # Comprehensive
        ["get_patient_summary", "get_lab_results", "get_lab_trend", "get_vital_signs", "detect_vital_alerts", "get_medication_orders", "get_clinical_scores", "submit_answer"],
        # With thinking
        ["think", "get_patient_summary", "get_lab_results", "get_vital_signs", "get_clinical_scores", "submit_answer"],
        # Discharge focus
        ["get_patient_summary", "get_admission_history", "get_lab_results", "get_vital_signs", "get_discharge_summary", "get_quality_indicators", "submit_answer"],
    ]

    for task in tasks:
        ticket = task.get("ticket", "")
        patient_id = task.get("patient_id", "P2001")
        hadm_id = task.get("hadm_id", "HADM_10001")
        expected_answer = task.get("expected_answer", "")

        for pattern_idx, pattern in enumerate(patterns):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticket},
            ]

            for tool_name in pattern:
                if tool_name in ("get_patient_summary", "get_admission_history"):
                    args = {"patient_id": patient_id}
                elif tool_name in ("get_lab_results", "get_lab_trend", "get_vital_signs", "get_medication_orders", "get_procedures"):
                    args = {"patient_id": patient_id, "hadm_id": hadm_id}
                elif tool_name == "detect_vital_alerts":
                    args = {"patient_id": patient_id, "hadm_id": hadm_id}
                elif tool_name == "get_clinical_scores":
                    args = {"patient_id": patient_id, "hadm_id": hadm_id}
                elif tool_name == "get_discharge_summary":
                    args = {"patient_id": patient_id, "hadm_id": hadm_id}
                elif tool_name == "get_quality_indicators":
                    args = {"patient_id": patient_id, "hadm_id": hadm_id}
                elif tool_name == "lookup_icd_code":
                    args = {"query": "heart failure"}
                elif tool_name == "think":
                    args = {"thought": f"Reviewing EHR for patient {patient_id}. {ticket[:150]}"}
                elif tool_name == "submit_answer":
                    args = {
                        "answer": expected_answer or "Clinical assessment complete",
                        "reasoning": "Based on comprehensive EHR review",
                    }
                else:
                    args = {}

                response = f"[{tool_name} result returned successfully]"

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"name": tool_name, "arguments": args}),
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{response}",
                })

            examples.append({
                "messages": messages,
                "metadata": {
                    "source": "synthetic_trajectory",
                    "domain": "ehr_management",
                    "task_id": task["id"],
                    "pattern": f"pattern_{pattern_idx}",
                },
            })

    return examples


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Main pipeline
# ═══════════════════════════════════════════════════════════════════════


def load_domain_tasks(domain: str, split: str = "train") -> list[dict]:
    """Load tasks for a domain, filtered to the specified split (default: train).

    IMPORTANT: Only use train-split tasks for SFT data generation to prevent
    data contamination. Test-split tasks must NEVER appear in training data.
    """
    path = PROJECT_ROOT / "data" / "domains" / domain / "tasks.json"
    if not path.exists():
        logger.warning(f"No tasks.json for {domain}")
        return []
    with open(path) as f:
        all_tasks = json.load(f)

    # Filter by split to prevent contamination
    split_path = PROJECT_ROOT / "data" / "domains" / domain / "split_tasks.json"
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
        if split in splits:
            valid_ids = set(splits[split])
            filtered = [t for t in all_tasks if t["id"] in valid_ids]
            logger.info(
                f"[{domain}] Loaded {len(filtered)}/{len(all_tasks)} tasks "
                f"(split={split}, filtered {len(all_tasks)-len(filtered)} test tasks)"
            )
            return filtered
        else:
            logger.warning(f"[{domain}] Split '{split}' not found in split_tasks.json, using all tasks")

    return all_tasks


def main():
    parser = argparse.ArgumentParser(description="P2 Enhanced SFT Data Generation")
    parser.add_argument("--target-count", type=int, default=550,
                        help="Target number of examples (default: 550)")
    parser.add_argument("--min-score", type=float, default=0.3,
                        help="Min action score for trajectory inclusion (default: 0.3)")
    parser.add_argument("--max-qa", type=int, default=250,
                        help="Max QA examples from medical_qa_200 (default: 250)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    print("\n" + "=" * 80)
    print("  P2-1: Enhanced Multi-Domain SFT Data Generation")
    print("  Target: 500+ diverse examples across 5 domains")
    print("=" * 80)

    all_examples = []
    domain_counts = Counter()

    # ── Step 1: Extract expert trajectories (lower threshold) ─────────
    print("\n  Step 1: Extracting expert trajectories (min_score={:.2f})...".format(args.min_score))
    trajectories = extract_expert_trajectories(min_action_score=args.min_score)
    all_examples.extend(trajectories)
    for ex in trajectories:
        domain_counts[ex["metadata"]["domain"]] += 1
    print(f"  → {len(trajectories)} trajectory examples")
    traj_domains = Counter(ex["metadata"]["domain"] for ex in trajectories)
    for d, c in sorted(traj_domains.items()):
        print(f"    {d}: {c}")

    # ── Step 2: Synthetic QA from medical_qa TRAIN split only ──────
    # CRITICAL: Only use train-split tasks to prevent contamination.
    # medical_qa_200 is excluded because it was generated from test benchmark files.
    print(f"\n  Step 2: Generating medical QA examples (max={args.max_qa}, TRAIN split only)...")
    qa_tasks_50 = load_domain_tasks("medical_qa", split="train")  # Explicit train split
    # Also load scaled train tasks if available
    qa_scaled_path = PROJECT_ROOT / "data" / "domains" / "medical_qa" / "tasks_scaled.json"
    qa_scaled = []
    if qa_scaled_path.exists():
        with open(qa_scaled_path) as f:
            qa_scaled_raw = json.load(f)
        # Filter scaled tasks to train split only
        split_path = PROJECT_ROOT / "data" / "domains" / "medical_qa" / "split_tasks.json"
        if split_path.exists():
            with open(split_path) as f:
                splits = json.load(f)
            train_ids = set(splits.get("train", []))
            qa_scaled = [t for t in qa_scaled_raw if t["id"] in train_ids]
        else:
            qa_scaled = qa_scaled_raw
    all_qa_tasks = qa_tasks_50 + qa_scaled
    # De-duplicate by ID
    seen_ids = set()
    unique_qa = []
    for t in all_qa_tasks:
        if t["id"] not in seen_ids:
            seen_ids.add(t["id"])
            unique_qa.append(t)
    random.shuffle(unique_qa)
    print(f"  → Using {len(unique_qa)} clean train tasks (excluded medical_qa_200 contaminated data)")
    qa_examples = generate_medical_qa_sft(unique_qa[:args.max_qa])
    all_examples.extend(qa_examples)
    domain_counts["medical_qa"] += len(qa_examples)
    print(f"  → {len(qa_examples)} QA examples from {len(unique_qa)} unique tasks")

    # ── Step 3: Synthetic trajectories for other domains ──────────────
    print("\n  Step 3: Generating domain-specific synthetic trajectories...")

    # Clinical diagnosis
    cdx_tasks = load_domain_tasks("clinical_diagnosis")
    cdx_examples = generate_clinical_dx_sft(cdx_tasks)
    all_examples.extend(cdx_examples)
    domain_counts["clinical_diagnosis"] += len(cdx_examples)
    print(f"  → clinical_diagnosis: {len(cdx_examples)} (from {len(cdx_tasks)} tasks × {len(cdx_examples)//max(len(cdx_tasks),1)} patterns)")

    # Visual diagnosis
    vdx_tasks = load_domain_tasks("visual_diagnosis")
    vdx_examples = generate_visual_dx_sft(vdx_tasks)
    all_examples.extend(vdx_examples)
    domain_counts["visual_diagnosis"] += len(vdx_examples)
    print(f"  → visual_diagnosis: {len(vdx_examples)} (from {len(vdx_tasks)} tasks × {len(vdx_examples)//max(len(vdx_tasks),1)} patterns)")

    # Drug interaction
    di_tasks = load_domain_tasks("drug_interaction")
    di_examples = generate_drug_interaction_sft(di_tasks)
    all_examples.extend(di_examples)
    domain_counts["drug_interaction"] += len(di_examples)
    print(f"  → drug_interaction: {len(di_examples)} (from {len(di_tasks)} tasks × {len(di_examples)//max(len(di_tasks),1)} patterns)")

    # EHR management
    ehr_tasks = load_domain_tasks("ehr_management")
    ehr_examples = generate_ehr_sft(ehr_tasks)
    all_examples.extend(ehr_examples)
    domain_counts["ehr_management"] += len(ehr_examples)
    print(f"  → ehr_management: {len(ehr_examples)} (from {len(ehr_tasks)} tasks × {len(ehr_examples)//max(len(ehr_tasks),1)} patterns)")

    # ── Step 4: De-duplication and balancing ──────────────────────────
    print("\n  Step 4: De-duplication and balancing...")

    # De-duplicate: only remove exact same (domain, task_id, pattern/strategy, model)
    # Keep trajectory examples from different models, and synthetic from different patterns
    seen_keys = set()
    unique_examples = []
    for ex in all_examples:
        meta = ex["metadata"]
        key = (
            meta.get("domain"),
            meta.get("task_id"),
            meta.get("pattern", meta.get("strategy", "")),
            meta.get("model", meta.get("source", "")),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            unique_examples.append(ex)

    print(f"  Before dedup: {len(all_examples)}, after: {len(unique_examples)}")

    # Shuffle
    random.shuffle(unique_examples)

    # ── Step 5: Statistics ────────────────────────────────────────────
    print("\n  Step 5: Dataset statistics...")
    final_domain_counts = Counter(ex["metadata"]["domain"] for ex in unique_examples)
    source_counts = Counter(ex["metadata"].get("source", "unknown") for ex in unique_examples)

    total_messages = sum(len(ex["messages"]) for ex in unique_examples)
    avg_messages = total_messages / max(len(unique_examples), 1)
    total_chars = sum(
        sum(len(m.get("content", "")) for m in ex["messages"])
        for ex in unique_examples
    )

    print(f"  Total examples: {len(unique_examples)}")
    print(f"  By domain:")
    for d, c in sorted(final_domain_counts.items()):
        print(f"    {d}: {c}")
    print(f"  By source:")
    for s, c in sorted(source_counts.items()):
        print(f"    {s}: {c}")
    print(f"  Avg messages/example: {avg_messages:.1f}")
    print(f"  Total characters: {total_chars:,}")

    # ── Step 6: Save ─────────────────────────────────────────────────
    output_path = args.output or str(OUTPUT_DIR / "p2_multidomain_sft.jsonl")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Step 6: Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats file
    stats = {
        "timestamp": datetime.now().isoformat(),
        "phase": "P2",
        "total_examples": len(unique_examples),
        "domain_counts": dict(final_domain_counts),
        "source_counts": dict(source_counts),
        "avg_messages_per_example": round(avg_messages, 1),
        "total_characters": total_chars,
        "min_action_score": args.min_score,
        "max_qa": args.max_qa,
    }
    stats_path = Path(output_path).with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats_path}")
    print(f"\n  {'✅' if len(unique_examples) >= 500 else '⚠️'} Generated {len(unique_examples)} examples (target: 500+)")

    # ── Step 7: Auto contamination check (BUGLOG BUG-002 prevention) ──
    print("\n  Step 7: Contamination check...")
    test_ids = _load_all_test_ids()
    contaminated = [
        ex for ex in unique_examples
        if ex.get("metadata", {}).get("task_id", "") in test_ids
    ]
    if contaminated:
        print(f"  ✗ CONTAMINATION DETECTED: {len(contaminated)} examples contain test task IDs!")
        print(f"    IDs: {[e['metadata']['task_id'] for e in contaminated[:5]]}...")
        print(f"    This violates BUGLOG BUG-002. Fix the data pipeline before using this file!")
    else:
        print(f"  ✓ CLEAN: 0/{len(unique_examples)} examples overlap with test splits")

    print("=" * 80)


if __name__ == "__main__":
    main()
