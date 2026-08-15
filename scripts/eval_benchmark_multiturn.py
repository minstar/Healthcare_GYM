#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-turn benchmark evaluation using AgentRunner.

Runs standard medical benchmarks (MedQA, MedMCQA, MMLU, VQA-RAD, SLAKE, PathVQA)
through the full multi-turn agent loop matching the v16 training format:
  think → search_evidence / analyze_options → submit_answer

This properly evaluates RL-trained models that learned to use tools in multi-turn mode,
unlike single-turn eval which truncates before submit_answer.

Usage:
    # TextQA benchmarks (MedQA + MedMCQA + MMLU) on GPU 0
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmark_multiturn.py \
        --model_path /path/to/merged_hf \
        --benchmarks medqa medmcqa mmlu \
        --domain medical_qa \
        --output-dir results/benchmarks_multiturn/v16_step60 \
        --max-turns 5

    # VQA benchmarks on GPU 4
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_benchmark_multiturn.py \
        --model_path /path/to/merged_hf \
        --benchmarks vqa_rad slake pathvqa \
        --domain visual_diagnosis \
        --output-dir results/benchmarks_multiturn/v16_step60 \
        --max-turns 5

    # Limit samples for testing
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmark_multiturn.py \
        --model_path /path/to/merged_hf \
        --benchmarks medqa \
        --max-samples 50

    # Prompting baselines (eval-only, zero training) — --prompt-mode
    #   default      Base+AR (unchanged; this is the paper's existing condition)
    #   strong_tool  Base+AR + materially stronger tool-use contract
    #   react        explicit ReAct (Thought/Action/Action Input/Observation)
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmark_multiturn.py \
        --model_path /path/to/merged_hf \
        --benchmarks medqa \
        --prompt-mode react
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False  # Qwen3.5-VL Conv3D workaround

from rouge_score import rouge_scorer as _rouge_module
_rouge_scorer = _rouge_module.RougeScorer(["rougeL"], use_stemmer=True)

# ── Lazy-loaded biobert-nli for LFQA hallucination/comprehensiveness ──
_nli_model = None
_nli_tokenizer = None
_nli_device = None


def _ensure_nli_model():
    """Load biobert-nli model on first use (lazy init)."""
    global _nli_model, _nli_tokenizer, _nli_device
    if _nli_model is not None:
        return
    from transformers import AutoModel, AutoTokenizer
    _nli_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading biobert-nli for hallucination/comprehensiveness scoring...")
    _nli_model = AutoModel.from_pretrained("gsarti/biobert-nli").to(_nli_device)
    _nli_tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")
    _nli_model.eval()
    logger.info(f"biobert-nli loaded on {_nli_device}")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from loguru import logger

# Closed-form exact match for the visual-QA benchmarks. Kept in its own module
# so it can be unit-tested and re-run over stored rollouts without importing
# torch or loading a model.
sys.path.insert(0, str(Path(__file__).parent))
import vqa_scoring  # noqa: E402

# ── Benchmark file registry ──
BENCHMARK_FILES = {
    # TextQA (MC)
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
    # MMLU subtypes
    "mmlu_anatomy": "evaluations/self-biorag/data/benchmark/mmlu_anatomy_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
    "mmlu_professional": "evaluations/self-biorag/data/benchmark/mmlu_professional_medicine_test.jsonl",
    "mmlu_genetics": "evaluations/self-biorag/data/benchmark/mmlu_medical_genetics_test.jsonl",
    "mmlu_biology": "evaluations/self-biorag/data/benchmark/mmlu_college_biology_test.jsonl",
    "mmlu_college_med": "evaluations/self-biorag/data/benchmark/mmlu_college_medicine_test.jsonl",
    # MedLFQA (long-form)
    "kqa_golden": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
    "live_qa": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
    "medication_qa": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
    "healthsearch_qa": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
    "kqa_silver": "evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl",
    # VQA (loaded differently - via datasets)
    "vqa_rad": "datasets/vqa/vqa_rad",
    "slake": "datasets/vqa/slake",
    "pathvqa": "datasets/vqa/pathvqa",
    "pmc_vqa": "datasets/vqa/pmc_vqa",
    "vqa_med_2021": "datasets/vqa/vqa_med_2021",
    "quilt_vqa": "datasets/vqa/quilt_vqa",
    # EHR (loaded differently - from JSON with tasks array)
    "mimic_iii": "data/ehr_benchmarks/mimic_iii_bench.json",
    "eicu": "data/ehr_benchmarks/eicu_bench.json",
}

# Domain mapping for benchmarks
BENCHMARK_DOMAIN = {
    "medqa": "medical_qa",
    "medmcqa": "medical_qa",
    "mmlu": "medical_qa",
    "mmlu_anatomy": "medical_qa",
    "mmlu_clinical": "medical_qa",
    "mmlu_professional": "medical_qa",
    "mmlu_genetics": "medical_qa",
    "mmlu_biology": "medical_qa",
    "mmlu_college_med": "medical_qa",
    "kqa_golden": "medical_qa",
    "live_qa": "medical_qa",
    "medication_qa": "medical_qa",
    "healthsearch_qa": "medical_qa",
    "kqa_silver": "medical_qa",
    "vqa_rad": "visual_diagnosis",
    "slake": "visual_diagnosis",
    "pathvqa": "visual_diagnosis",
    "pmc_vqa": "visual_diagnosis",
    "vqa_med_2021": "visual_diagnosis",
    "quilt_vqa": "visual_diagnosis",
    "mimic_iii": "ehr_management",
    "eicu": "ehr_management",
}


def load_textqa_benchmark(name: str) -> list[dict]:
    """Load a TextQA benchmark from JSONL and convert to task format."""
    filepath = PROJECT_ROOT / BENCHMARK_FILES[name]
    if not filepath.exists():
        logger.error(f"Benchmark file not found: {filepath}")
        return []

    # MedLFQA benchmarks
    MEDLFQA_BENCHMARKS = {"kqa_golden", "live_qa", "medication_qa", "healthsearch_qa", "kqa_silver"}

    tasks = []
    with open(filepath) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)

            # Handle MedLFQA format (Question / Free_form_answer)
            must_have = []
            nice_to_have = []
            if name in MEDLFQA_BENCHMARKS:
                question = item.get("Question", "")
                answer = item.get("Free_form_answer", "").strip()
                must_have = item.get("Must_have", [])
                nice_to_have = item.get("Nice_to_have", [])
            else:
                instances = item.get("instances", {})
                question = instances.get("input", "")
                answer = instances.get("output", "").strip()

            # Extract options from question text.
            #
            # The self-biorag export ends every MMLU question with a dangling
            # "\nOption: " sentinel (1,089/1,089 rows). The lookahead below
            # stops at "Option [A-E]:" or end-of-string, NEITHER of which
            # matches the bare sentinel, so it was absorbed into the LAST
            # option's text -- and any gold equal to that option could never
            # match a letter again: 352/1,089 golds were unmappable for every
            # arm, depressing MMLU accuracy 18-30pp arm-dependently.
            # Strip it for OPTION PARSING ONLY: the prompt (`ticket`) keeps the
            # original text, so pre-fix and post-fix runs stay prompt-identical
            # and stored runs can be rescored without re-inference.
            opt_src = re.sub(r"Option\s*:\s*$", "", question.rstrip())
            options = {}
            for letter in "ABCDE":
                pat = rf"Option {letter}:\s*(.+?)(?=Option [A-E]:|$)"
                m = re.search(pat, opt_src, re.DOTALL)
                if m:
                    options[letter] = m.group(1).strip()

            task = {
                "id": f"{name}_{idx}",
                "description": {
                    "purpose": f"Answer {name} question",
                    "difficulty": "medium",
                    "category": name,
                },
                "ticket": question,
                "correct_answer": answer,
                "answer": answer,
                "options": options,
                "must_have": must_have,
                "nice_to_have": nice_to_have,
                "evaluation_criteria": {
                    "actions": [
                        {"name": "submit_answer", "arguments": {"answer": answer}}
                    ],
                    "nl_assertions": [],
                    "reward_basis": ["ACTION"],
                },
            }
            tasks.append(task)

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def load_vqa_benchmark(name: str) -> list[dict]:
    """Load a VQA benchmark and convert to task format."""
    data_dir = PROJECT_ROOT / BENCHMARK_FILES[name]

    # Try to load test split from HuggingFace datasets cache or local JSON
    test_file = data_dir / "test.json"
    if not test_file.exists():
        test_file = data_dir / "test.jsonl"
    if not test_file.exists():
        # Try loading via datasets library
        try:
            return _load_vqa_from_datasets(name, data_dir)
        except Exception as e:
            logger.error(f"Cannot load VQA benchmark {name}: {e}")
            return []

    tasks = []
    with open(test_file) as f:
        if test_file.suffix == ".jsonl":
            items = [json.loads(l) for l in f if l.strip()]
        else:
            items = json.load(f)

    for idx, item in enumerate(items):
        question = item.get("question", item.get("input", ""))
        answer = str(item.get("answer", item.get("output", ""))).strip()
        image_path = item.get("image_path", item.get("image", ""))

        # Make image path absolute if relative
        if image_path and not Path(image_path).is_absolute():
            image_path = str(data_dir / "images" / image_path)

        task = {
            "id": f"{name}_{idx}",
            "description": {
                "purpose": f"Answer {name} visual question",
                "difficulty": "medium",
                "category": name,
            },
            "ticket": question,
            "correct_answer": answer,
            "answer": answer,
            "_image_path": image_path if image_path else None,
            "evaluation_criteria": {
                "actions": [
                    {"name": "submit_answer", "arguments": {"answer": answer}}
                ],
                "nl_assertions": [],
                "reward_basis": ["ACTION"],
            },
        }
        tasks.append(task)

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def load_ehr_benchmark(name: str) -> list[dict]:
    """Load an EHR benchmark from JSON (already in task format)."""
    filepath = PROJECT_ROOT / BENCHMARK_FILES[name]
    if not filepath.exists():
        logger.error(f"EHR benchmark file not found: {filepath}")
        return []

    with open(filepath) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    # EHR tasks are already in AgentRunner task format
    # Add correct_answer field if missing
    for task in tasks:
        if "correct_answer" not in task:
            task["correct_answer"] = ""
        if "answer" not in task:
            task["answer"] = ""

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def _load_vqa_from_datasets(name: str, data_dir: Path) -> list[dict]:
    """Load VQA benchmark using the datasets library or raw files."""
    # Check for commonly used file patterns
    for pattern in ["*test*.json", "*test*.jsonl", "*test*.csv"]:
        matches = list(data_dir.glob(pattern))
        if matches:
            logger.info(f"Found {matches[0]} for {name}")
            with open(matches[0]) as f:
                if matches[0].suffix == ".jsonl":
                    items = [json.loads(l) for l in f if l.strip()]
                else:
                    items = json.load(f)
            tasks = []
            for idx, item in enumerate(items):
                q = item.get("question", item.get("input", ""))
                a = str(item.get("answer", item.get("output", ""))).strip()
                tasks.append({
                    "id": f"{name}_{idx}",
                    "ticket": q,
                    "correct_answer": a,
                    "answer": a,
                    "description": {"purpose": f"{name} VQA", "category": name},
                    "evaluation_criteria": {
                        "actions": [{"name": "submit_answer"}],
                        "nl_assertions": [],
                        "reward_basis": ["ACTION"],
                    },
                })
            return tasks
    raise FileNotFoundError(f"No test files found in {data_dir}")


def _run_single_task_multiturn(runner, task, env, max_turns):
    """Run a single task with multi-turn loop + forced submit on last turn.

    Unlike AgentRunner.run_task(), this injects a nudge message before the
    LAST turn (turn_idx == max_turns-1) telling the model it MUST submit now.
    Note the timing: the model gets exactly one generation after the nudge and
    no turn in which to act if it ignores it, so at max_turns=5 the whole
    commitment decision rides on a single generation at a hard cap. This is
    arm-symmetric but makes the no_answer_rate metric maximally sensitive;
    do not compare arms across different max_turns settings.
    """
    from bioagents.evaluation.agent_runner import (
        TurnRecord, TaskResult, build_system_prompt,
        parse_tool_call_with_format, format_assistant_turn,
        format_tool_observation, native_tools_for_prompt_mode,
    )

    prompt_mode = getattr(runner.config, "prompt_mode", "default")

    task_id = task["id"]

    # Reset environment
    obs, info = env.reset(options={"task_id": task_id})

    # Build conversation
    tools_for_prompt = info["tools"]
    if runner.config.no_think and tools_for_prompt:
        tools_for_prompt = [
            t for t in tools_for_prompt
            if t.get("function", {}).get("name") != "think"
        ]

    system_prompt = build_system_prompt(
        info["policy"], tools_for_prompt,
        domain=runner.config.domain, task=task,
        prompt_mode=prompt_mode,
    )
    # Build user message — include image for VQA tasks
    image_path = task.get("_image_path")
    if image_path and os.path.exists(image_path):
        user_content = [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": obs},
        ]
    else:
        user_content = obs

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    # default / strong_tool get the catalog through the chat template, as
    # before. react gets it as prompt text (build_system_prompt above) and
    # withholds tools= here, so the template's native tool-calling contract —
    # the contract that beat ReAct on 4,418 of 4,429 turns — is not injected
    # alongside it. Same tools either way; only the emission contract differs.
    openai_tools = native_tools_for_prompt_mode(tools_for_prompt, prompt_mode)

    turns = []
    submitted_answer = ""
    t_total = 0.0

    for turn_idx in range(max_turns):
        # On penultimate turn, inject submit nudge
        if turn_idx == max_turns - 1 and not submitted_answer:
            messages.append({
                "role": "user",
                "content": (
                    "IMPORTANT: This is your LAST turn. You MUST call submit_answer now "
                    "with your best answer based on the information you have gathered. "
                    "Do NOT call any other tool. Call submit_answer immediately."
                ),
            })

        t0 = time.time()
        raw_output = runner.generate(messages, tools=openai_tools)
        latency = time.time() - t0
        t_total += latency

        turn = TurnRecord(turn_idx=turn_idx, raw_output=raw_output, latency_seconds=latency)

        # Parse tool call, recording which format branch accepted it so
        # prompt_mode="react" adherence is measurable rather than assumed.
        tool_call, tool_fmt = parse_tool_call_with_format(raw_output)
        turn.tool_call_format = tool_fmt

        if tool_call is not None:
            turn.parsed_tool_call = tool_call
            tool_name = tool_call.get("name", "")

            # Execute tool via environment
            action = json.dumps(tool_call)
            observation, reward, terminated, truncated, step_info = env.step(action)

            if isinstance(observation, (dict, list)):
                observation_str = json.dumps(observation, indent=2, ensure_ascii=False)
            else:
                observation_str = str(observation) if observation is not None else ""

            turn.tool_response = observation_str
            # react mode replays the assistant turn verbatim and returns the
            # result as "Observation:", matching what its system prompt
            # promises. default / strong_tool are byte-for-byte unchanged.
            messages.append({
                "role": "assistant",
                "content": format_assistant_turn(raw_output, tool_call, prompt_mode),
            })
            messages.append({
                "role": "user",
                "content": format_tool_observation(
                    tool_name, observation_str, prompt_mode
                ),
            })
            turns.append(turn)

            if tool_name == "submit_answer":
                # str(), because the model writes this field and nothing upstream
                # constrains its JSON type. `submit_answer({"answer": 2})` parses
                # to an int, and every consumer downstream assumes text:
                # _check_answer does submitted.strip(), _compute_rouge_l does
                # .lower(). The AttributeError is caught by the per-task handler,
                # which records the task as incorrect with turns=0 -- so a model
                # that answered gets scored wrong, silently, and the zero lands in
                # both the accuracy numerator and avg_turns.
                #
                # It does not cancel between conditions: on the same 1061-item
                # slake set base_strong_tool hit it 15 times and base_react never,
                # which is up to ~1.4 pp of arm-specific loss. Coerce here, at the
                # single point where the model's answer enters, rather than at
                # each consumer.
                submitted_answer = str(tool_call.get("arguments", {}).get("answer", ""))
                if not submitted_answer.strip():
                    # A submit_answer whose <parameter=answer> block never
                    # closes (missing </parameter>) parses to arguments={}:
                    # the model DID answer and the letter was silently dropped,
                    # then the fallback recorded the raw XML as the answer and
                    # scored it wrong. 197 such rows in the stored campaign,
                    # 165 of them correct once recovered -- and every one on a
                    # single arm's side of a comparison, so it does not cancel.
                    # Recover from the raw turn here, at the same single entry
                    # point the str() coercion above guards. The caller records
                    # answer_source="submit_answer_recovered" for these rows,
                    # so the strict convention (malformed call = failure)
                    # remains reportable from the same artifact.
                    m = re.search(
                        r"<parameter=answer>\s*(.*?)\s*(?:</parameter>|</function>|</tool_call>|\Z)",
                        raw_output, re.DOTALL)
                    if m and m.group(1).strip():
                        submitted_answer = m.group(1).strip()
                break

            if terminated or truncated:
                break

            # Detect repetition
            if len(turns) >= 3:
                recent_names = [
                    t.parsed_tool_call.get("name", "") if t.parsed_tool_call else ""
                    for t in turns[-3:]
                ]
                if len(set(recent_names)) == 1 and recent_names[0] not in ("submit_answer", ""):
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have called '{recent_names[0]}' multiple times. "
                            "Please submit your final answer now using submit_answer."
                        ),
                    })
        else:
            # A REJECTED tool call is not a final answer. The parser requires a
            # closed `</function></tool_call>`; a call truncated at the token
            # cap has no terminator, and treating it as plain-text ended the
            # episode with the model's THINKING recorded as its submission --
            # 321 such episodes across 40 pf transcript files, at 10x
            # different rates per arm (fmtmatch 79-89/file vs 6-20 elsewhere),
            # so the bias does not subtract out of arm comparisons. Give the
            # model the same corrective the repetition guard gives, and let
            # the episode continue; only XML-free prose is a final answer.
            if "<tool_call>" in raw_output or "<function=" in raw_output:
                turn.parse_error = True
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your tool call was malformed or truncated and could not "
                        "be executed. Emit ONE complete, well-formed tool call, "
                        "or call submit_answer with your final answer."
                    ),
                })
                turns.append(turn)
                continue
            turn.is_final_answer = True
            messages.append({"role": "assistant", "content": raw_output})
            turns.append(turn)
            break

    return turns, submitted_answer, t_total, env._tool_call_log


def _run_single_task_multiturn_debug(runner, task, env, max_turns):
    """Same as _run_single_task_multiturn but also returns the final message list.

    Used only by scripts/rebuttal/verify_react_transcript.py, which has to show
    the EXACT transcript the model saw. Kept as a thin wrapper (rather than a
    flag on the hot path) so the eval path is untouched.
    """
    seen: list = []
    original = runner.generate

    def _spy(messages, tools=None):
        seen.append(json.loads(json.dumps(messages, default=str)))
        return original(messages, tools=tools)

    runner.generate = _spy
    try:
        out = _run_single_task_multiturn(runner, task, env, max_turns)
    finally:
        runner.generate = original
    return out, (seen[-1] if seen else [])


def run_benchmark_multiturn(
    benchmark_name: str,
    tasks: list[dict],
    runner,
    domain: str,
    max_turns: int,
    output_dir: Path,
    resume_from: int = 0,
    vqa_scorer: str = "cf_em",
    prior_results: list[dict] | None = None,
    dump_transcripts: bool = False,
):
    """Run a benchmark through multi-turn AgentRunner loop.

    `vqa_scorer` selects the scoring rule for the CLOSED-VOCABULARY visual-QA
    benchmarks only (vqa_rad, slake, pathvqa):
        "cf_em"     -- closed-form exact match (default; see scripts/vqa_scoring.py)
        "substring" -- the pre-2026-07-29 rule, kept reachable so the published
                       numbers stay reproducible for the rebuttal table
    Every other benchmark (text QA, long-form QA, EHR, and the open-vocabulary
    VQA sets) is scored exactly as before, by an unmodified `_check_answer`.

    Returns:
        dict with accuracy, avg_turns, avg_reward, per-sample results
    """
    from bioagents.gym.agent_env import BioAgentGymEnv
    from bioagents.evaluation.agent_runner import (
        summarize_format_adherence, aggregate_format_adherence,
        multimodal_tool_forwarding, reset_multimodal_request_stats,
    )

    LFQA_BENCHMARKS = {"kqa_golden", "live_qa", "medication_qa", "healthsearch_qa", "kqa_silver"}
    EHR_BENCHMARKS_SET = {"mimic_iii", "eicu"}
    is_lfqa = benchmark_name in LFQA_BENCHMARKS
    is_ehr = benchmark_name in EHR_BENCHMARKS_SET

    # --- visual-QA scoring rule -------------------------------------------
    # CF-EM applies ONLY to the closed-vocabulary visual-QA sets. The
    # vocabulary is built from the FULL benchmark file, never from the
    # (possibly --max-samples-limited) task list, so a score does not depend
    # on how many samples were run.
    cf_vocab = None
    if vqa_scorer == "cf_em" and benchmark_name in vqa_scoring.CLOSED_VOCAB_BENCHMARKS:
        cf_vocab = vqa_scoring.load_vocab(benchmark_name, PROJECT_ROOT)
        if cf_vocab is None:
            logger.warning(
                f"  [{benchmark_name}] CF-EM requested but the benchmark file "
                f"could not be loaded; falling back to the substring rule."
            )
    scoring_rule = "cf_em" if cf_vocab is not None else "substring"
    if benchmark_name in vqa_scoring.VQA_BENCHMARKS:
        logger.info(
            f"  [{benchmark_name}] VQA scoring rule = {scoring_rule}"
            + (f" ({vqa_scoring.CF_EM_VERSION}, "
               f"{len(cf_vocab.labels)} non-polar labels over "
               f"{len(cf_vocab.items)} benchmark items)" if cf_vocab is not None
               else " (pre-2026-07-29 substring containment)")
        )

    prompt_mode = getattr(runner.config, "prompt_mode", "default")
    adherence_per_task: list[dict] = []

    # Per-benchmark, so the counts reported below belong to THIS benchmark and
    # a text-only benchmark run after a VQA one still records 0 image requests.
    reset_multimodal_request_stats()

    # Rows carried over from an interrupted run. _save_partial rewrites the file
    # from `results` alone, so anything not seeded here is destroyed on the first
    # save rather than resumed.
    results = list(prior_results or [])
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    rouge_l_sum = sum(r.get("rouge_l") or 0.0 for r in results)
    # hallucination/comprehensiveness are None on unanswered rows -- they
    # aggregate over ANSWERED rows only, so counts travel with the sums.
    hall_sum = sum(r["hallucination"] for r in results
                   if r.get("hallucination") is not None)
    hall_n = sum(1 for r in results if r.get("hallucination") is not None)
    comp_sum = sum(r["comprehensiveness"] for r in results
                   if r.get("comprehensiveness") is not None)
    comp_n = sum(1 for r in results if r.get("comprehensiveness") is not None)
    action_score_sum = sum(r.get("action_score", 0.0) for r in results)
    adherence_per_task.extend(
        r["format_adherence"] for r in results if isinstance(r.get("format_adherence"), dict)
    )
    t_start = time.time()

    for i, task in enumerate(tasks):
        if i < resume_from:
            continue

        # Create fresh env for each task
        env = BioAgentGymEnv(domain=domain, max_turns=max_turns)

        # Inject task into env's task map
        env._task_map[task["id"]] = task
        env._tasks.append(task)

        try:
            turns, submitted, latency, tool_log = _run_single_task_multiturn(
                runner, task, env, max_turns
            )

            # Output-format adherence for this task. Recorded before scoring so
            # a react-arm number can never be reported without the evidence
            # that the model actually emitted ReAct.
            adherence = summarize_format_adherence(turns)
            adherence_per_task.append(adherence)

            if dump_transcripts:
                # raw_output lives only in the in-memory TurnRecord; without
                # this dump an unanswered episode can never be re-scored after
                # the fact -- the entire no-answer class of the 2026-08-14
                # campaign had to be re-run for exactly that reason. Appended
                # per task so a preempted job keeps what it paid for.
                with open(output_dir / f"{benchmark_name}_transcripts.jsonl",
                          "a", encoding="utf-8") as tf:
                    tf.write(json.dumps({
                        "task_id": task["id"],
                        "turns": [{
                            "turn_idx": t.turn_idx,
                            "raw_output": t.raw_output,
                            "tool": (t.parsed_tool_call or {}).get("name")
                                    if t.parsed_tool_call else None,
                            "is_final_answer": bool(getattr(t, "is_final_answer", False)),
                            # A rejected/truncated tool call the loop retried
                            # rather than accepting as an answer — without this
                            # the transcript cannot distinguish that turn kind.
                            "parse_error": bool(getattr(t, "parse_error", False)),
                        } for t in turns],
                    }, ensure_ascii=False) + "\n")

            # Where did the answer come from? Recorded per row so the strict
            # convention (recovered/malformed = failure) and cap-exhaustion
            # rates stay reportable from the artifact itself.
            capped = len(turns) >= max_turns
            answer_source = "none"
            if submitted:
                last_tc = turns[-1].parsed_tool_call if turns else None
                if last_tc and last_tc.get("name") == "submit_answer" and \
                        not str(last_tc.get("arguments", {}).get("answer", "")).strip():
                    answer_source = "submit_answer_recovered"
                else:
                    answer_source = "submit_answer"
            elif turns and getattr(turns[-1], "is_final_answer", False):
                # Only salvage a plain-text final message. A budget exhausted
                # on a non-submit tool call stays "" (no-answer sentinel) --
                # the old behavior mined the trailing tool-call XML instead,
                # which scored wrong automatically at an arm-dependent rate.
                submitted = _extract_answer_fallback(turns[-1].raw_output)
                answer_source = "final_text" if submitted else "none"

            gold = task["correct_answer"].strip()
            options = task.get("options", {})

            if is_ehr:
                # EHR: action-based scoring — check if expected tool calls were made
                expected_actions = task.get("evaluation_criteria", {}).get("actions", [])
                called_names = [
                    t.parsed_tool_call.get("name", "")
                    for t in turns if t.parsed_tool_call
                ]
                action_hits = 0
                for exp in expected_actions:
                    exp_name = exp.get("name", "")
                    if exp_name in called_names:
                        action_hits += 1
                action_score = action_hits / max(len(expected_actions), 1)
                action_score_sum += action_score
                is_correct = action_score >= 0.5  # at least half of expected actions called
                rouge_l = None
            elif is_lfqa:
                # LFQA: compute ROUGE-L on submitted answer vs gold
                rouge_l = _compute_rouge_l(submitted, gold)
                rouge_l_sum += rouge_l
                is_correct = rouge_l >= 0.3  # threshold for binary correct
                # Hallucination & comprehensiveness via biobert-nli
                must_have = task.get("must_have", [])
                nice_to_have = task.get("nice_to_have", [])
                hall = _compute_hallucination(submitted, must_have, nice_to_have)
                comp = _compute_comprehensiveness(submitted, must_have)
                if hall is not None:
                    hall_sum += hall
                    hall_n += 1
                if comp is not None:
                    comp_sum += comp
                    comp_n += 1
            elif cf_vocab is not None:
                # Closed-vocabulary VQA: score the ANSWER, not the transcript.
                # Both rules are computed on every row so the rebuttal table
                # showing "what it was / what it is" needs no second pass.
                row_idx = vqa_scoring.task_row_index(task["id"], benchmark_name)
                cf_pred, is_correct, cf_kind, cf_span = vqa_scoring.cf_predict(
                    submitted, gold, cf_vocab, idx=row_idx
                )
                substring_correct = _check_answer(submitted, gold, options)
                rouge_l = None
            else:
                # MC / open-vocabulary VQA: exact/letter match, unchanged
                is_correct = _check_answer(submitted, gold, options)
                rouge_l = None

            if is_correct:
                correct += 1
            total += 1

            result_entry = {
                "task_id": task["id"],
                "gold": gold,
                "submitted": submitted,
                "correct": is_correct,
                "answer_source": answer_source,
                "capped": capped,
                "turns": len(turns),
                "latency": latency,
                "react_rate": round(adherence["react_rate"], 4),
                "format_adherence": adherence,
            }
            if is_ehr:
                result_entry["action_score"] = round(action_score, 4)
                result_entry["actions_expected"] = len(expected_actions)
                result_entry["actions_hit"] = action_hits
                result_entry["tools_called"] = called_names
            elif is_lfqa:
                result_entry["rouge_l"] = round(rouge_l, 4)
                result_entry["hallucination"] = (None if hall is None
                                                 else round(hall, 2))
                result_entry["comprehensiveness"] = (None if comp is None
                                                     else round(comp, 2))
            elif cf_vocab is not None:
                # A VQA number that does not say how it was scored is worthless.
                result_entry["scored_by"] = vqa_scoring.CF_EM_VERSION
                result_entry["cf_correct"] = is_correct
                result_entry["cf_pred"] = sorted(cf_pred)
                result_entry["cf_kind"] = cf_kind
                result_entry["cf_span"] = cf_span
                result_entry["substring_correct"] = substring_correct
            results.append(result_entry)

            # Progress logging every 10 samples
            if total % 10 == 0:
                elapsed = time.time() - t_start
                rate = total / elapsed * 60
                eta = (len(tasks) - total) / max(rate, 0.01)
                if is_ehr:
                    avg_as = action_score_sum / total
                    acc = correct / total
                    logger.info(
                        f"  [{benchmark_name}] {total}/{len(tasks)} "
                        f"action_score={avg_as:.3f} acc={acc:.3f} "
                        f"rate={rate:.1f}/min ETA={eta:.0f}min"
                    )
                elif is_lfqa:
                    avg_rl = rouge_l_sum / total
                    avg_h = (hall_sum / hall_n) if hall_n else float("nan")
                    avg_c = (comp_sum / comp_n) if comp_n else float("nan")
                    logger.info(
                        f"  [{benchmark_name}] {total}/{len(tasks)} "
                        f"rouge_l={avg_rl:.3f} hall={avg_h:.1f}% comp={avg_c:.1f}% "
                        f"rate={rate:.1f}/min ETA={eta:.0f}min"
                    )
                else:
                    acc = correct / total
                    logger.info(
                        f"  [{benchmark_name}] {total}/{len(tasks)} "
                        f"acc={acc:.3f} rate={rate:.1f}/min ETA={eta:.0f}min"
                    )

            # Periodic save every 10 samples
            if total % 10 == 0:
                _save_partial(benchmark_name, results, correct, total, output_dir,
                              scoring_rule=(scoring_rule if benchmark_name
                                            in vqa_scoring.VQA_BENCHMARKS else None))

        except Exception as e:
            logger.error(f"Error on task {task['id']}: {e}")
            total += 1
            result_entry = {
                "task_id": task["id"],
                "gold": task["correct_answer"],
                "submitted": "",
                "correct": False,
                "answer_source": "error",
                "capped": False,
                "turns": 0,
                "error": str(e),
                "react_rate": 0.0,
                "format_adherence": {},
            }
            if is_ehr:
                result_entry["action_score"] = 0.0
                result_entry["actions_expected"] = 0
                result_entry["actions_hit"] = 0
                result_entry["tools_called"] = []
            elif is_lfqa:
                # An errored task produced no answer: rouge stays 0.0 (it
                # feeds `accuracy`, whose unanswered-scores-wrong semantics
                # are unchanged), but the answered-only quality metrics are
                # None -- an error must not count as 100% hallucination.
                result_entry["rouge_l"] = 0.0
                result_entry["hallucination"] = None
                result_entry["comprehensiveness"] = None
            elif cf_vocab is not None:
                # An errored task is wrong under BOTH rules, and must still say
                # which rule produced the number it contributes to.
                result_entry["scored_by"] = vqa_scoring.CF_EM_VERSION
                result_entry["cf_correct"] = False
                result_entry["cf_pred"] = []
                result_entry["cf_kind"] = "error"
                result_entry["cf_span"] = ""
                result_entry["substring_correct"] = False
            results.append(result_entry)

    elapsed = time.time() - t_start
    accuracy = correct / max(total, 1)

    format_adherence = aggregate_format_adherence(adherence_per_task)
    format_adherence["prompt_mode"] = prompt_mode

    summary = {
        "benchmark": benchmark_name,
        # Recorded so a results file can never be misattributed to the wrong
        # baseline condition when it is reported.
        "prompt_mode": prompt_mode,
        # Which arm of the multimodal tool-forwarding comparison produced this
        # number, and how many image requests actually carried the catalog.
        # On the image benchmarks (vqa_rad, slake, ...) "arm": "withheld" means
        # the agent had NO tools — the pre-2026-07-29 measurement, reproduced
        # on purpose. On a text-only benchmark multimodal_requests is 0, which
        # is the recorded proof that this switch could not have touched it.
        "multimodal_tool_forwarding": multimodal_tool_forwarding(),
        # Output-format adherence. Under prompt_mode="react" this is the
        # number the arm MUST be reported with. ReAct is now the ONLY
        # tool-calling contract in the prompt (the template's native contract
        # is suppressed; the catalog is carried as text), so react_rate no
        # longer measures a prompt clash — it measures whether the backbone
        # will adopt an instructed format at all. A low react score at
        # react_rate ~0 still means "the model ignored the format", not "ReAct
        # scaffolding does not help": opposite conclusions, and only this
        # number tells them apart.
        "format_adherence": format_adherence,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_turns": sum(r.get("turns", 0) for r in results) / max(len(results), 1),
        "avg_reward": sum(r.get("reward", 0) for r in results) / max(len(results), 1),
        "avg_latency": sum(r.get("latency", 0) for r in results) / max(len(results), 1),
        # Cap exhaustion and answer provenance as first-class metrics. The turn
        # budget binds on 87-100% of items depending on benchmark, so an arm's
        # score is inseparable from whether it commits an answer at the cap;
        # comparisons are only clean when the two arms' no_answer_rate differ
        # by <2pp (stat-gate reports CONFOUNDED otherwise). "extraction" tags
        # which rule produced `submitted`; pre-tag files mined the trailing
        # tool call on unanswered episodes and cannot be mixed with these.
        "max_turns": max_turns,
        "n_capped": sum(1 for r in results if r.get("capped")),
        "n_no_answer": sum(1 for r in results
                           if r.get("answer_source") in ("none", "error")),
        "n_final_text": sum(1 for r in results
                            if r.get("answer_source") == "final_text"),
        "n_recovered": sum(1 for r in results
                           if r.get("answer_source") == "submit_answer_recovered"),
        "no_answer_rate": (sum(1 for r in results
                               if r.get("answer_source") in ("none", "error"))
                           / max(total, 1)),
        "extraction": "sentinel-v3/2026-08-15",
        "total_time_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    if benchmark_name in vqa_scoring.VQA_BENCHMARKS:
        # Which rule produced `accuracy`, and what the other rule would have
        # said on the same rollouts. A VQA number that does not say how it was
        # scored is worthless; both numbers ship together so the rebuttal's
        # "was / is" table can be read straight off the artifact.
        block = {
            "rule": scoring_rule,
            "version": (vqa_scoring.CF_EM_VERSION if cf_vocab is not None
                        else "substring/pre-2026-07-29"),
            "closed_vocab_benchmark": benchmark_name in vqa_scoring.CLOSED_VOCAB_BENCHMARKS,
            "requested": vqa_scorer,
        }
        if cf_vocab is not None:
            sub_correct = sum(1 for r in results if r.get("substring_correct"))
            block.update({
                "vocab_labels": len(cf_vocab.labels),
                "vocab_built_from_n_items": len(cf_vocab.items),
                "cf_em": accuracy,
                "cf_em_correct": correct,
                "substring_accuracy": sub_correct / max(total, 1),
                "substring_correct": sub_correct,
                "no_commit_rate": (sum(1 for r in results
                                       if r.get("cf_kind") != "error"
                                       and not r.get("cf_pred"))
                                   / max(total, 1)),
            })
            rows = [{"submitted": r.get("submitted", ""), "gold": r.get("gold", ""),
                     "_i": vqa_scoring.task_row_index(r["task_id"], benchmark_name)}
                    for r in results]
            _, cf_stats = vqa_scoring.score_all(rows, cf_vocab)
            block["cf_bacc_guard"] = cf_stats["cf_bacc"]
        summary["vqa_scoring"] = block
        summary["metric"] = (
            f"{scoring_rule} (see scripts/vqa_scoring.py)")

    # Route the benchmark's primary metric explicitly. Consumers read
    # summary["accuracy"] uniformly across benchmark types even where
    # summary["metric"] says otherwise; primary_metric/primary_value give them
    # a single field that is right for every type. "accuracy" itself is left
    # untouched (for LFQA it is the rouge_l>=0.3 binary, as before).
    if is_ehr:
        summary["avg_action_score"] = action_score_sum / max(total, 1)
        summary["metric"] = "action_score (expected tool call coverage)"
        summary["primary_metric"] = "avg_action_score"
        summary["primary_value"] = summary["avg_action_score"]
    elif is_lfqa:
        summary["avg_rouge_l"] = rouge_l_sum / max(total, 1)
        # Answered-only denominators. The old total denominator made the
        # metric `100 x abstention + (1-abstention) x true hallucination`,
        # i.e. mostly a re-encoding of abstention with the sign inverted.
        # Abstention is not erased -- it lives in no_answer_rate, one key up.
        # An arm with zero answered rows reports None, not a number.
        summary["avg_hallucination"] = (hall_sum / hall_n) if hall_n else None
        summary["avg_comprehensiveness"] = (comp_sum / comp_n) if comp_n else None
        summary["hallucination_answered_n"] = hall_n
        summary["comprehensiveness_answered_n"] = comp_n
        summary["metric"] = "rouge_l + hallucination + comprehensiveness"
        summary["primary_metric"] = "avg_rouge_l"
        summary["primary_value"] = summary["avg_rouge_l"]
    else:
        summary["primary_metric"] = "accuracy"
        summary["primary_value"] = accuracy

    # Save final results
    out_path = output_dir / f"{benchmark_name}_multiturn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _mm = summary["multimodal_tool_forwarding"]
    fa_line = (
        f"  prompt_mode={prompt_mode}  react_rate={format_adherence['react_rate']:.3f} "
        f"({format_adherence['n_react']}/{format_adherence['n_turns']} turns)  "
        f"formats={format_adherence['formats']}\n"
        f"  multimodal_tools={_mm['arm']} "
        f"({_mm['multimodal_requests_with_tools']}/{_mm['multimodal_requests']} "
        f"image requests carried the catalog)\n"
    )
    if is_ehr:
        logger.info(
            f"\n{'='*60}\n"
            f"  {benchmark_name}: action_score={summary['avg_action_score']:.3f} "
            f"acc={accuracy:.3f} ({correct}/{total})\n"
            f"  avg_turns={summary['avg_turns']:.1f}\n"
            f"{fa_line}"
            f"  time={elapsed:.0f}s  saved={out_path}\n"
            f"{'='*60}"
        )
    elif is_lfqa:
        _h = summary["avg_hallucination"]
        _c = summary["avg_comprehensiveness"]
        logger.info(
            f"\n{'='*60}\n"
            f"  {benchmark_name}: rouge_l={summary['avg_rouge_l']:.3f} "
            f"hall={'n/a' if _h is None else f'{_h:.1f}%'}"
            f" comp={'n/a' if _c is None else f'{_c:.1f}%'}"
            f" (answered n={summary['hallucination_answered_n']})\n"
            f"  (correct@0.3={correct}/{total}, acc={accuracy:.3f})\n"
            f"  avg_turns={summary['avg_turns']:.1f}  avg_reward={summary['avg_reward']:.3f}\n"
            f"{fa_line}"
            f"  time={elapsed:.0f}s  saved={out_path}\n"
            f"{'='*60}"
        )
    else:
        logger.info(
            f"\n{'='*60}\n"
            f"  {benchmark_name}: accuracy={accuracy:.3f} ({correct}/{total})\n"
            f"  avg_turns={summary['avg_turns']:.1f}  avg_reward={summary['avg_reward']:.3f}\n"
            f"{fa_line}"
            f"  time={elapsed:.0f}s  saved={out_path}\n"
            f"{'='*60}"
        )

    return summary


def _check_answer(submitted: str, gold: str, options: dict) -> bool:
    """Check if submitted answer matches gold, handling letter/text mismatches.

    Cases:
    - Gold is letter "D", submitted is "D" → exact match
    - Gold is text "Cross-linking of DNA", submitted is "D" → check if options["D"] matches gold
    - Gold is text, submitted is text → case-insensitive match
    - Submitted is letter, gold is text → find which letter has gold text, compare
    """
    submitted = submitted.strip()
    gold = gold.strip()

    if not submitted:
        return False

    # Direct match (case-insensitive)
    if submitted.lower() == gold.lower():
        return True

    # Gold is a short letter (A-E)
    if len(gold) <= 2 and gold.upper() in "ABCDE":
        # Check if submitted starts with the gold letter
        if submitted.upper().startswith(gold.upper()):
            return True
        # Check if submitted text matches the option text for the gold letter
        gold_text = options.get(gold.upper(), "")
        if gold_text and submitted.lower() == gold_text.lower():
            return True
        return False

    # Gold is full text — find which letter it corresponds to
    gold_letter = None
    for letter, text in options.items():
        if text.strip().lower() == gold.lower():
            gold_letter = letter
            break

    if gold_letter:
        # Submitted is a letter
        first_char = submitted[0].upper() if submitted else ""
        if first_char == gold_letter.upper():
            return True
        # Submitted starts with "X." or "X)" pattern
        m = re.match(r'^([A-E])[.\):\s]', submitted.upper())
        if m and m.group(1) == gold_letter.upper():
            return True

    # Substring match for free-text answers (skip if gold is empty)
    if gold and gold.lower() in submitted.lower():
        return True

    return False


def _compute_rouge_l(submitted: str, gold: str) -> float:
    """Compute ROUGE-L F1 between submitted answer and gold reference."""
    if not submitted or not gold:
        return 0.0
    scores = _rouge_scorer.score(gold, submitted)
    return scores["rougeL"].fmeasure


def _nli_cosine(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using biobert-nli."""
    _ensure_nli_model()
    encoded = _nli_tokenizer(
        [text_a, text_b], padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(_nli_device)
    with torch.no_grad():
        output = _nli_model(**encoded)
    # Mean pooling
    embs = output[0]
    mask = encoded["attention_mask"].unsqueeze(-1).expand(embs.size()).float()
    pooled = (embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.cosine_similarity(pooled[0:1], pooled[1:2]).item()


def _compute_hallucination(submitted: str, must_have: list, nice_to_have: list):
    """Hallucination rate: % of statements with cosine < 0.5.

    None (not 100.0) when there is nothing to score -- no submission, or no
    reference statements. Scoring an EMPTY submission as 100% hallucination
    re-encoded abstention as hallucination with the sign inverted: across 113
    stored LFQA files, corr(no_answer_rate, avg_hallucination) = 0.82 and the
    metric ranked arms backwards (an arm that answered nothing reported 100.00).
    Unanswered belongs in no_answer_rate, which is reported alongside.
    """
    all_stmts = must_have + nice_to_have
    if not all_stmts or not submitted or not submitted.strip():
        return None
    hall = sum(1 for s in all_stmts if _nli_cosine(submitted, s) < 0.5)
    return hall / len(all_stmts) * 100


def _compute_comprehensiveness(submitted: str, must_have: list):
    """Comprehensiveness: % of must-have statements with cosine >= 0.5.

    None when there is nothing to score (same convention and reason as
    _compute_hallucination: it shares the total denominator and conflated
    abstention with low coverage).
    """
    if not must_have or not submitted or not submitted.strip():
        return None
    comp = sum(1 for s in must_have if _nli_cosine(submitted, s) >= 0.5)
    return comp / len(must_have) * 100


def _extract_answer_fallback(text: str) -> str:
    """Extract answer from raw text when no submit_answer was called."""
    # Try Qwen3.5 XML format
    m = re.search(r'<parameter=answer>\s*(.*?)\s*</parameter>', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try JSON format
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if m:
        return m.group(1).strip()

    # Try "The answer is X" pattern
    m = re.search(r'(?:the answer is|answer:)\s*([A-E])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Malformed tool-call XML that failed to parse reaches this function with
    # the XML in hand. Mining its first 100 characters used to record the tool
    # call ITSELF as the answer -- silently wrong on every affected row, at an
    # arm-DEPENDENT rate (5-56% by arm/benchmark), which turned arm-vs-arm
    # deltas into partial format-compliance contests. An episode that never
    # produced an answer must stay visibly unanswered: return the "" sentinel.
    if "<tool_call>" in text or "<function=" in text:
        return ""
    return text[:100].strip()


def _load_resume_partial(benchmark_name, output_dir, resume_from, skipped_tasks, vqa_scorer):
    """Load the rows a previous run of this benchmark already produced.

    Resuming is only safe when the rows being kept were produced from the same
    tasks under the same scoring rule; otherwise the merged artifact would report
    one accuracy over two different measurements. Every mismatch is fatal -- there
    is no partial-credit path here, because the failure mode is a plausible-looking
    number rather than a crash.
    """
    path = Path(output_dir) / f"{benchmark_name}_partial.json"
    if not path.exists():
        raise SystemExit(
            f"[fatal] --resume-from {resume_from} but no {path}. "
            f"Resume needs the rows it is continuing from; drop --resume-from to start over."
        )

    with open(path) as f:
        partial = json.load(f)

    prior = partial.get("results", [])
    if partial.get("benchmark") != benchmark_name:
        raise SystemExit(
            f"[fatal] {path} holds benchmark '{partial.get('benchmark')}', not '{benchmark_name}'"
        )
    if len(prior) != resume_from:
        raise SystemExit(
            f"[fatal] --resume-from {resume_from} but {path} holds {len(prior)} rows. "
            f"Pass --resume-from {len(prior)} to continue from where it stopped."
        )

    mismatched = [
        (i, row.get("task_id"), task.get("id"))
        for i, (row, task) in enumerate(zip(prior, skipped_tasks))
        if row.get("task_id") != task.get("id")
    ]
    if mismatched:
        i, got, want = mismatched[0]
        raise SystemExit(
            f"[fatal] {path} row {i} is task '{got}' but the benchmark's task {i} is '{want}' "
            f"({len(mismatched)} mismatched). The partial came from a different task order "
            f"or a different benchmark file; it cannot be merged."
        )

    # Pre-sentinel partials mined the trailing tool-call XML as the answer on
    # unanswered episodes; rows produced under that rule cannot share an
    # accuracy with sentinel-v2 rows -- the same one-number-two-measurements
    # failure the vqa_scoring guard below already refuses. VQA rows are exempt:
    # their capped episodes structurally end in a parsed submit_answer, so the
    # fallback never fired on them (0 artifacts across all 190 stored files).
    if prior and benchmark_name not in vqa_scoring.VQA_BENCHMARKS \
            and any("answer_source" not in row for row in prior):
        raise SystemExit(
            f"[fatal] {path} was written before the sentinel-v2 extraction fix "
            f"(rows carry no 'answer_source'). Mixing extraction rules inside one "
            f"accuracy is not resumable; restart the benchmark, or finish it under "
            f"the old code and rescore the artifact with "
            f"scripts/rebuttal/rescore_extraction.py."
        )

    # Old partials carry no vqa_scoring block, which is itself the tell that they
    # are substring-scored. Finishing one under cf_em would mix two rules in a
    # single accuracy; finish it under --vqa-scorer substring and rescore the
    # completed artifact instead (scripts/rebuttal/rescore_vqa.py).
    if benchmark_name in vqa_scoring.CLOSED_VOCAB_BENCHMARKS:
        prior_rule = (partial.get("vqa_scoring") or {}).get("rule", "substring")
        if prior_rule != vqa_scorer:
            raise SystemExit(
                f"[fatal] {path} was scored with '{prior_rule}' but this run scores with "
                f"'{vqa_scorer}'. Re-run with --vqa-scorer {prior_rule} to finish it, then "
                f"rescore the completed artifact with scripts/rebuttal/rescore_vqa.py."
            )

    logger.info(f"Resuming {benchmark_name} from {len(prior)} rows in {path}")
    return prior


def _save_partial(benchmark_name, results, correct, total, output_dir,
                  scoring_rule=None):
    """Save partial results for resumability."""
    partial = {
        "benchmark": benchmark_name,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        # The resume guard keys on rows' answer_source; the tag here makes the
        # partial's extraction rule readable without inspecting rows.
        "extraction": "sentinel-v3/2026-08-15",
        "results": results,
    }
    if scoring_rule is not None:
        # Partials get read straight into rebuttal tables, so they carry the
        # rule too. The pre-2026-07-29 partials have no such field, which is
        # itself the tell that they are substring-scored.
        sub_correct = sum(1 for r in results if r.get("substring_correct"))
        partial["vqa_scoring"] = {
            "rule": scoring_rule,
            "version": (vqa_scoring.CF_EM_VERSION if scoring_rule == "cf_em"
                        else "substring/pre-2026-07-29"),
            "substring_correct": sub_correct,
            "substring_accuracy": sub_correct / max(total, 1),
        }
    path = output_dir / f"{benchmark_name}_partial.json"
    with open(path, "w") as f:
        json.dump(partial, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn benchmark evaluation")
    parser.add_argument("--model_path", required=True, help="Path to merged HF checkpoint")
    parser.add_argument("--benchmarks", nargs="+", default=["medqa"],
                        choices=list(BENCHMARK_FILES.keys()),
                        help="Benchmarks to evaluate")
    parser.add_argument("--domain", default=None,
                        help="Override domain (default: auto from benchmark)")
    parser.add_argument("--output-dir", default="results/benchmarks_multiturn",
                        help="Output directory")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max turns per task (default 10: think/search cycles + submit)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per benchmark (0=all)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max new tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from sample index")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable think() tool (ablation)")
    parser.add_argument("--prompt-mode", default="default",
                        choices=["default", "strong_tool", "react"],
                        help="Prompting baseline (eval-only, zero training). "
                             "'default' = Base+AR, unchanged. "
                             "'strong_tool' = Base+AR with a materially stronger "
                             "tool-use contract. "
                             "'react' = explicit ReAct scaffolding "
                             "(Thought/Action/Action Input/Observation) over the "
                             "SAME tool set. Isolates prompting/scaffolding gains "
                             "from TT-OPD.")
    parser.add_argument("--vqa-scorer", default="cf_em",
                        choices=["cf_em", "substring"],
                        help="Scoring rule for the CLOSED-VOCABULARY visual-QA "
                             "benchmarks (vqa_rad, slake, pathvqa). "
                             "'cf_em' = closed-form exact match (default). "
                             "'substring' = the pre-2026-07-29 rule that scored "
                             "an image-blind constant paragraph at 56.5%% on "
                             "VQA-RAD and 45.4%% on SLAKE; kept reachable ONLY "
                             "so the published numbers stay reproducible. "
                             "Never affects text QA, long-form QA, EHR, or the "
                             "open-vocabulary VQA sets. "
                             "See scripts/vqa_scoring.py for the rule and its limits.")
    parser.add_argument("--dump-transcripts", action="store_true",
                        help="Append every task's full turn-by-turn raw output to "
                             "<benchmark>_transcripts.jsonl in the output dir. "
                             "Without it, episodes that end without an answer "
                             "cannot be re-scored post-hoc (only the extracted "
                             "'submitted' string is stored in results).")
    parser.add_argument("--backend", default="transformers",
                        choices=["transformers", "vllm", "sglang"],
                        help="Inference backend (default: transformers)")
    parser.add_argument("--server-url", default=None,
                        help="SGLang server URL (required for sglang backend)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once, reuse across benchmarks
    from bioagents.evaluation.agent_runner import AgentRunner, RunConfig

    # Use the first benchmark's domain for model loading
    first_domain = args.domain or BENCHMARK_DOMAIN[args.benchmarks[0]]

    config = RunConfig(
        model_name_or_path=args.model_path,
        backend=args.backend,
        domain=first_domain,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        log_dir=str(output_dir / "logs"),
        no_think=args.no_think,
        server_url=args.server_url,
        prompt_mode=args.prompt_mode,
    )

    logger.info(f"Prompting baseline: prompt_mode={args.prompt_mode}")
    logger.info(f"Loading model: {args.model_path}")
    runner = AgentRunner(config)
    runner.load_model()
    logger.info("Model loaded successfully")

    # One offset cannot be correct for several benchmarks at once, and the loop
    # below would apply it to every one of them.
    if args.resume_from > 0 and len(args.benchmarks) > 1:
        raise SystemExit(
            f"[fatal] --resume-from {args.resume_from} was given with {len(args.benchmarks)} "
            f"benchmarks ({', '.join(args.benchmarks)}); it would be applied to each of them. "
            f"Resume one benchmark per run."
        )

    all_summaries = {}

    for bench_name in args.benchmarks:
        domain = args.domain or BENCHMARK_DOMAIN[bench_name]

        # Update runner's domain config for this benchmark
        runner.config.domain = domain

        # Load benchmark data
        VQA_BENCHMARKS = {"vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"}
        EHR_BENCHMARKS = {"mimic_iii", "eicu"}
        if bench_name in VQA_BENCHMARKS:
            tasks = load_vqa_benchmark(bench_name)
        elif bench_name in EHR_BENCHMARKS:
            tasks = load_ehr_benchmark(bench_name)
        else:
            tasks = load_textqa_benchmark(bench_name)

        if not tasks:
            logger.warning(f"No tasks loaded for {bench_name}, skipping")
            continue

        # Apply offset (resume-from) first, then max_samples limit
        prior_results = None
        if args.resume_from > 0:
            prior_results = _load_resume_partial(
                bench_name, output_dir, args.resume_from,
                tasks[:args.resume_from], args.vqa_scorer,
            )
            tasks = tasks[args.resume_from:]
        if args.max_samples > 0:
            tasks = tasks[:args.max_samples]

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {bench_name}: {len(tasks)} samples (offset={args.resume_from}), domain={domain}, max_turns={args.max_turns}")
        logger.info(f"{'='*60}")

        summary = run_benchmark_multiturn(
            benchmark_name=bench_name,
            tasks=tasks,
            runner=runner,
            domain=domain,
            max_turns=args.max_turns,
            output_dir=output_dir,
            resume_from=0,  # Already applied above
            vqa_scorer=args.vqa_scorer,
            prior_results=prior_results,
            dump_transcripts=args.dump_transcripts,
        )
        all_summaries[bench_name] = {
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "total": summary["total"],
            "avg_turns": summary["avg_turns"],
            "time": summary["total_time_seconds"],
            "react_rate": summary["format_adherence"]["react_rate"],
        }

    # Print final comparison table
    logger.info(f"\n{'='*60}")
    logger.info("MULTI-TURN BENCHMARK RESULTS")
    logger.info(f"Model: {Path(args.model_path).name}")
    logger.info(f"Prompt mode: {args.prompt_mode}")
    logger.info(
        f"{'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>8} "
        f"{'Turns':>8} {'ReActRate':>10} {'Time':>10}"
    )
    logger.info("-" * 77)
    for name, s in all_summaries.items():
        logger.info(
            f"{name:<15} {s['accuracy']:>9.3f} {s['correct']:>9d} "
            f"{s['total']:>7d} {s['avg_turns']:>7.1f} {s['react_rate']:>9.3f} "
            f"{s['time']:>9.0f}s"
        )
    if args.prompt_mode == "react":
        logger.info(
            "NOTE: react_rate = fraction of assistant turns that actually parsed as "
            "ReAct (Thought/Action/Action Input). ReAct is the only tool-calling "
            "contract in this prompt — the chat template's native contract is "
            "suppressed and the full tool catalog is carried as prompt text — so "
            "react_rate now measures format adoption, not a prompt clash. A low "
            "score at high react_rate is evidence about ReAct; a low score at low "
            "react_rate means this backbone will not adopt the format, and the arm "
            "cannot be reported as a ReAct control."
        )
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
