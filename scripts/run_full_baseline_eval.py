#!/usr/bin/env python3
"""Full Baseline Evaluation on Complete Test Datasets.

Evaluates 3 models on all available benchmarks using vLLM for fast batch inference.

Text QA (7,634 test examples):
- MedQA: 1,273
- MedMCQA: 4,183
- MMLU x6: 1,089 (anatomy 135, clinical 265, professional 272, genetics 100, biology 144, college_med 173)

Usage:
    # Run all models in parallel
    python scripts/run_full_baseline_eval.py --parallel

    # Run single model
    python scripts/run_full_baseline_eval.py --model qwen3

    # Run specific benchmarks
    python scripts/run_full_baseline_eval.py --model qwen3 --benchmarks medqa medmcqa
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent

# Model configs — 3 VL models (fixed)
# All models support Vision+Text and are used in the Autonomous GYM
MODELS = {
    "lingshu": {
        "name": "Lingshu-7B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Lingshu-7B"),
        "gpus": "0,1",
        "type": "vlm",
    },
    "qwen2vl": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "path": str(PROJECT_ROOT / "checkpoints/models/Qwen2.5-VL-7B-Instruct"),
        "gpus": "2,3",
        "type": "vlm",
    },
    "step3vl": {
        "name": "Step3-VL-10B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Step3-VL-10B"),
        "gpus": "4,5",
        "type": "vlm",
    },
}

# Benchmark paths (relative to PROJECT_ROOT)
TEXT_BENCHMARKS = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
    "mmlu_professional": "evaluations/self-biorag/data/benchmark/mmlu_professional_medicine_test.jsonl",
    "mmlu_anatomy": "evaluations/self-biorag/data/benchmark/mmlu_anatomy_test.jsonl",
    "mmlu_genetics": "evaluations/self-biorag/data/benchmark/mmlu_medical_genetics_test.jsonl",
    "mmlu_biology": "evaluations/self-biorag/data/benchmark/mmlu_college_biology_test.jsonl",
    "mmlu_college_med": "evaluations/self-biorag/data/benchmark/mmlu_college_medicine_test.jsonl",
}

ALL_TEXT_BENCHMARKS = list(TEXT_BENCHMARKS.keys())


def load_benchmark_data(benchmark: str) -> list[dict]:
    """Load all test examples for a benchmark."""
    rel_path = TEXT_BENCHMARKS[benchmark]
    data_path = PROJECT_ROOT / rel_path
    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        return []

    data = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_item(item: dict) -> tuple:
    """Parse a Self-BioRAG format item into (question, options, correct_answer)."""
    instances = item.get("instances", {})
    raw_input = instances.get("input", "")
    raw_output = instances.get("output", "")

    if not raw_input:
        question = item.get("question", "")
        options = item.get("options", {})
        if isinstance(options, list):
            options = {chr(65 + i): opt for i, opt in enumerate(options)}
        correct = item.get("answer_idx", item.get("answer", ""))
        return question, options, correct

    # Parse "QUESTION: ... Option A: ... Option B: ..."
    if "QUESTION:" in raw_input:
        rest = raw_input.split("QUESTION:", 1)[1].strip()
    else:
        rest = raw_input.strip()

    option_pattern = r'Option\s+([A-E]):\s*(.*?)(?=Option\s+[A-E]:|$)'
    option_matches = re.findall(option_pattern, rest, re.DOTALL)

    question = ""
    options = {}

    if option_matches:
        first_opt_pos = rest.find("Option A:")
        if first_opt_pos == -1:
            first_opt_pos = rest.find("Option B:")
        if first_opt_pos > 0:
            question = rest[:first_opt_pos].strip().rstrip('\n')

        for letter, text in option_matches:
            options[letter.strip()] = text.strip()
    else:
        question = rest

    # Determine correct answer letter from output text
    correct_letter = ""
    if raw_output:
        raw_output_clean = raw_output.strip()
        for letter, text in options.items():
            if text.strip().lower() == raw_output_clean.lower():
                correct_letter = letter
                break
            if raw_output_clean.lower() in text.lower() or text.lower() in raw_output_clean.lower():
                correct_letter = letter
                break
        if not correct_letter and raw_output_clean and raw_output_clean[0].upper() in options:
            correct_letter = raw_output_clean[0].upper()
        if not correct_letter:
            correct_letter = "B"  # fallback

    return question, options, correct_letter


def build_prompt(question: str, options: dict) -> str:
    """Build MC prompt."""
    prompt = f"Question: {question}\n\nOptions:\n"
    for letter, text in sorted(options.items()):
        prompt += f"  {letter}) {text}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt


def extract_answer_letter(response: str, valid_letters: list[str] = None) -> str:
    """Extract answer letter from model response."""
    if valid_letters is None:
        valid_letters = ["A", "B", "C", "D"]

    response = response.strip()

    # Direct letter
    if response and response[0].upper() in valid_letters:
        return response[0].upper()

    patterns = [
        r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-D])\)?",
        r"\b([A-D])\b\s*(?:\)|\.|\:)",
        r"^\s*\(?([A-D])\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    for letter in valid_letters:
        if letter in response.upper():
            return letter

    return "A"


def evaluate_model_vllm(model_key: str, benchmarks: list[str], output_dir: Path):
    """Evaluate a model using vLLM for fast batch inference."""
    from vllm import LLM, SamplingParams

    model_info = MODELS[model_key]
    model_name = model_info["name"]
    model_path = model_info["path"]
    gpu_ids = model_info["gpus"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    print(f"\n{'='*70}")
    print(f"  Evaluating {model_name} on {len(benchmarks)} benchmarks")
    print(f"  GPUs: {gpu_ids}")
    print(f"{'='*70}\n")

    # Load vLLM model
    print(f"[{model_name}] Loading model with vLLM...")
    t0 = time.time()

    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=len(gpu_ids.split(",")),
            max_model_len=4096,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        )
        print(f"[{model_name}] Model loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[{model_name}] vLLM load failed: {e}")
        print(f"[{model_name}] Falling back to transformers backend...")
        return evaluate_model_transformers(model_key, benchmarks, output_dir)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["\n", "Question:", "Options:"],
    )

    all_results = {}
    total_correct = 0
    total_count = 0

    for benchmark in benchmarks:
        print(f"\n[{model_name}] Evaluating {benchmark}...")
        data = load_benchmark_data(benchmark)
        if not data:
            all_results[benchmark] = {"error": "No data"}
            continue

        # Prepare prompts
        prompts = []
        answers = []
        valid_indices = []

        for i, item in enumerate(data):
            question, options, correct = parse_item(item)
            if question and options:
                prompt = build_prompt(question, options)
                # Apply chat template
                messages = [
                    {"role": "system", "content": "You are a medical expert taking a medical exam. Answer each question with only the correct option letter."},
                    {"role": "user", "content": prompt},
                ]
                try:
                    from transformers import AutoTokenizer
                    if not hasattr(evaluate_model_vllm, '_tokenizer'):
                        evaluate_model_vllm._tokenizer = AutoTokenizer.from_pretrained(
                            model_path, trust_remote_code=True
                        )
                    tokenizer = evaluate_model_vllm._tokenizer
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(formatted)
                except Exception:
                    prompts.append(f"<|im_start|>system\nYou are a medical expert taking a medical exam. Answer each question with only the correct option letter.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
                answers.append(correct)
                valid_indices.append(i)

        print(f"  [{model_name}] {benchmark}: {len(prompts)} prompts prepared, generating...")
        t1 = time.time()

        # Batch generate
        outputs = llm.generate(prompts, sampling_params)

        elapsed = time.time() - t1
        print(f"  [{model_name}] {benchmark}: generated in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} q/s)")

        # Score
        correct_count = 0
        per_question = []

        for idx, (output, gold) in enumerate(zip(outputs, answers)):
            response = output.outputs[0].text.strip()
            pred = extract_answer_letter(response)
            is_correct = pred == gold

            if is_correct:
                correct_count += 1

            per_question.append({
                "idx": valid_indices[idx],
                "predicted": pred,
                "correct": gold,
                "is_correct": is_correct,
                "raw_response": response[:200],
            })

        accuracy = correct_count / max(len(prompts), 1)
        all_results[benchmark] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(prompts),
            "per_question": per_question,
        }

        total_correct += correct_count
        total_count += len(prompts)

        print(f"  [{model_name}] {benchmark}: {accuracy:.1%} ({correct_count}/{len(prompts)})")

    # Overall
    overall_acc = total_correct / max(total_count, 1)
    all_results["_overall"] = {
        "accuracy": overall_acc,
        "correct": total_correct,
        "total": total_count,
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"full_baseline_{model_key}_{timestamp}.json"

    report = {
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_question"}
            for k, v in all_results.items()
        },
        "detailed_results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: {model_name}")
    print(f"{'='*70}")
    for bench in benchmarks:
        r = all_results.get(bench, {})
        if "error" in r:
            print(f"  {bench:25s}: ERROR")
        else:
            print(f"  {bench:25s}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
    print(f"  {'─'*60}")
    print(f"  {'OVERALL':25s}: {overall_acc:.1%} ({total_correct}/{total_count})")
    print(f"{'='*70}")
    print(f"  Saved to: {output_path}")

    return all_results


def evaluate_model_transformers(model_key: str, benchmarks: list[str], output_dir: Path):
    """Fallback: evaluate using transformers (slower but handles VL models)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_info = MODELS[model_key]
    model_name = model_info["name"]
    model_path = model_info["path"]
    gpu_ids = model_info["gpus"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    print(f"\n[{model_name}] Loading with transformers backend...")
    t0 = time.time()

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
    )

    if is_qwen_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{model_name}] Loaded in {time.time()-t0:.1f}s")

    all_results = {}
    total_correct = 0
    total_count = 0

    for benchmark in benchmarks:
        print(f"\n[{model_name}] Evaluating {benchmark}...")
        data = load_benchmark_data(benchmark)
        if not data:
            all_results[benchmark] = {"error": "No data"}
            continue

        correct_count = 0
        per_question = []
        t1 = time.time()

        for i, item in enumerate(data):
            question, options, correct_answer = parse_item(item)
            if not question or not options:
                continue

            prompt = build_prompt(question, options)
            messages = [
                {"role": "system", "content": "You are a medical expert taking a medical exam. Answer each question with only the correct option letter."},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            pred = extract_answer_letter(response)
            is_correct = pred == correct_answer

            if is_correct:
                correct_count += 1

            per_question.append({
                "idx": i,
                "predicted": pred,
                "correct": correct_answer,
                "is_correct": is_correct,
            })

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t1
                print(f"  [{model_name}] {benchmark}: {i+1}/{len(data)} ({correct_count/(i+1):.1%}), {elapsed:.0f}s")

        total = len(per_question)
        accuracy = correct_count / max(total, 1)
        all_results[benchmark] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total,
            "per_question": per_question,
        }

        total_correct += correct_count
        total_count += total

        elapsed = time.time() - t1
        print(f"  [{model_name}] {benchmark}: {accuracy:.1%} ({correct_count}/{total}) in {elapsed:.0f}s")

    overall_acc = total_correct / max(total_count, 1)
    all_results["_overall"] = {
        "accuracy": overall_acc,
        "correct": total_correct,
        "total": total_count,
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"full_baseline_{model_key}_{timestamp}.json"

    report = {
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_question"}
            for k, v in all_results.items()
        },
        "detailed_results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  RESULTS: {model_name}")
    print(f"{'='*70}")
    for bench in benchmarks:
        r = all_results.get(bench, {})
        if "error" in r:
            print(f"  {bench:25s}: ERROR")
        else:
            print(f"  {bench:25s}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
    print(f"  {'─'*60}")
    print(f"  {'OVERALL':25s}: {overall_acc:.1%} ({total_correct}/{total_count})")
    print(f"{'='*70}")
    print(f"  Saved to: {output_path}")

    return all_results


def run_single_model(args):
    """Run evaluation for a single model (for multiprocessing)."""
    model_key, benchmarks, output_dir = args
    model_info = MODELS[model_key]

    # Set environment for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = model_info["gpus"]

    if model_info["type"] == "vlm":
        return evaluate_model_transformers(model_key, benchmarks, output_dir)
    else:
        return evaluate_model_vllm(model_key, benchmarks, output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full Baseline Evaluation on Complete Datasets")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--benchmarks", nargs="+", default=ALL_TEXT_BENCHMARKS,
                        choices=ALL_TEXT_BENCHMARKS)
    parser.add_argument("--parallel", action="store_true", help="Run models in parallel")
    parser.add_argument("--output-dir", default="logs/full_baseline")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Healthcare AI GYM — Full Baseline Evaluation")
    print(f"  Benchmarks: {', '.join(args.benchmarks)}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}")

    # Count total examples
    total_examples = 0
    for bench in args.benchmarks:
        data = load_benchmark_data(bench)
        total_examples += len(data)
        print(f"  {bench}: {len(data)} test examples")
    print(f"  Total: {total_examples} test examples")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    if args.parallel and len(models_to_run) > 1:
        print(f"\n  Running {len(models_to_run)} models in parallel...")
        # Use subprocess for true parallelism with separate CUDA contexts
        processes = []
        for model_key in models_to_run:
            model_info = MODELS[model_key]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = model_info["gpus"]
            cmd = [
                sys.executable, __file__,
                "--model", model_key,
                "--benchmarks", *args.benchmarks,
                "--output-dir", args.output_dir,
            ]
            log_path = output_dir / f"{model_key}_eval.log"
            print(f"  Starting {model_info['name']} on GPUs {model_info['gpus']} -> {log_path}")
            with open(log_path, "w") as log_f:
                p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                                     cwd=str(PROJECT_ROOT))
            processes.append((model_key, p, log_path))

        # Wait for all
        for model_key, p, log_path in processes:
            print(f"  Waiting for {MODELS[model_key]['name']}...")
            p.wait()
            print(f"  {MODELS[model_key]['name']}: exit code {p.returncode} (log: {log_path})")

        print(f"\n  All evaluations complete!")

    else:
        for model_key in models_to_run:
            model_info = MODELS[model_key]
            os.environ["CUDA_VISIBLE_DEVICES"] = model_info["gpus"]

            if model_info["type"] == "vlm":
                evaluate_model_transformers(model_key, args.benchmarks, output_dir)
            else:
                evaluate_model_vllm(model_key, args.benchmarks, output_dir)

    # Aggregate and print comparison
    print(f"\n\n{'#'*70}")
    print(f"  AGGREGATED RESULTS")
    print(f"{'#'*70}")

    result_files = sorted(output_dir.glob("full_baseline_*.json"))
    results_by_model = {}
    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            model_name = data["model_name"]
            results_by_model[model_name] = data["benchmarks"]
        except Exception:
            pass

    if results_by_model:
        # Print comparison table
        benchmarks_list = args.benchmarks
        header = f"{'Benchmark':25s}"
        for model_name in results_by_model:
            header += f" | {model_name:>20s}"
        print(header)
        print("-" * len(header))

        for bench in benchmarks_list:
            row = f"{bench:25s}"
            for model_name, benchmarks_data in results_by_model.items():
                r = benchmarks_data.get(bench, {})
                if "accuracy" in r:
                    row += f" | {r['accuracy']:>19.1%}"
                elif "error" in r:
                    row += f" | {'ERROR':>20s}"
                else:
                    row += f" | {'N/A':>20s}"
            print(row)

        print("-" * len(header))
        row = f"{'OVERALL':25s}"
        for model_name, benchmarks_data in results_by_model.items():
            r = benchmarks_data.get("_overall", {})
            if "accuracy" in r:
                row += f" | {r['accuracy']:>19.1%}"
            else:
                row += f" | {'N/A':>20s}"
        print(row)

    # Save aggregated comparison
    if results_by_model:
        comparison_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_path, "w") as f:
            json.dump(results_by_model, f, indent=2, ensure_ascii=False)
        print(f"\n  Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
