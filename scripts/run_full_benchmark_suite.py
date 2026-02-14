#!/usr/bin/env python3
"""Complete Benchmark Suite for ALL evaluation categories.

Categories:
1. Text MC QA (6,545 examples) - MedQA, MedMCQA, MMLU x6
2. Vision QA (6 datasets) - VQA-RAD, SLAKE, PathVQA, PMC-VQA, VQA-Med-2021, Quilt-VQA
3. MedLFQA Long-form QA (~4,948 examples) - KQA Golden, LiveQA, MedicationQA, HealthSearchQA, KQA Silver
4. Agent Tasks - Full test splits across 8 domains
5. EHR Benchmarks (full dataset) - MIMIC-III (all qualifying ICU patients) + eICU (all qualifying ICU patients)

Usage:
    python scripts/run_full_benchmark_suite.py --category vqa --model qwen2vl --gpus 6,7
    python scripts/run_full_benchmark_suite.py --category medlfqa --model qwen3 --gpus 6,7
    python scripts/run_full_benchmark_suite.py --category ehr --model qwen3 --gpus 6,7
"""

import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = {
    "qwen3": {
        "name": "Qwen3-8B-Base",
        "path": "/data/project/private/minstar/models/Qwen3-8B-Base",
        "type": "causal",
        "supports_vision": False,
    },
    "qwen2vl": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "path": str(PROJECT_ROOT / "checkpoints/models/Qwen2.5-VL-7B-Instruct"),
        "type": "vlm",
        "supports_vision": True,
    },
    "lingshu": {
        "name": "Lingshu-7B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Lingshu-7B"),
        "type": "vlm",
        "supports_vision": True,
    },
}

MEDLFQA_DATASETS = {
    "kqa_golden": {
        "path": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
        "name": "KQA Golden",
    },
    "live_qa": {
        "path": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
        "name": "LiveQA",
    },
    "medication_qa": {
        "path": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
        "name": "MedicationQA",
    },
    "healthsearch_qa": {
        "path": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
        "name": "HealthSearchQA",
    },
    "kqa_silver": {
        "path": "evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl",
        "name": "KQA Silver",
    },
}


def load_medlfqa_data(dataset_key, max_samples=0):
    info = MEDLFQA_DATASETS[dataset_key]
    data_path = PROJECT_ROOT / info["path"]
    if not data_path.exists():
        print(f"[WARN] Not found: {data_path}", flush=True)
        return []
    data = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if max_samples > 0:
        data = data[:max_samples]
    return data


def compute_rouge_l(prediction, reference):
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def compute_token_f1(prediction, reference):
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_must_have_coverage(prediction, must_have):
    if not must_have:
        return 1.0
    pred_lower = prediction.lower()
    covered = 0
    for item in must_have:
        words = [w for w in item.lower().split() if len(w) > 3]
        if words:
            match_count = sum(1 for w in words if w in pred_lower)
            if match_count / len(words) >= 0.5:
                covered += 1
        elif item.lower() in pred_lower:
            covered += 1
    return covered / len(must_have)


def compute_nice_to_have_coverage(prediction, nice_to_have):
    if not nice_to_have:
        return 1.0
    pred_lower = prediction.lower()
    covered = 0
    for item in nice_to_have:
        words = [w for w in item.lower().split() if len(w) > 3]
        if words:
            match_count = sum(1 for w in words if w in pred_lower)
            if match_count / len(words) >= 0.5:
                covered += 1
        elif item.lower() in pred_lower:
            covered += 1
    return covered / len(nice_to_have)


def evaluate_medlfqa(model_key, output_dir, max_samples=0):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_info = MODELS[model_key]
    model_name = model_info["name"]
    model_path = model_info["path"]

    print(f"\n{'='*70}", flush=True)
    print(f"  MedLFQA Evaluation: {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl") or ("qwen2" in model_type.lower() and "vl" in model_type.lower())

    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", attn_implementation="sdpa")

    if is_qwen_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[{model_name}] Model loaded", flush=True)

    all_results = {}

    for dataset_key, dataset_info in MEDLFQA_DATASETS.items():
        print(f"\n[{model_name}] Evaluating {dataset_info['name']}...", flush=True)
        data = load_medlfqa_data(dataset_key, max_samples=max_samples)
        if not data:
            all_results[dataset_key] = {"error": "No data"}
            continue

        print(f"  Loaded {len(data)} examples", flush=True)
        metrics_sum = {"rouge_l": 0, "token_f1": 0, "must_have": 0, "nice_to_have": 0}
        per_question = []
        t0 = time.time()

        # --- Batched inference for speed (4x-8x faster than sequential) ---
        BATCH_SIZE = 8
        valid_items = [(i, item) for i, item in enumerate(data) if item.get("Question", "")]

        for batch_start in range(0, len(valid_items), BATCH_SIZE):
            batch = valid_items[batch_start:batch_start + BATCH_SIZE]
            batch_messages = []
            batch_refs = []

            for idx, item in batch:
                question = item.get("Question", "")
                reference = item.get("Free_form_answer", "")
                must_have = item.get("Must_have", [])
                nice_to_have = item.get("Nice_to_have", [])
                batch_refs.append((idx, reference, must_have, nice_to_have))

                messages = [
                    {"role": "system", "content": "You are a medical expert. Provide detailed, accurate, evidence-based answers."},
                    {"role": "user", "content": f"Question: {question}\n\nProvide a comprehensive answer."},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_messages.append(text)

            # Batch tokenize with padding
            inputs = tokenizer(
                batch_messages, return_tensors="pt", truncation=True,
                max_length=4096, padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode each response in the batch
            for j, (idx, reference, must_have, nice_to_have) in enumerate(batch_refs):
                input_len = inputs["input_ids"].shape[-1]
                generated = outputs[j][input_len:]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()

                rouge_l = compute_rouge_l(response, reference)
                token_f1 = compute_token_f1(response, reference)
                mh_cov = compute_must_have_coverage(response, must_have)
                nth_cov = compute_nice_to_have_coverage(response, nice_to_have)

                metrics_sum["rouge_l"] += rouge_l
                metrics_sum["token_f1"] += token_f1
                metrics_sum["must_have"] += mh_cov
                metrics_sum["nice_to_have"] += nth_cov
                per_question.append({"idx": idx, "rouge_l": rouge_l, "token_f1": token_f1,
                                     "must_have_coverage": mh_cov, "nice_to_have_coverage": nth_cov})

            done = batch_start + len(batch)
            if done % 50 < BATCH_SIZE or done == len(valid_items):
                elapsed = time.time() - t0
                avg_rl = metrics_sum["rouge_l"] / max(len(per_question), 1)
                print(f"  [{model_name}] {dataset_info['name']}: {done}/{len(valid_items)} "
                      f"ROUGE-L={avg_rl:.3f} {elapsed:.0f}s", flush=True)

        n = len(per_question)
        if n == 0:
            all_results[dataset_key] = {"error": "No valid questions"}
            continue

        avg_metrics = {k: v / n for k, v in metrics_sum.items()}
        all_results[dataset_key] = {"name": dataset_info["name"], "total": n, **avg_metrics}
        elapsed = time.time() - t0
        print(f"  [{model_name}] {dataset_info['name']}: ROUGE-L={avg_metrics['rouge_l']:.3f} "
              f"Token-F1={avg_metrics['token_f1']:.3f} Must-Have={avg_metrics['must_have']:.3f} "
              f"({n} examples, {elapsed:.0f}s)", flush=True)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"medlfqa_{model_key}_{ts}.json"
    report = {"model_name": model_name, "model_path": model_path, "timestamp": datetime.now().isoformat(),
              "category": "medlfqa", "benchmarks": {k: {kk: vv for kk, vv in v.items() if kk != "per_question"} if isinstance(v, dict) else v for k, v in all_results.items()}}
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}", flush=True)
    print(f"  MedLFQA RESULTS: {model_name}", flush=True)
    print(f"  {'Dataset':<20} {'ROUGE-L':>10} {'Token-F1':>10} {'Must-Have':>10} {'N':>6}", flush=True)
    print(f"  {'-'*56}", flush=True)
    for key, r in all_results.items():
        if "error" in r:
            continue
        print(f"  {r['name']:<20} {r['rouge_l']:>10.3f} {r['token_f1']:>10.3f} {r['must_have']:>10.3f} {r['total']:>6}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Saved: {out_path}", flush=True)

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return all_results


def evaluate_vqa(model_key, output_dir, max_samples=0):
    model_info = MODELS[model_key]
    if not model_info["supports_vision"]:
        print(f"[SKIP] {model_info['name']} does not support vision.", flush=True)
        return {}

    sys.path.insert(0, str(PROJECT_ROOT))
    from bioagents.evaluation.vqa_benchmark_eval import VQABenchmarkConfig, VQABenchmarkEvaluator

    vqa_config = VQABenchmarkConfig(
        model_name_or_path=model_info["path"],
        model_name=model_info["name"],
        benchmarks=["vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"],
        max_samples=max_samples or 0,
        output_dir=str(output_dir / "vqa"),
        use_images=True,
    )
    evaluator = VQABenchmarkEvaluator(vqa_config)
    results = evaluator.evaluate_all()
    del evaluator
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return results


def evaluate_ehr(model_key, output_dir, max_samples=0):
    """Evaluate on EHR benchmarks: MIMIC-III (50 patients) + eICU (50 patients).

    Tests the model's ability to navigate real-world Electronic Health Records
    using tool-use in the BIOAgents EHR management environment.

    Metrics: action_score (tool usage accuracy), task completion rate, avg turns.
    """
    model_info = MODELS[model_key]
    model_name = model_info["name"]
    model_path = model_info["path"]

    print(f"\n{'='*70}", flush=True)
    print(f"  EHR Benchmark Evaluation: {model_name}", flush=True)
    print(f"  Datasets: MIMIC-III Clinical Database v1.4, eICU CRD v2.0", flush=True)
    print(f"{'='*70}", flush=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    from bioagents.evaluation.ehr_benchmark_eval import (
        EHRBenchmarkConfig,
        EHRBenchmarkEvaluator,
    )

    ehr_output = output_dir / "ehr"
    ehr_config = EHRBenchmarkConfig(
        model_name_or_path=model_path,
        model_name=model_name,
        benchmarks=["mimic_iii", "eicu"],
        max_samples=max_samples or 0,
        max_turns=15,
        output_dir=str(ehr_output),
    )
    evaluator = EHRBenchmarkEvaluator(ehr_config)
    results = evaluator.evaluate_all()

    del evaluator
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return results


def evaluate_textqa(model_key, output_dir, max_samples=0):
    """Evaluate on TextQA benchmarks: MedQA, MedMCQA, MMLU (6 subcategories)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_info = MODELS[model_key]
    model_name = model_info["name"]
    model_path = model_info["path"]

    print(f"\n{'='*70}", flush=True)
    print(f"  TextQA Evaluation: {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load model
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl") or ("qwen2" in model_type.lower() and "vl" in model_type.lower())

    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True,
                       device_map="auto", attn_implementation="sdpa")

    if is_qwen_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TextQA benchmark files
    TEXTQA_BENCHMARKS = {
        "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
        "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
        "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
    }

    all_results = {}
    for bm_key, bm_path in TEXTQA_BENCHMARKS.items():
        full_path = PROJECT_ROOT / bm_path
        if not full_path.exists():
            print(f"  [SKIP] {bm_key}: file not found", flush=True)
            continue

        # Load data
        data = []
        with open(full_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    data.append(d)
        if max_samples > 0:
            data = data[:max_samples]

        print(f"\n  [{model_name}] {bm_key}: {len(data)} examples", flush=True)
        correct = 0
        total = 0
        t0 = time.time()

        BATCH_SIZE = 8
        for batch_start in range(0, len(data), BATCH_SIZE):
            batch = data[batch_start:batch_start + BATCH_SIZE]
            batch_prompts = []
            batch_answers = []

            for item in batch:
                instances = item.get("instances", {})
                question = instances.get("input", "") if isinstance(instances, dict) else ""
                answer = instances.get("output", "") if isinstance(instances, dict) else ""
                if not question:
                    continue

                messages = [
                    {"role": "system", "content": "Answer the medical question by selecting the best option. Reply with ONLY the letter (A, B, C, or D)."},
                    {"role": "user", "content": question},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(text)
                batch_answers.append(answer.strip())

            if not batch_prompts:
                continue

            inputs = tokenizer(batch_prompts, return_tensors="pt", truncation=True,
                             max_length=4096, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)

            for j in range(len(batch_prompts)):
                input_len = inputs["input_ids"].shape[-1]
                generated = outputs[j][input_len:]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()

                # Extract answer letter
                pred = ""
                for ch in response:
                    if ch in "ABCD":
                        pred = ch
                        break

                ref = ""
                for ch in batch_answers[j]:
                    if ch in "ABCD":
                        ref = ch
                        break

                total += 1
                if pred == ref:
                    correct += 1

            done = min(batch_start + BATCH_SIZE, len(data))
            if done % 100 < BATCH_SIZE or done == len(data):
                elapsed = time.time() - t0
                acc = correct / max(total, 1)
                print(f"    {bm_key}: {done}/{len(data)} acc={acc:.3f} {elapsed:.0f}s", flush=True)

        acc = correct / max(total, 1)
        all_results[bm_key] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [{model_name}] {bm_key}: accuracy={acc:.4f} ({correct}/{total})", flush=True)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"textqa_{model_key}_{ts}.json"
    report = {"model_name": model_name, "model_path": model_path,
              "timestamp": datetime.now().isoformat(),
              "category": "textqa", "benchmarks": all_results}
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  TextQA RESULTS: {model_name}", flush=True)
    print(f"  {'Benchmark':<20} {'Accuracy':>10} {'Correct':>8} {'Total':>8}", flush=True)
    print(f"  {'-'*46}", flush=True)
    for key, r in all_results.items():
        print(f"  {key:<20} {r['accuracy']:>10.4f} {r['correct']:>8} {r['total']:>8}", flush=True)

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Benchmark Suite — VQA(6) + TextQA(3) + LongQA(5) + EHR(2)")
    parser.add_argument("--category", choices=["vqa", "medlfqa", "textqa", "ehr", "all"], default="all")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Model key (predefined)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Direct path to model checkpoint (overrides --model)")
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="results/benchmarks")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Handle model_path override
    if args.model_path:
        model_key = "custom"
        model_name = Path(args.model_path).name
        MODELS["custom"] = {
            "name": model_name,
            "path": args.model_path,
            "type": "causal",
            "supports_vision": False,
        }
        args.model = "custom"
    elif not args.model:
        parser.error("Either --model or --model_path is required")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODELS[args.model]
    print(f"\n{'#'*70}", flush=True)
    print(f"  Healthcare AI GYM — Full Benchmark Suite", flush=True)
    print(f"  Model: {model_info['name']} | Category: {args.category} | GPUs: {args.gpus}", flush=True)
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    if args.category in ("textqa", "all"):
        evaluate_textqa(args.model, output_dir, args.max_samples)

    if args.category in ("vqa", "all"):
        evaluate_vqa(args.model, output_dir, args.max_samples)

    if args.category in ("medlfqa", "all"):
        evaluate_medlfqa(args.model, output_dir, args.max_samples)

    if args.category in ("ehr", "all"):
        evaluate_ehr(args.model, output_dir, args.max_samples)

    print(f"\n{'='*70}", flush=True)
    print(f"  ALL EVALUATIONS COMPLETE — Results in {output_dir}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
