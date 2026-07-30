"""Convert BIOAgents tasks.json to veRL parquet format."""
import json
import os
from pathlib import Path

import datasets
import pandas as pd


def build_system_prompt(task):
    """Build system prompt based on task type."""
    domain = task.get("_source_domain", "text_qa")
    has_options = bool(task.get("options"))

    if has_options:
        return (
            "You are a medical AI assistant. Answer the following medical question. "
            "Think step by step, analyze the evidence, and provide your final answer. "
            "Your response MUST end with 'Answer: X' where X is the letter of your chosen option (A, B, C, D, or E)."
        )
    else:
        return (
            "You are a medical AI assistant. Answer the following medical question thoroughly. "
            "Think step by step and provide a detailed, evidence-based response."
        )


def build_user_content(task):
    """Build user message from task."""
    parts = []
    if task.get("raw_question"):
        parts.append(task["raw_question"])
    elif task.get("ticket"):
        parts.append(task["ticket"])

    if task.get("options"):
        parts.append("\nOptions:")
        for key, val in sorted(task["options"].items()):
            parts.append(f"  {key}: {val}")

    return "\n".join(parts)


def convert_tasks(tasks_path, split_path, output_dir):
    with open(tasks_path) as f:
        all_tasks = json.load(f)

    with open(split_path) as f:
        splits = json.load(f)

    task_by_id = {t["id"]: t for t in all_tasks}

    os.makedirs(output_dir, exist_ok=True)

    for split_name, task_ids in splits.items():
        records = []
        for idx, tid in enumerate(task_ids):
            task = task_by_id.get(tid)
            if task is None:
                continue

            system_prompt = build_system_prompt(task)
            user_content = build_user_content(task)

            correct = task.get("correct_answer", "")
            raw_answer = task.get("raw_answer", "")
            domain = task.get("_source_domain", "unknown")
            has_options = bool(task.get("options"))

            record = {
                "data_source": "bioagents_medical",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "ability": "medical",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": correct if has_options else raw_answer,
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "task_id": tid,
                    "domain": domain,
                    "correct_answer": correct,
                    "raw_answer": raw_answer,
                    "has_options": has_options,
                    "options": task.get("options", {}),
                },
            }
            records.append(record)

        ds = datasets.Dataset.from_list(records)
        out_path = os.path.join(output_dir, f"{split_name}.parquet")
        ds.to_parquet(out_path)
        print(f"  {split_name}: {len(ds)} tasks → {out_path}")


if __name__ == "__main__":
    import argparse

    base = os.environ.get("HCGYM_ROOT", str(Path(__file__).resolve().parents[2]))
    out_root = os.environ.get("HCGYM_RUN_ROOT", base)

    # The default pool is the DECONTAMINATED one. This block used to be
    # argument-less and hardcoded to full_4modality_combined, the pre-clean pool
    # (4543/916) -- so the parquet every training script actually consumes,
    # full_4modality_clean (3390/850), could not be produced by any committed code.
    # A fresh clone would either fail the training preflight or, worse, be repointed
    # at the one pool this script did reproduce and silently train on the
    # contaminated split, discarding what verify_clean_split.py certifies.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="full_4modality_clean",
                    help="pool directory under data/domains (default: the clean split)")
    ap.add_argument("--tasks-path", default=None)
    ap.add_argument("--split-path", default=None)
    ap.add_argument("--output-dir", default=None,
                    help="default: <run root>/data/verl_parquet/<pool>")
    args = ap.parse_args()

    pool_dir = f"{base}/data/domains/{args.pool}"
    convert_tasks(
        tasks_path=args.tasks_path or f"{pool_dir}/tasks.json",
        split_path=args.split_path or f"{pool_dir}/split_tasks.json",
        output_dir=args.output_dir or f"{out_root}/data/verl_parquet/{args.pool}",
    )
    print("Done!")
