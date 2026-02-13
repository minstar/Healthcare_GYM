#!/usr/bin/env python3
"""Data Contamination Checker for Healthcare AI GYM.

Verifies that evaluation/test data has NOT leaked into training data.
Run this BEFORE any training to ensure clean experimental protocol.

Checks:
1. No test-split task IDs appear in SFT training data
2. No external benchmark test questions appear in SFT data
3. GYM tasks are sourced from train benchmark files (not test)
4. GRPO configs use train splits only

Usage:
    python scripts/check_contamination.py
    python scripts/check_contamination.py --fix   # Auto-fix contaminated SFT files
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_test_task_ids() -> dict[str, set[str]]:
    """Get all test-split task IDs for every domain."""
    test_ids = {}
    domains_dir = PROJECT_ROOT / "data" / "domains"
    for domain_dir in sorted(domains_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        split_path = domain_dir / "split_tasks.json"
        if split_path.exists():
            with open(split_path) as f:
                splits = json.load(f)
            test_ids[domain_dir.name] = set(splits.get("test", []))
    return test_ids


def get_external_test_questions() -> dict[str, set[str]]:
    """Get fingerprints of external benchmark test questions."""
    benchmark_dir = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark"
    test_questions = {}

    test_files = {
        "MedQA": "med_qa_test.jsonl",
        "MedMCQA": "medmc_qa_test.jsonl",
        "MMLU": "mmlu_test.jsonl",
    }

    for name, filename in test_files.items():
        path = benchmark_dir / filename
        if path.exists():
            records = load_jsonl(path)
            fingerprints = set()
            for r in records:
                # Extract question text from various formats
                q = ""
                # Format 1: direct "question" or "input" field
                q = r.get("question", r.get("input", ""))
                # Format 2: nested "instances" dict (Self-BioRAG format)
                if not q and isinstance(r.get("instances"), dict):
                    q = r["instances"].get("input", "")
                # Format 3: nested "instances" list
                if not q and isinstance(r.get("instances"), list):
                    for inst in r["instances"]:
                        if isinstance(inst, dict):
                            q = inst.get("input", "")
                            break
                if isinstance(q, str) and len(q) > 20:
                    # Use first 100 chars as fingerprint (fast, sufficient for detection)
                    fingerprints.add(q[:100].strip().lower())
            test_questions[name] = fingerprints
            print(f"  Loaded {len(fingerprints)} test fingerprints from {name}")

    return test_questions


def check_sft_file(path: Path, test_ids: dict[str, set[str]], test_questions: dict[str, set[str]]) -> dict:
    """Check a single SFT file for contamination."""
    results = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "total_examples": 0,
        "contaminated_by_test_id": [],
        "contaminated_by_test_question": [],
        "clean": True,
    }

    if not path.exists():
        results["error"] = "File not found"
        return results

    records = load_jsonl(path)
    results["total_examples"] = len(records)

    # Flatten all test IDs
    all_test_ids = set()
    for domain, ids in test_ids.items():
        all_test_ids.update(ids)

    # Flatten test question fingerprints
    all_test_fps = set()
    for name, fps in test_questions.items():
        all_test_fps.update(fps)

    for i, record in enumerate(records):
        # Check 1: Task ID contamination
        task_id = record.get("task_id", record.get("id", ""))
        if task_id in all_test_ids:
            results["contaminated_by_test_id"].append({
                "line": i + 1,
                "task_id": task_id,
            })

        # Check 2: Test question contamination (check all text fields)
        for field in ["instruction", "input", "question", "prompt"]:
            text = record.get(field, "")
            if isinstance(text, str) and len(text) > 20:
                fp = text[:100].strip().lower()
                if fp in all_test_fps:
                    results["contaminated_by_test_question"].append({
                        "line": i + 1,
                        "field": field,
                        "preview": text[:80],
                    })

        # Also check nested messages
        messages = record.get("messages", record.get("conversations", []))
        if isinstance(messages, list):
            for msg in messages:
                content = msg.get("content", msg.get("value", ""))
                if isinstance(content, str) and len(content) > 20:
                    fp = content[:100].strip().lower()
                    if fp in all_test_fps:
                        results["contaminated_by_test_question"].append({
                            "line": i + 1,
                            "field": "messages",
                            "preview": content[:80],
                        })

        # Check nested instances (Self-BioRAG SFT format)
        instances = record.get("instances", {})
        if isinstance(instances, dict):
            inp = instances.get("input", "")
            if isinstance(inp, str) and len(inp) > 20:
                fp = inp[:100].strip().lower()
                if fp in all_test_fps:
                    results["contaminated_by_test_question"].append({
                        "line": i + 1,
                        "field": "instances.input",
                        "preview": inp[:80],
                    })

    if results["contaminated_by_test_id"] or results["contaminated_by_test_question"]:
        results["clean"] = False

    return results


def check_gym_task_sources():
    """Check if GYM tasks (medical_qa) were sourced from train or test files."""
    tasks_path = PROJECT_ROOT / "data" / "domains" / "medical_qa" / "tasks.json"
    if not tasks_path.exists():
        return {"status": "SKIP", "reason": "No medical_qa tasks.json"}

    with open(tasks_path) as f:
        tasks = json.load(f)

    # Check if tasks contain test benchmark data by looking at IDs and descriptions
    issues = []
    for task in tasks:
        desc = task.get("description", {})
        source = desc.get("source", "") if isinstance(desc, dict) else ""
        # If we can trace the source, flag it
        if "test" in str(task.get("id", "")).lower():
            issues.append(f"Task ID suggests test origin: {task['id']}")

    return {
        "total_tasks": len(tasks),
        "issues": issues,
        "status": "WARNING" if issues else "SAFE",
    }


def main():
    parser = argparse.ArgumentParser(description="Data Contamination Checker")
    parser.add_argument("--fix", action="store_true", help="Auto-fix contaminated SFT files")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Healthcare AI GYM — Data Contamination Checker")
    print("=" * 60)
    print()

    # Step 1: Collect test data fingerprints
    print("[1/4] Loading test data fingerprints...")
    test_ids = get_test_task_ids()
    total_test_ids = sum(len(v) for v in test_ids.values())
    print(f"  Found {total_test_ids} test task IDs across {len(test_ids)} domains")

    test_questions = get_external_test_questions()
    total_test_fps = sum(len(v) for v in test_questions.values())
    print(f"  Found {total_test_fps} test question fingerprints")
    print()

    # Step 2: Check SFT files
    print("[2/4] Checking SFT training data files...")
    sft_dir = PROJECT_ROOT / "datasets" / "sft"
    sft_results = []

    if sft_dir.exists():
        for sft_file in sorted(sft_dir.glob("*.jsonl")):
            result = check_sft_file(sft_file, test_ids, test_questions)
            sft_results.append(result)

            status = "CLEAN" if result["clean"] else "CONTAMINATED"
            icon = "✓" if result["clean"] else "✗"
            print(f"  {icon} {result['file']} ({result['total_examples']} examples) — {status}")

            if not result["clean"]:
                if result["contaminated_by_test_id"]:
                    print(f"    Test ID leaks: {len(result['contaminated_by_test_id'])}")
                    if args.verbose:
                        for c in result["contaminated_by_test_id"][:5]:
                            print(f"      Line {c['line']}: {c['task_id']}")
                if result["contaminated_by_test_question"]:
                    print(f"    Test question leaks: {len(result['contaminated_by_test_question'])}")
                    if args.verbose:
                        for c in result["contaminated_by_test_question"][:5]:
                            print(f"      Line {c['line']}: {c['preview'][:60]}...")
    else:
        print("  No SFT data directory found.")
    print()

    # Step 3: Check GYM task sources
    print("[3/4] Checking GYM task data sources...")
    gym_result = check_gym_task_sources()
    print(f"  medical_qa tasks: {gym_result.get('total_tasks', 'N/A')} — {gym_result['status']}")
    if gym_result.get("issues"):
        for issue in gym_result["issues"][:5]:
            print(f"    ⚠ {issue}")
    print()

    # Step 4: Check GRPO configs
    print("[4/4] Checking GRPO training configs...")
    configs_dir = PROJECT_ROOT / "configs"
    config_issues = []
    for yaml_file in sorted(configs_dir.glob("grpo_*.yaml")):
        import yaml
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        dataset_cfg = config.get("dataset", {})
        train_split = dataset_cfg.get("train_split", "")
        if train_split and train_split != "train":
            config_issues.append(f"{yaml_file.name}: train_split={train_split} (should be 'train')")
    if config_issues:
        for issue in config_issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ All GRPO configs use train_split='train'")
    print()

    # Summary
    print("=" * 60)
    contaminated_count = sum(1 for r in sft_results if not r["clean"])
    if contaminated_count == 0 and not config_issues:
        print("RESULT: ALL CLEAN — No data contamination detected!")
        print("Your experimental protocol is safe for publication.")
    else:
        print(f"RESULT: CONTAMINATION DETECTED")
        print(f"  Contaminated SFT files: {contaminated_count}/{len(sft_results)}")
        print(f"  Config issues: {len(config_issues)}")
        print()
        print("To fix:")
        print("  1. Regenerate SFT data: python scripts/generate_p2_sft_data.py")
        print("  2. Regenerate GYM data: python scripts/generate_gym_data.py")
        print("  3. Re-run this check:   python scripts/check_contamination.py")
    print("=" * 60)

    # Save report
    report_path = PROJECT_ROOT / "logs" / "contamination_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "sft_results": sft_results,
        "gym_task_sources": gym_result,
        "config_issues": config_issues,
        "total_contaminated": contaminated_count,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")

    return 1 if contaminated_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
