"""Train/eval contamination audit for the RL task pool used in Healthcare AI GYM.

Answers the contamination question with numbers a reader can reproduce from the public repo:
which RL training tasks were derived from the *evaluation* split of a benchmark we
later report on, and how much of each evaluation set that accounts for.

Provenance is established from three kinds of on-disk evidence, never from memory:
  1. builder scripts that name their input file (e.g. med_qa_train_gpt4.jsonl),
  2. the ``_image_path`` a VQA task kept (``vqarad_train_00314.jpg`` -> train split),
  3. the ``category`` label the converter wrote (``MMLU_test_anatomy`` -> test split).
Anything the evidence does not settle is reported as UNVERIFIED rather than assumed
clean -- an audit that guesses in its own favour is not an audit.

Usage:
    python scripts/rebuttal/audit_contamination.py
    python scripts/rebuttal/audit_contamination.py --pool data/domains/agentic_rl_combined
    python scripts/rebuttal/audit_contamination.py --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = PROJECT_ROOT / "data" / "domains" / "full_4modality_combined"

# Evaluation-set sizes as reported in Appendix "Benchmark Suite" of the paper.
EVAL_SET_SIZE = {
    "MedQA": 1273,
    "MedMCQA": 4183,
    "MMLU-Med (6 sub.)": 1089,
    "VQA-RAD": 451,
    "SLAKE": 1061,
    "PathVQA": 6719,
    "LiveQA": 100,
    "MedicationQA": 666,
    "HealthSearchQA": 3077,
    "KQA-Golden": 201,
    "KQA-Silver": 904,
}

# (source, category-prefix) -> (benchmark, provenance, evidence)
# provenance: TRAIN  = built from the benchmark's train split, disjoint from our eval set
#             EVAL   = built from the split we evaluate on -> contamination
#             UNVERIFIED = on-disk evidence does not settle the split
PROVENANCE_RULES: list[tuple[str, str, str, str, str]] = [
    # source_exact,          category_prefix,   benchmark,            provenance,   evidence
    ("MedQA",                "",                "MedQA",              "TRAIN",
     "scripts/expand_medqa_gym_tasks.py reads med_qa_train_gpt4.jsonl"),
    ("med_qa_train",         "",                "MedQA",              "TRAIN",
     "source tag names the train file"),
    ("med_qa",               "",                "MedQA",              "EVAL",
     "medqa_loader.load_medqa_jsonl(split='test') -> med_qa_test.jsonl"),
    ("medmc_qa_train",       "",                "MedMCQA",            "TRAIN",
     "category=MedMCQA_train"),
    ("medmc_qa",             "",                "MedMCQA",            "EVAL",
     "category=MedMCQA_dev, the split MedMCQA is scored on"),
    ("",                     "MMLU_test_",      "MMLU-Med (6 sub.)",  "EVAL",
     "category names the MMLU test split"),
    ("vqa_rad",              "",                "VQA-RAD",            "TRAIN",
     "_image_path=.../vqarad_train_#####.jpg"),
    ("pathvqa",              "",                "PathVQA",            "TRAIN",
     "_image_path=.../pathvqa_train_#####.jpg"),
    ("slake",                "",                "SLAKE",              "UNVERIFIED",
     "_image_path is null; loader default is split='test' but not provable on disk"),
    ("MedLFQA/live_qa",      "",                "LiveQA",             "EVAL",
     "MedLFQA ships eval-only splits"),
    ("MedLFQA/medication_qa", "",               "MedicationQA",       "EVAL",
     "MedLFQA ships eval-only splits"),
    ("MedLFQA/healthsearch_qa", "",             "HealthSearchQA",     "EVAL",
     "MedLFQA ships eval-only splits"),
    ("MedLFQA/kqa_golden",   "",                "KQA-Golden",         "EVAL",
     "MedLFQA ships eval-only splits"),
    ("MedLFQA/kqa_silver",   "",                "KQA-Silver",         "EVAL",
     "MedLFQA ships eval-only splits"),
]


def classify(task: dict[str, Any]) -> tuple[str, str, str] | None:
    """Map one task to (benchmark, provenance, evidence), or None if not benchmark-derived."""
    desc = task.get("description") or {}
    source = str(desc.get("source", ""))
    category = str(desc.get("category", ""))
    for src, cat_prefix, bench, provenance, evidence in PROVENANCE_RULES:
        if src and source != src:
            continue
        if cat_prefix and not category.startswith(cat_prefix):
            continue
        if not src and not cat_prefix:
            continue
        return bench, provenance, evidence
    return None


def normalize(task: dict[str, Any]) -> str:
    text = task.get("raw_question") or task.get("ticket") or ""
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def load_pool(pool: Path) -> tuple[list[dict], dict[str, list[str]]]:
    tasks = json.loads((pool / "tasks.json").read_text())
    splits = json.loads((pool / "split_tasks.json").read_text())
    return tasks, splits


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL,
                    help="task pool directory holding tasks.json + split_tasks.json")
    ap.add_argument("--json", type=Path, default=None, help="also write findings as JSON")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tasks, splits = load_pool(args.pool)
    by_id: dict[str, dict] = {t["id"]: t for t in tasks}
    train_ids = [i for i in splits.get("train", []) if i in by_id]
    test_ids = [i for i in splits.get("test", []) if i in by_id]
    train_set, test_set = set(train_ids), set(test_ids)

    print(f"pool: {args.pool.relative_to(PROJECT_ROOT)}")
    print(f"rows in tasks.json: {len(tasks)}   unique task ids: {len(by_id)}")

    # ---- 1. integrity of the pool itself -------------------------------------
    section("1. Task-pool integrity")
    dup_rows = len(tasks) - len(by_id)
    id_overlap = sorted(train_set & test_set)
    print(f"duplicate rows sharing an id      : {dup_rows}")
    print(f"unique train tasks                : {len(train_set)}")
    print(f"unique held-out tasks             : {len(test_set)}")
    print(f"ids listed in BOTH train and test : {len(id_overlap)}")
    if id_overlap:
        print(f"  e.g. {id_overlap[:5]}")

    train_text: dict[str, str] = {}
    for i in train_set:
        text = normalize(by_id[i])
        if len(text) > 20:
            train_text.setdefault(text, i)
    text_dupes = [i for i in test_set
                  if len(normalize(by_id[i])) > 20 and normalize(by_id[i]) in train_text]
    pct = 100 * len(text_dupes) / len(test_set) if test_set else 0.0
    print(f"held-out tasks whose question text also appears in train: "
          f"{len(text_dupes)} ({pct:.1f}%)")

    # ---- 2. provenance ledger ------------------------------------------------
    section("2. Provenance of benchmark-derived training tasks")
    ledger: dict[tuple[str, str], int] = collections.Counter()
    evidence_of: dict[tuple[str, str], str] = {}
    unmapped = 0
    for i in train_set:
        hit = classify(by_id[i])
        if hit is None:
            unmapped += 1
            continue
        bench, provenance, evidence = hit
        ledger[(bench, provenance)] += 1
        evidence_of[(bench, provenance)] = evidence

    print(f"{'benchmark':<22} {'provenance':<11} {'tasks':>6} {'% of eval set':>14}  evidence")
    rows: list[dict[str, Any]] = []
    for (bench, provenance), n in sorted(ledger.items(), key=lambda kv: (-kv[1], kv[0])):
        size = EVAL_SET_SIZE.get(bench)
        share = f"{100 * n / size:.1f}%" if size and provenance != "TRAIN" else "--"
        print(f"{bench:<22} {provenance:<11} {n:>6} {share:>14}  {evidence_of[(bench, provenance)]}")
        rows.append({"benchmark": bench, "provenance": provenance, "tasks": n,
                     "eval_set_size": size, "share_of_eval_set": share,
                     "evidence": evidence_of[(bench, provenance)]})

    contaminated = sum(n for (_, p), n in ledger.items() if p == "EVAL")
    unverified = sum(n for (_, p), n in ledger.items() if p == "UNVERIFIED")
    clean = sum(n for (_, p), n in ledger.items() if p == "TRAIN")

    section("3. Summary")
    total = len(train_set)
    print(f"train tasks from a benchmark TRAIN split : {clean:>5} ({100*clean/total:.1f}%)")
    print(f"train tasks from a benchmark EVAL  split : {contaminated:>5} ({100*contaminated/total:.1f}%)")
    print(f"train tasks with UNVERIFIED split        : {unverified:>5} ({100*unverified/total:.1f}%)")
    print(f"train tasks not benchmark-derived        : {unmapped:>5} ({100*unmapped/total:.1f}%)")

    if args.json:
        args.json.write_text(json.dumps({
            "pool": str(args.pool),
            "rows": len(tasks), "unique_ids": len(by_id), "duplicate_rows": dup_rows,
            "train_tasks": len(train_set), "heldout_tasks": len(test_set),
            "ids_in_both_splits": id_overlap,
            "heldout_text_duplicates": len(text_dupes),
            "ledger": rows,
            "totals": {"train_split": clean, "eval_split": contaminated,
                       "unverified": unverified, "not_benchmark_derived": unmapped},
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
