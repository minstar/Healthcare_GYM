"""Build a deduplicated, decontaminated train/test split for the RL task pool.

Companion to ``scripts/rebuttal/audit_contamination.py``: the audit *measures* the
defects (duplicate rows, train/test id overlap, benchmark-EVAL provenance in
train, held-out questions whose text also appears in train); this script
*removes* them and writes a clean pool that
``scripts/verl/convert_tasks_to_parquet.py`` consumes unchanged.

Provenance detection and text normalization are imported from the audit module
itself -- never re-implemented -- so the cleaning rule is byte-for-byte the rule
the rebuttal already published.  Tasks the audit maps to a benchmark TRAIN split
are kept (they are disjoint from our evaluation sets); tasks it maps to a
benchmark EVAL split are dropped from train; UNVERIFIED tasks are kept but
counted in the manifest so the residual risk stays visible.

Cleaning order (each step records the exact ids it dropped in MANIFEST.json):
  1. deduplicate tasks.json by id, keeping the FIRST occurrence;
  2. drop split entries whose id is missing from tasks.json, and duplicate
     entries within each split list (first occurrence kept, order preserved);
  3. drop from TRAIN every task whose provenance is a benchmark EVAL split;
  4. drop from TRAIN every id that also appears in TEST;
  5. drop from TEST every item whose normalized question text also appears in
     the remaining (cleaned) train pool, using the audit's normalize() and its
     >20-character guard.

The script is deterministic and idempotent: no timestamps, set-derived lists
are sorted before writing, split order otherwise preserves the input order.

Usage:
    python scripts/rebuttal/build_clean_split.py
    python scripts/rebuttal/build_clean_split.py --pool data/domains/full_4modality_combined \
        --out data/domains/full_4modality_clean
    python scripts/rebuttal/build_clean_split.py --dry-run
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = PROJECT_ROOT / "scripts" / "rebuttal" / "audit_contamination.py"
TEXT_MIN_LEN = 20  # same guard as audit_contamination.py uses for text overlap


def load_audit_module():
    """Import scripts/rebuttal/audit_contamination.py so classify()/normalize()
    are reused verbatim -- the single source of truth for provenance rules."""
    spec = importlib.util.spec_from_file_location("audit_contamination", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def domain_distribution(ids: Iterable[str], by_id: dict[str, dict]) -> dict[str, int]:
    counter = collections.Counter(
        str(by_id[i].get("_source_domain", "unknown")) for i in ids
    )
    return {k: counter[k] for k in sorted(counter)}


def provenance_ledger(ids: Iterable[str], by_id: dict[str, dict], classify) -> list[dict[str, Any]]:
    counter: collections.Counter[tuple[str, str]] = collections.Counter()
    not_benchmark = 0
    for i in ids:
        hit = classify(by_id[i])
        if hit is None:
            not_benchmark += 1
            continue
        bench, provenance, _evidence = hit
        counter[(bench, provenance)] += 1
    rows = [
        {"benchmark": bench, "provenance": provenance, "tasks": n}
        for (bench, provenance), n in sorted(counter.items())
    ]
    rows.append({"benchmark": None, "provenance": "NOT_BENCHMARK_DERIVED",
                 "tasks": not_benchmark})
    return rows


def dedup_keep_first(items: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (unique items in first-seen order, duplicate occurrences dropped)."""
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []
    for item in items:
        if item in seen:
            dropped.append(item)
        else:
            seen.add(item)
            kept.append(item)
    return kept, dropped


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path,
                    default=Path("data/domains/full_4modality_combined"),
                    help="input pool directory holding tasks.json + split_tasks.json")
    ap.add_argument("--out", type=Path,
                    default=Path("data/domains/full_4modality_clean"),
                    help="output pool directory (created; input is never modified)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be done without writing any files")
    args = ap.parse_args(list(argv) if argv is not None else None)

    pool = resolve(args.pool)
    out = resolve(args.out)
    audit = load_audit_module()

    tasks = json.loads((pool / "tasks.json").read_text())
    splits = json.loads((pool / "split_tasks.json").read_text())
    in_train: list[str] = list(splits.get("train", []))
    in_test: list[str] = list(splits.get("test", []))

    # ---- 1. deduplicate tasks.json by id, keep FIRST occurrence --------------
    by_id: dict[str, dict] = {}
    clean_tasks: list[dict] = []
    dup_row_ids: list[str] = []
    for task in tasks:
        tid = task["id"]
        if tid in by_id:
            dup_row_ids.append(tid)
        else:
            by_id[tid] = task
            clean_tasks.append(task)

    # ---- 2. sanitize split lists: unknown ids, duplicate entries -------------
    def sanitize(entries: list[str]) -> tuple[list[str], list[str], list[str]]:
        known = [i for i in entries if i in by_id]
        missing = [i for i in entries if i not in by_id]
        unique, dup_entries = dedup_keep_first(known)
        return unique, missing, dup_entries

    train_ids, train_missing, train_dup_entries = sanitize(in_train)
    test_ids, test_missing, test_dup_entries = sanitize(in_test)
    test_set = set(test_ids)

    ledger_before = provenance_ledger(train_ids, by_id, audit.classify)
    domains_before = {
        "train": domain_distribution(train_ids, by_id),
        "test": domain_distribution(test_ids, by_id),
    }

    # ---- 3. drop benchmark-EVAL-provenance tasks from TRAIN ------------------
    eval_dropped: list[str] = []
    eval_by_benchmark: collections.Counter[str] = collections.Counter()
    kept_after_eval: list[str] = []
    for i in train_ids:
        hit = audit.classify(by_id[i])
        if hit is not None and hit[1] == "EVAL":
            eval_dropped.append(i)
            eval_by_benchmark[hit[0]] += 1
        else:
            kept_after_eval.append(i)

    # ---- 4. drop from TRAIN every id that also appears in TEST ---------------
    overlap_dropped = [i for i in kept_after_eval if i in test_set]
    final_train = [i for i in kept_after_eval if i not in test_set]

    # ---- 5. drop from TEST items whose question text appears in final TRAIN --
    train_text: set[str] = set()
    for i in final_train:
        text = audit.normalize(by_id[i])
        if len(text) > TEXT_MIN_LEN:
            train_text.add(text)
    text_dropped: list[str] = []
    final_test: list[str] = []
    for i in test_ids:
        text = audit.normalize(by_id[i])
        if len(text) > TEXT_MIN_LEN and text in train_text:
            text_dropped.append(i)
        else:
            final_test.append(i)

    ledger_after = provenance_ledger(final_train, by_id, audit.classify)
    domains_after = {
        "train": domain_distribution(final_train, by_id),
        "test": domain_distribution(final_test, by_id),
    }

    manifest = {
        "builder": "scripts/rebuttal/build_clean_split.py",
        "provenance_rules_from": rel(AUDIT_PATH),
        "input": {
            "pool": rel(pool),
            "tasks_rows": len(tasks),
            "tasks_unique_ids": len(by_id),
            "train_entries": len(in_train),
            "train_unique_ids": len(train_ids),
            "test_entries": len(in_test),
            "test_unique_ids": len(test_ids),
        },
        "drops": {
            "tasks_duplicate_rows": {
                "reason": "row shares an id with an earlier row in tasks.json; "
                          "first occurrence kept",
                "count": len(dup_row_ids),
                "ids": sorted(set(dup_row_ids)),
                "note": f"{len(dup_row_ids)} rows dropped across "
                        f"{len(set(dup_row_ids))} distinct ids",
            },
            "train_missing_from_tasks": {
                "reason": "train split entry has no row in tasks.json",
                "count": len(train_missing),
                "ids": sorted(set(train_missing)),
            },
            "train_duplicate_split_entries": {
                "reason": "id listed more than once in the train split; "
                          "first occurrence kept",
                "count": len(train_dup_entries),
                "ids": sorted(set(train_dup_entries)),
            },
            "test_missing_from_tasks": {
                "reason": "test split entry has no row in tasks.json",
                "count": len(test_missing),
                "ids": sorted(set(test_missing)),
            },
            "test_duplicate_split_entries": {
                "reason": "id listed more than once in the test split; "
                          "first occurrence kept",
                "count": len(test_dup_entries),
                "ids": sorted(set(test_dup_entries)),
            },
            "train_eval_provenance": {
                "reason": "task derived from a benchmark EVAL split "
                          "(audit_contamination.classify -> provenance EVAL); "
                          "benchmark-TRAIN and UNVERIFIED tasks are kept",
                "count": len(eval_dropped),
                "by_benchmark": {k: eval_by_benchmark[k]
                                 for k in sorted(eval_by_benchmark)},
                "ids": sorted(eval_dropped),
            },
            "train_id_in_test": {
                "reason": "id listed in both train and test splits; "
                          "kept in test, dropped from train",
                "count": len(overlap_dropped),
                "ids": sorted(overlap_dropped),
            },
            "test_text_in_train": {
                "reason": "normalized question text (audit_contamination.normalize, "
                          f">{TEXT_MIN_LEN} chars) also appears in the cleaned "
                          "train pool",
                "count": len(text_dropped),
                "ids": sorted(text_dropped),
            },
        },
        "output": {
            "pool": rel(out),
            "tasks_rows": len(clean_tasks),
            "train": len(final_train),
            "test": len(final_test),
        },
        "train_provenance_ledger": {
            "before": ledger_before,
            "after": ledger_after,
        },
        "domain_distribution": {
            "before": domains_before,
            "after": domains_after,
        },
    }

    # ---- report --------------------------------------------------------------
    print(f"pool: {rel(pool)}  ->  {rel(out)}{'  (dry run)' if args.dry_run else ''}")
    print(f"tasks.json rows            : {len(tasks)} -> {len(clean_tasks)} "
          f"(-{len(dup_row_ids)} duplicate rows)")
    print(f"train entries              : {len(in_train)} -> {len(final_train)}")
    print(f"  - missing from tasks.json: {len(train_missing)}")
    print(f"  - duplicate split entries: {len(train_dup_entries)}")
    print(f"  - benchmark-EVAL prov.   : {len(eval_dropped)} "
          f"{dict(sorted(eval_by_benchmark.items()))}")
    print(f"  - id also in test        : {len(overlap_dropped)}")
    print(f"test entries               : {len(in_test)} -> {len(final_test)}")
    print(f"  - missing from tasks.json: {len(test_missing)}")
    print(f"  - duplicate split entries: {len(test_dup_entries)}")
    print(f"  - question text in train : {len(text_dropped)}")

    if args.dry_run:
        print("\ndry run: no files written")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "tasks.json", clean_tasks)
    dump_json(out / "split_tasks.json", {"train": final_train, "test": final_test})
    dump_json(out / "MANIFEST.json", manifest)
    print(f"\nwrote {rel(out)}/tasks.json, split_tasks.json, MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
