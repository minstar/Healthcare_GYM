"""Verify a cleaned RL task pool produced by scripts/rebuttal/build_clean_split.py.

Loads the OUTPUT pool and asserts, exiting non-zero on any failure:
  1. tasks.json contains no duplicate ids;
  2. every id listed in split_tasks.json exists in tasks.json;
  3. no id is listed more than once within a split list;
  4. train and test splits share no id;
  5. no train task has benchmark-EVAL provenance
     (per scripts/rebuttal/audit_contamination.classify, reused verbatim);
  6. no held-out question's normalized text appears in the train pool
     (per audit_contamination.normalize, same >20-character guard).

Usage:
    python scripts/rebuttal/verify_clean_split.py
    python scripts/rebuttal/verify_clean_split.py --pool data/domains/full_4modality_clean
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = PROJECT_ROOT / "scripts" / "rebuttal" / "audit_contamination.py"
TEXT_MIN_LEN = 20  # same guard as audit_contamination.py uses for text overlap


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_contamination", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path,
                    default=Path("data/domains/full_4modality_clean"),
                    help="cleaned pool directory holding tasks.json + split_tasks.json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    pool = args.pool if args.pool.is_absolute() else PROJECT_ROOT / args.pool
    audit = load_audit_module()

    tasks = json.loads((pool / "tasks.json").read_text())
    splits = json.loads((pool / "split_tasks.json").read_text())
    train = list(splits.get("train", []))
    test = list(splits.get("test", []))
    by_id = {}
    for t in tasks:
        by_id.setdefault(t["id"], t)

    failures = 0

    def check(name: str, bad: list, detail: str = "") -> None:
        nonlocal failures
        if bad:
            failures += 1
            print(f"FAIL {name}: {len(bad)} offending ids, e.g. {sorted(bad)[:5]}")
        else:
            print(f"PASS {name}{'  (' + detail + ')' if detail else ''}")

    print(f"pool: {pool}")
    print(f"tasks.json rows: {len(tasks)}   train: {len(train)}   test: {len(test)}\n")

    # 1. no duplicate ids in tasks.json
    id_counts = collections.Counter(t["id"] for t in tasks)
    dup_ids = [i for i, n in id_counts.items() if n > 1]
    check("no duplicate ids in tasks.json", dup_ids,
          f"{len(id_counts)} unique ids")

    # 2. every split id exists in tasks.json
    unknown = [i for i in train + test if i not in by_id]
    check("every split id exists in tasks.json", unknown,
          f"{len(train) + len(test)} split entries checked")

    # 3. no duplicate entries within a split list
    dup_entries = ([i for i, n in collections.Counter(train).items() if n > 1]
                   + [i for i, n in collections.Counter(test).items() if n > 1])
    check("no duplicate entries within a split list", dup_entries)

    # 4. train ∩ test = ∅
    overlap = sorted(set(train) & set(test))
    check("train and test splits share no id", overlap,
          f"{len(set(train))} train vs {len(set(test))} test")

    # 5. no train task has benchmark-EVAL provenance
    eval_tasks = []
    for i in train:
        if i not in by_id:
            continue
        hit = audit.classify(by_id[i])
        if hit is not None and hit[1] == "EVAL":
            eval_tasks.append(i)
    check("no train task has benchmark-EVAL provenance", eval_tasks)

    # 6. no held-out question text appears in the train pool
    train_text = set()
    for i in train:
        if i not in by_id:
            continue
        text = audit.normalize(by_id[i])
        if len(text) > TEXT_MIN_LEN:
            train_text.add(text)
    text_dupes = []
    for i in test:
        if i not in by_id:
            continue
        text = audit.normalize(by_id[i])
        if len(text) > TEXT_MIN_LEN and text in train_text:
            text_dupes.append(i)
    check("no held-out question text appears in train", text_dupes,
          f"{len(train_text)} normalized train texts")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
