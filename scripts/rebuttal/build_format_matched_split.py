#!/usr/bin/env python3
"""Build the split that separates gradeability from answer format.

WHY THIS EXISTS
---------------
`build_gold_complete_split.py --policy drop` removes the training rows whose
ground_truth is empty, and the arm trained on the result recovers ~8 pp of
MedQA. But that intervention is not one variable. Every empty-gold row in the
pool is open-ended, so dropping them also re-weights the answer format:

    full_4modality_clean       3,390 train rows   38.2% multiple-choice
    full_4modality_gold_drop   1,834 train rows   70.6% multiple-choice

MedQA is a multiple-choice benchmark. "We removed ungradeable rows" and "we
trained on 1.85x more multiple-choice" describe the same partition of this pool,
and the second is the explanation a reviewer reaches for first.

The train pool is a 2x2 with one empty cell:

                     gold present    gold empty
    multiple-choice        1,294             0
    open-ended               540         1,556

so format and gradeability cannot be crossed by subsetting alone. What CAN be
built is a split with gold_drop's exact size and exact format profile whose
open-ended half is ungradeable instead of gradeable:

    1,294 multiple-choice (gradeable, the same rows gold_drop uses)
  +   540 open-ended with EMPTY ground_truth, sampled from the 1,556
  = 1,834 rows, 70.6% multiple-choice

Read the three arms together:

    clean      3,390 rows, 38.2% MCQA, 54.1% gradeable   -> MedQA 0.65-0.71
    gold_drop  1,834 rows, 70.6% MCQA,  100% gradeable   -> MedQA 0.78
    THIS       1,834 rows, 70.6% MCQA, 70.6% gradeable   -> ?

Near gold_drop means the recovery is a format effect and the gradeability story
is dead. Near clean means gradeability is the cause and the format confound is
ruled out. Either answer is publishable; the existing pair cannot produce either.

WHY IT READS THE PARQUET, NOT tasks.json
----------------------------------------
The two disagree, and the parquet is what training consumed. Classified from
tasks.json, 1,768 of the 2,096 open-ended tasks carry an answer somewhere;
classified from the converted parquet, only 540 do. The converter's field
mapping is where the other 1,228 lose their answer -- which is the defect this
whole line of work is about. Defining the intervention on the trainer's actual
input keeps the arms comparable and avoids re-running that conversion.

TEST IS NEVER TOUCHED. test.parquet is copied byte-for-byte, so all three arms
score on the same 850 rows.

    python scripts/rebuttal/build_format_matched_split.py \
        --clean <verl_parquet>/full_4modality_clean \
        --out   <verl_parquet>/full_4modality_fmtmatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import pandas as pd


def gold_empty(row) -> bool:
    rm = row["reward_model"]
    return not (rm.get("ground_truth") or "").strip()


def is_mcqa(row) -> bool:
    return bool(row["extra_info"].get("has_options"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True, help="verl_parquet dir of the unfiltered split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260811, help="fixes the open-ended draw")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clean = pathlib.Path(args.clean)
    df = pd.read_parquet(clean / "train.parquet")
    mc = df.apply(is_mcqa, axis=1)
    empty = df.apply(gold_empty, axis=1)

    cells = {
        "mc_gold": df[mc & ~empty],
        "mc_nogold": df[mc & empty],
        "open_gold": df[~mc & ~empty],
        "open_nogold": df[~mc & empty],
    }
    print(f"train rows {len(df)}")
    for k, v in cells.items():
        print(f"  {k:<12}: {len(v)}")

    n_open = len(cells["open_gold"])          # matches gold_drop's format profile exactly
    if n_open > len(cells["open_nogold"]):
        print(f"[fatal] need {n_open} empty-gold open-ended rows, have "
              f"{len(cells['open_nogold'])}", file=sys.stderr)
        return 1
    if len(cells["mc_gold"]) == 0:
        print("[fatal] no gradeable multiple-choice rows; cannot format-match", file=sys.stderr)
        return 1

    picked = cells["open_nogold"].sample(n=n_open, random_state=args.seed)
    out_df = pd.concat([cells["mc_gold"], picked]).sample(frac=1.0, random_state=args.seed) \
                                                  .reset_index(drop=True)

    # Assert the properties this split exists to have, rather than trusting the
    # arithmetic: gold_drop's size, gold_drop's format profile, and an
    # open-ended half that really is ungradeable.
    target_size = len(cells["mc_gold"]) + len(cells["open_gold"])
    target_share = len(cells["mc_gold"]) / target_size
    n_mc = int(out_df.apply(is_mcqa, axis=1).sum())
    n_gradeable = int((~out_df.apply(gold_empty, axis=1)).sum())
    assert len(out_df) == target_size, f"size {len(out_df)} != gold_drop's {target_size}"
    assert abs(n_mc / len(out_df) - target_share) < 1e-9, "format profile differs from gold_drop"
    assert n_gradeable == n_mc, "the open-ended half is supposed to be ungradeable"

    print(f"\n{len(out_df)} train rows")
    print(f"  multiple-choice : {n_mc} ({100 * n_mc / len(out_df):.1f}%)"
          f"   — gold_drop is {100 * target_share:.1f}%")
    print(f"  gradeable       : {n_gradeable} ({100 * n_gradeable / len(out_df):.1f}%)"
          f"   — gold_drop is 100.0%")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out / "train.parquet", index=False)
    shutil.copyfile(clean / "test.parquet", out / "test.parquet")

    def md5(p):
        return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()

    same_test = md5(clean / "test.parquet") == md5(out / "test.parquet")
    assert same_test, "test.parquet copy does not match the source"
    (out / "MANIFEST.json").write_text(json.dumps({
        "built_by": "build_format_matched_split.py",
        "purpose": "hold answer format at gold_drop's profile while varying gradeability",
        "seed": args.seed,
        "source": str(clean),
        "train_rows": len(out_df),
        "train_multiple_choice": n_mc,
        "train_gradeable": n_gradeable,
        "cells_in_source": {k: len(v) for k, v in cells.items()},
        "test_parquet_md5": md5(out / "test.parquet"),
        "test_identical_to_source": same_test,
    }, indent=1))
    print(f"\nwrote {out}")
    print(f"  test.parquet copied verbatim (md5 {md5(out / 'test.parquet')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
