#!/usr/bin/env python3
"""Rescore stored multiturn eval artifacts under the 2026-08-14 extraction fixes.

Three scoring defects were found in the campaign's text-benchmark artifacts,
two of which are repairable from the stored ``submitted`` strings alone — no
re-inference:

  S1  MMLU option-sentinel. The self-biorag export ends every MMLU question
      with a dangling "\\nOption: " that the option parser absorbed into the
      LAST option's text, so 352/1,089 golds could not map to any letter for
      ANY arm (accuracy depressed 18–30pp, arm-dependently). Repair: reload
      options with the fixed loader and re-run ``_check_answer``.
  S2  Unclosed submit_answer. ``<parameter=answer>`` blocks missing their
      ``</parameter>`` parsed to ``arguments={}``; the emitted answer letter
      was dropped, the raw XML recorded as the answer, and the row scored
      wrong. 197 rows campaign-wide, concentrated on single arms. Repair:
      recover the answer from the stored string (it survives inside the
      100-char truncation for MC letters).
  S3  No-answer fallback. Episodes that exhausted the turn cap without a
      final answer had the trailing tool-call XML recorded as ``submitted``.
      NOT repairable post-hoc (the row keeps no transcript); those rows are
      *classified* here (answer_source="none") so the two conventions below
      are computable, and they need a re-run for a true answer.

Two conventions are reported side by side, because they differ by up to ~3pp
on exactly the contrasts under review and a number that does not say which
convention produced it is not reportable:

  strict     a malformed submit_answer is a failure (S1 applied, S2 not)
  recovered  the emitted answer counts       (S1 and S2 applied)

Usage:
    python3 scripts/rebuttal/rescore_extraction.py \
        --results-root /path/to/eval_results \
        --out-root     /path/to/eval_results_rescored \
        [--arms q4b_grpo_s480 ...] [--benchmarks medqa mmlu]

Writes one rescored JSON per input file (originals untouched) plus
``RESCORE_SUMMARY.md`` at the out-root. Only MC text benchmarks (medqa, mmlu)
are re-SCORED; LFQA files get classification columns only (their primary
metric needs the rouge stack and their artifacts need a re-run anyway); VQA
files are skipped (0 artifacts across all stored files — their capped episodes
structurally end in a parsed submit_answer).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # scripts/
sys.path.insert(0, str(_HERE.parents[2]))          # repo root

from eval_benchmark_multiturn import load_textqa_benchmark, _check_answer  # noqa: E402

MC_BENCHMARKS = ("medqa", "mmlu")
LFQA_BENCHMARKS = ("kqa_golden", "medication_qa", "live_qa", "healthsearch_qa", "kqa_silver")

# Non-answer predicate over STORED strings. Matches the sentinel-era analysis
# (stat-gate's DEGENERATE_RX) plus the JSON-form tool call that only the
# untrained backbones emit as a final message — the XML-only predicate
# under-counted base's non-answers 1.3% -> 8.5% on MedQA.
NON_ANSWER_RX = re.compile(r"<tool_call>|<function=|^\s*\{\s*\"name\"\s*:|^\s*$", re.I)

# S2 recovery over the stored (possibly 100-char-truncated) string.
RECOVER_RX = re.compile(
    r"<parameter=answer>\s*(.*?)\s*(?:</parameter>|</function>|</tool_call>|\Z)", re.DOTALL)


def classify(submitted: str) -> str:
    """strict-convention answer_source for a stored row."""
    if NON_ANSWER_RX.search(submitted or ""):
        return "none"
    return "answered"


def recover(submitted: str) -> str | None:
    """S2: the answer the model emitted inside a malformed submit_answer, if any."""
    m = RECOVER_RX.search(submitted or "")
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def rescore_file(path: str, bench: str, taskmap: dict | None):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = doc.get("results", [])
    out_rows = []
    n = len(rows)
    stats = dict(n=n, acc_stored=doc.get("accuracy"),
                 correct_strict=0, correct_recovered=0,
                 none_strict=0, none_recovered=0, n_recovered_rows=0)
    for r in rows:
        sub = str(r.get("submitted", ""))
        src = classify(sub)
        rec = recover(sub) if src == "none" else None
        row = dict(r)
        row["answer_source_strict"] = src
        row["recovered_answer"] = rec
        if bench in MC_BENCHMARKS and taskmap is not None:
            gold, options = taskmap.get(r.get("task_id"), (r.get("gold", ""), {}))
            c_strict = (src == "answered") and _check_answer(sub, gold, options)
            eff = rec if (rec is not None) else (sub if src == "answered" else "")
            c_rec = bool(eff) and _check_answer(eff, gold, options)
            row["correct_strict"] = bool(c_strict)
            row["correct_recovered"] = bool(c_rec)
            stats["correct_strict"] += bool(c_strict)
            stats["correct_recovered"] += bool(c_rec)
        stats["none_strict"] += (src == "none")
        stats["none_recovered"] += (src == "none" and rec is None)
        stats["n_recovered_rows"] += (rec is not None)
        out_rows.append(row)

    doc["results"] = out_rows
    doc["rescore"] = {
        "version": "rescore-extraction/2026-08-14",
        "conventions": ["strict (S1)", "recovered (S1+S2)"],
        "non_answer_predicate": NON_ANSWER_RX.pattern,
    }
    if bench in MC_BENCHMARKS:
        doc["accuracy_strict"] = stats["correct_strict"] / max(n, 1)
        doc["accuracy_recovered"] = stats["correct_recovered"] / max(n, 1)
    doc["no_answer_rate_strict"] = stats["none_strict"] / max(n, 1)
    doc["no_answer_rate_recovered"] = stats["none_recovered"] / max(n, 1)
    return doc, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm dir names to include (default: all)")
    ap.add_argument("--benchmarks", nargs="*",
                    default=list(MC_BENCHMARKS) + list(LFQA_BENCHMARKS))
    args = ap.parse_args()

    root, out_root = Path(args.results_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    taskmaps = {}
    for b in args.benchmarks:
        if b in MC_BENCHMARKS:
            taskmaps[b] = {t["id"]: (t["correct_answer"].strip(), t.get("options", {}))
                           for t in load_textqa_benchmark(b)}

    lines = ["# Rescore summary (strict = S1, recovered = S1+S2) — "
             "generated by scripts/rebuttal/rescore_extraction.py",
             "",
             "| arm | bench | file | n | acc stored | acc strict | acc recovered | "
             "no-answer strict | no-answer recovered | recovered rows |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    n_files = 0
    for arm_dir in sorted(root.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        if args.arms and arm not in args.arms:
            continue
        for b in args.benchmarks:
            for p in sorted(glob.glob(str(arm_dir / f"{b}_multiturn_*.json"))):
                doc, s = rescore_file(p, b, taskmaps.get(b))
                dst = out_root / arm / os.path.basename(p)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh, indent=2, ensure_ascii=False)
                n_files += 1
                acc_s = f"{doc.get('accuracy_strict', float('nan')):.4f}" if b in MC_BENCHMARKS else "—"
                acc_r = f"{doc.get('accuracy_recovered', float('nan')):.4f}" if b in MC_BENCHMARKS else "—"
                lines.append(
                    f"| {arm} | {b} | {os.path.basename(p)[-20:-5]} | {s['n']} "
                    f"| {s['acc_stored']:.4f} | {acc_s} | {acc_r} "
                    f"| {100*s['none_strict']/max(s['n'],1):.1f}% "
                    f"| {100*s['none_recovered']/max(s['n'],1):.1f}% "
                    f"| {s['n_recovered_rows']} |")

    summary_path = out_root / "RESCORE_SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {n_files} file(s) rescored -> {out_root}")
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
