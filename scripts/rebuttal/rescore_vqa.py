#!/usr/bin/env python3
"""Rescore stored visual-QA rollouts under BOTH scoring rules and print the pairs.

This is the rebuttal-facing deliverable: what the numbers were (substring
containment, the rule that produced every published VQA row) and what they are
(CF-EM, scripts/vqa_scoring.py).

It reads only stored rollouts, so it needs no GPU and no model.

    python scripts/rebuttal/rescore_vqa.py \
        --rollouts /data/project/private/minstar/workspace/hcgym_rebuttal/eval_results \
        --benchmark vqa_rad

Every rollout file must carry per-row `submitted` and `gold`. The dataset row
index is recovered from `task_id` (minted as f"{benchmark}_{row}"), which is
what lets the closed-set head be question-conditioned leave-one-out.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import vqa_scoring  # noqa: E402


def check_answer_substring(submitted: str, gold: str, options: dict) -> bool:
    """Verbatim copy of eval_benchmark_multiturn._check_answer.

    Copied rather than imported because importing that module pulls in torch
    and a VLM stack. `--verify-shipped` replays the stored `correct` field
    through this function and asserts 0 mismatches, which is the proof that
    the copy is faithful.
    """
    import re
    submitted = submitted.strip()
    gold = gold.strip()
    if not submitted:
        return False
    if submitted.lower() == gold.lower():
        return True
    if len(gold) <= 2 and gold.upper() in "ABCDE":
        if submitted.upper().startswith(gold.upper()):
            return True
        gold_text = options.get(gold.upper(), "")
        if gold_text and submitted.lower() == gold_text.lower():
            return True
        return False
    gold_letter = None
    for letter, text in options.items():
        if text.strip().lower() == gold.lower():
            gold_letter = letter
            break
    if gold_letter:
        first_char = submitted[0].upper() if submitted else ""
        if first_char == gold_letter.upper():
            return True
        m = re.match(r'^([A-E])[.\):\s]', submitted.upper())
        if m and m.group(1) == gold_letter.upper():
            return True
    if gold and gold.lower() in submitted.lower():
        return True
    return False


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def boot_ci(flags, reps=2000, seed=0):
    """Percentile bootstrap CI for a mean of 0/1 flags."""
    if not flags:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(reps):
        means.append(sum(flags[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * reps)] * 100, means[int(0.975 * reps)] * 100)


def load_arm(path: Path):
    with open(path) as f:
        d = json.load(f)
    return d.get("results", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="directory holding <arm>/<benchmark>_partial.json")
    ap.add_argument("--benchmark", default="vqa_rad")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--verify-shipped", action="store_true", default=True,
                    help="replay the stored `correct` field through the copied "
                         "substring rule and assert 0 mismatches")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    bench = args.benchmark
    root = Path(args.rollouts)
    arms = args.arms or sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / f"{bench}_partial.json").exists())
    if not arms:
        print(f"no rollouts for {bench} under {root}")
        return 1

    vocab = vqa_scoring.load_vocab(bench, PROJECT_ROOT)
    if vocab is None:
        print(f"{bench} is not a closed-vocabulary benchmark; CF-EM does not apply")
        return 1

    print("=" * 100)
    print(f"RESCORING {bench}   rule A = substring (published)   "
          f"rule B = {vqa_scoring.CF_EM_VERSION}")
    print(f"closed vocabulary: {len(vocab.labels)} non-polar labels "
          f"over {len(vocab.items)} benchmark items")
    print("=" * 100)

    per_arm, replay_bad = {}, 0
    for arm in arms:
        rows = load_arm(root / arm / f"{bench}_partial.json")
        recs = []
        for r in rows:
            idx = vqa_scoring.task_row_index(r["task_id"], bench)
            sub = check_answer_substring(r.get("submitted", ""), r["gold"], {})
            if args.verify_shipped and "correct" in r and sub != r["correct"]:
                replay_bad += 1
            _, cf_ok, kind, span = vqa_scoring.cf_predict(
                r.get("submitted", ""), r["gold"], vocab, idx=idx)
            recs.append({"task_id": r["task_id"], "gold": r["gold"],
                         "len": len(r.get("submitted", "")),
                         "sub": sub, "cf": cf_ok, "kind": kind})
        per_arm[arm] = recs

    if args.verify_shipped:
        print(f"\nSHIPPED REPLAY: stored `correct` reproduced by the copied "
              f"substring rule with {replay_bad} mismatches "
              f"over {sum(len(v) for v in per_arm.values())} rows")
        if replay_bad:
            print("  *** the copy is NOT faithful; do not trust the pairs ***")
            return 1

    # ---- all rows per arm -------------------------------------------------
    print("\n" + "-" * 100)
    print("ALL SCORED ROWS PER ARM")
    print("-" * 100)
    print(f"  {'arm':<20}{'n':>5}{'meanlen':>9}{'substring':>11}{'CF-EM':>9}"
          f"{'delta':>9}{'CF-BAcc':>9}{'nocommit':>10}")
    for arm, recs in per_arm.items():
        n = len(recs)
        sub = sum(r["sub"] for r in recs) / n * 100
        cf = sum(r["cf"] for r in recs) / n * 100
        ml = sum(r["len"] for r in recs) / n
        srows = [{"submitted": "", "gold": r["gold"]} for r in recs]
        # recompute BAcc/nocommit through score_all for the guard number
        full = [{"submitted": x.get("submitted", ""), "gold": x["gold"],
                 "_i": vqa_scoring.task_row_index(x["task_id"], bench)}
                for x in load_arm(root / arm / f"{bench}_partial.json")]
        _, st = vqa_scoring.score_all(full, vocab)
        print(f"  {arm:<20}{n:>5}{ml:>9.0f}{sub:>10.1f}%{cf:>8.1f}%"
              f"{cf - sub:>+9.1f}{st['cf_bacc'] * 100:>8.1f}%"
              f"{st['no_commit_rate'] * 100:>9.1f}%")

    # ---- paired subset ----------------------------------------------------
    common = set.intersection(*[{r["task_id"] for r in v} for v in per_arm.values()])
    print("\n" + "-" * 100)
    print(f"PAIRED SUBSET: the {len(common)} task_ids all arms share "
          f"(the comparison that counts)")
    print("-" * 100)
    print(f"  {'arm':<20}{'n':>5}{'meanlen':>9}{'substring':>11}{'CF-EM':>9}"
          f"{'delta':>9}{'CF-EM 95% CI':>20}")
    paired = {}
    for arm, recs in per_arm.items():
        sel = sorted([r for r in recs if r["task_id"] in common],
                     key=lambda r: r["task_id"])
        paired[arm] = sel
        n = len(sel)
        sub = sum(r["sub"] for r in sel) / n * 100
        cf = sum(r["cf"] for r in sel) / n * 100
        ml = sum(r["len"] for r in sel) / n
        lo, hi = boot_ci([1 if r["cf"] else 0 for r in sel])
        print(f"  {arm:<20}{n:>5}{ml:>9.0f}{sub:>10.1f}%{cf:>8.1f}%"
              f"{cf - sub:>+9.1f}   [{lo:5.1f}, {hi:5.1f}]")

    def order(key):
        return " > ".join(
            f"{a}({sum(r[key] for r in paired[a]) / len(paired[a]) * 100:.1f})"
            for a in sorted(paired, key=lambda a: -sum(r[key] for r in paired[a])))

    print(f"\n  ORDERING under substring : {order('sub')}")
    print(f"  ORDERING under CF-EM     : {order('cf')}")

    # ---- significance -----------------------------------------------------
    print("\n" + "-" * 100)
    print("IS ANY PAIRWISE GAP SIGNIFICANT?  McNemar exact on the paired subset")
    print("-" * 100)
    names = sorted(paired)
    for key, label in (("sub", "substring"), ("cf", "CF-EM")):
        print(f"  -- {label}")
        for i in range(len(names)):
            for j in range(len(names)):
                if i >= j:
                    continue
                a, b = names[i], names[j]
                ra = {r["task_id"]: r[key] for r in paired[a]}
                rb = {r["task_id"]: r[key] for r in paired[b]}
                bb = sum(1 for t in common if ra[t] and not rb[t])
                cc = sum(1 for t in common if rb[t] and not ra[t])
                d = (sum(ra.values()) - sum(rb.values())) / len(common) * 100
                p = mcnemar_exact(bb, cc)
                flag = "" if p >= 0.05 else "   <-- significant"
                print(f"     {a:<18} - {b:<18} = {d:>+6.2f} pp   "
                      f"(discordant {bb}/{cc}, McNemar p={p:.3f}){flag}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"benchmark": bench,
                       "cf_version": vqa_scoring.CF_EM_VERSION,
                       "arms": {a: v for a, v in per_arm.items()}}, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
