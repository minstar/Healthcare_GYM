#!/usr/bin/env python3
"""Measure what each reward dimension actually contributes to the GRPO gradient.

Motivation
----------
The paper describes a multi-dimensional clinical reward (accuracy, format,
process, safety, coherence, assertion) and presents it as the objective the
agent is trained against.  Two facts about the released code make that claim
worth measuring rather than asserting:

  1. `scripts/verl/reward_fn.py:compute_score` -- the function verl actually
     calls during RL -- never calls `compute_composite_reward`.  The reward
     that trained every published run is
         accuracy + 0.1*format_bonus - 0.2*n_invalid_tool_calls
     with a degenerate-response sentinel, i.e. three terms, not six.
  2. The training parquet carries only
         correct_answer, domain, has_options, index, options, raw_answer,
         split, task_id
     so `nl_assertions` and `expected_actions` are absent.  Inside training the
     assertion dimension is pinned at its neutral 0.5 and the tool half of the
     process dimension has nothing to compare against.

This script answers "how much is each dimension individually worth?" by
measurement on stored rollouts rather than by algebra on assumed
distributions.

Why within-group variance is the right quantity
-----------------------------------------------
GRPO forms its advantage per prompt group (verl
`core_algos.compute_grpo_outcome_advantage`):

    A_i = (r_i - mean_g r) / std_g r

Only the *centered* reward enters.  So for a reward that is a weighted sum
r = sum_d w_d * s_d, the numerator is exactly

    r_i - mean_g r = sum_d w_d * (s_d,i - mean_g s_d)

and a dimension that is CONSTANT WITHIN A GROUP contributes exactly zero to
the gradient no matter how large its weight.  A dimension can therefore have a
substantial weight, a substantial mean, and still supply literally no learning
signal.  That is not an approximation; it is an identity.

We report the exact variance decomposition

    Var_g(r) = sum_d w_d * Cov_g(s_d, r)                      (sums to Var_g(r))
    share_d  = w_d * Cov_g(s_d, r) / Var_g(r)                 (sums to 1)

which is signed: a dimension anti-correlated with the total takes a negative
share, meaning it actively cancels signal from the others.

Two rewards are decomposed on the SAME rollouts:

  * `trained`   -- the reward that verl actually optimized, reconstructed from
                   the fields verl logged.  This one is self-checking: the
                   reconstruction is compared against the logged `score` for
                   every row and the run aborts if they disagree, so the
                   decomposition cannot silently drift from what ran.
  * `composite` -- the 6D reward the paper describes, recomputed from the
                   stored response text with the paper's default weights.

Usage
-----
    python scripts/rebuttal/decompose_reward_signal.py \
        --dumps /path/to/rollout_dumps --arms q4b_grpo q9b_grpo \
        --out /path/to/out.json

Reads nothing but the dumps; writes one JSON report.  No GPU, no network.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# The composite lives in the Healthcare_GYM package; the reward that trained
# lives in scripts/verl.  Make both importable regardless of cwd.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (str(_REPO), str(_REPO / "scripts" / "verl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bioagents.evaluation.rewards import (  # noqa: E402
    _compute_coherence_score,
    accuracy_reward_soft,
    format_reward_composite,
    process_reward_reasoning_quality,
    process_reward_tool_usage,
)

# Paper's default weights, copied from compute_composite_reward.  Kept as a
# literal rather than imported so that a change to the library default shows up
# here as a disagreement instead of silently redefining the measurement.
PAPER_WEIGHTS = {
    "accuracy": 0.25,
    "format": 0.10,
    "process": 0.20,
    "safety": 0.20,
    "coherence": 0.10,
    "assertion": 0.15,
}

# From reward_fn.py.  Same reasoning as above: literals, not imports.
INVALID_TOOL_PENALTY = 0.2
DEGENERATE_SENTINEL = -999.0
FORMAT_BONUS = 0.1
_ANSWER_RE = re.compile(r"Answer:\s*[A-E]")


TRAINED_DIMS = ["accuracy", "format", "tool_penalty", "degenerate"]


def trained_components(row: dict) -> dict[str, float] | None:
    """The reward verl actually optimized, split into its additive terms.

    The terms are constructed to sum EXACTLY to the score verl logged, which is
    what `main` then asserts row by row.  `compute_score` has two branches and
    they do not shape the reward the same way:

      * has_options (MCQA): accuracy is 0/1 and a +0.1 format bonus applies,
        but only to a correct answer that also emitted `Answer: X`.
      * open-ended: the reward is the token-F1 overlap, logged as `acc_partial`,
        and there is NO format bonus in this branch at all.

    Degenerate rollouts differ by branch too.  MCQA under DEGENERATE_EXCLUDE
    returns the -999 sentinel, which `core_algos` DROPS from the group
    statistics -- such a rollout influences no dimension and is returned as
    None here so it is excluded from the variance exactly as it is in training.
    The open-ended branch has never honoured that flag and returns a flat -1.0,
    which does enter the group statistics; it is kept, with the whole reward
    attributed to a `degenerate` term because the filter overrides every other
    dimension, making their marginal contribution genuinely zero on that row.
    """
    logged = float(row.get("score", 0.0))
    if logged <= DEGENERATE_SENTINEL + 1.0:
        return None  # core_algos drops the sentinel from the group statistics

    zero = dict.fromkeys(TRAINED_DIMS, 0.0)
    if row.get("degenerate", 0.0):
        return {**zero, "degenerate": logged}

    tool = -INVALID_TOOL_PENALTY * float(row.get("n_invalid_tool_calls", 0.0))
    if row.get("has_options", 0.0):
        acc = float(row.get("acc", 0.0))
        fmt = FORMAT_BONUS if (acc > 0 and _ANSWER_RE.search(row.get("output", ""))) else 0.0
        return {**zero, "accuracy": acc, "format": fmt, "tool_penalty": tool}
    # Open-ended: base reward is the raw token-F1, which verl logs as acc_partial.
    return {**zero, "accuracy": float(row.get("acc_partial", 0.0)), "tool_penalty": tool}


def composite_components(row: dict) -> dict[str, float]:
    """The paper's 6D reward, recomputed from the stored response.

    Assumptions, stated because they bound what this measurement means:
      * The dumped `output` is the full trajectory, so it is scored as the
        final turn (`is_final=True`).  turn_idx is taken from the tool-call
        count verl logged.  This matches how verl scores: the reward manager
        hands `compute_score` the whole `solution_str`, once, at the end.
        It also PINS the format dimension: `format_reward_composite` with
        is_final=True returns 1.0 for any response over 10 characters and 0.3
        otherwise, and every stored rollout clears 10 characters, so format is
        a literal constant here.  That is a property of the function, not of
        the sample -- but it is conditional on end-of-trajectory scoring.  Were
        the composite applied PER TURN, intermediate turns would take the
        `format_reward_tool_call` branch, which does vary.  The zero-signal
        result below is therefore a statement about the composite as a verl
        reward, not about the format reward in general.
      * `expected_actions` and `nl_assertions` are empty because the training
        parquet has no such columns -- this is the condition that actually
        holds inside training, not a simplification.
      * safety is evaluated by the same fallback the library uses when no
        patient context is supplied.  Training supplies none.
    """
    resp = row.get("output", "") or ""
    gt = row.get("gts", "") or ""
    n_tool = int(row.get("n_tool_calls", 0.0))

    accuracy = accuracy_reward_soft(resp, gt, "")
    fmt = format_reward_composite(resp, turn_idx=n_tool, is_final=True)
    tool_score = process_reward_tool_usage([], [])
    reasoning = process_reward_reasoning_quality(resp, gt)
    process = 0.5 * tool_score + 0.5 * reasoning

    try:
        from bioagents.evaluation.safety_eval import compute_safety_reward

        safety = compute_safety_reward(
            response=resp,
            task_domain="",
            patient_allergies=[],
            patient_conditions=[],
            emergency_type="",
        ).get("total", 1.0)
    except Exception:
        safety = 1.0

    coherence = _compute_coherence_score(resp, True)
    assertion = 0.5  # no nl_assertions in the training data -> library neutral

    return {
        "accuracy": float(accuracy),
        "format": float(fmt),
        "process": float(process),
        "safety": float(safety),
        "coherence": float(coherence),
        "assertion": float(assertion),
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def decompose(groups, dims, weights):
    """Exact within-group variance decomposition.

    groups: list of lists of per-dimension score dicts (one list per prompt).
    Returns per-dimension share of Var_g(r), plus diagnostics.

    Uses the population (1/n) covariance, matching what GRPO's centering does;
    the shares are scale-invariant so the choice does not affect them, but it
    does affect the reported absolute std, which we want to be the std GRPO
    divides by.
    """
    tot_var = 0.0
    contrib = dict.fromkeys(dims, 0.0)
    dim_std_sum = dict.fromkeys(dims, 0.0)
    dim_dead = dict.fromkeys(dims, 0)
    n_groups = 0
    n_dead_groups = 0

    for g in groups:
        if len(g) < 2:
            continue
        n_groups += 1
        n = len(g)
        means = {d: _mean([s[d] for s in g]) for d in dims}
        cen = {d: [s[d] - means[d] for s in g] for d in dims}
        totals = [sum(weights.get(d, 0.0) * cen[d][i] for d in dims) for i in range(n)]

        var = sum(t * t for t in totals) / n
        tot_var += var
        if var <= 1e-12:
            n_dead_groups += 1

        for d in dims:
            # w_d * Cov_g(s_d, r); summing over d reproduces Var_g(r) exactly.
            contrib[d] += weights.get(d, 0.0) * sum(cen[d][i] * totals[i] for i in range(n)) / n
            sd = math.sqrt(sum(c * c for c in cen[d]) / n)
            dim_std_sum[d] += sd
            if sd <= 1e-12:
                dim_dead[d] += 1

    out = {
        "n_groups": n_groups,
        "mean_group_std": math.sqrt(tot_var / n_groups) if n_groups else 0.0,
        "pct_groups_zero_signal": 100.0 * n_dead_groups / n_groups if n_groups else 0.0,
        "dims": {},
    }
    for d in dims:
        out["dims"][d] = {
            "weight": weights.get(d, 0.0),
            "share_of_variance_pct": (100.0 * contrib[d] / tot_var) if tot_var > 1e-12 else 0.0,
            "mean_within_group_std": dim_std_sum[d] / n_groups if n_groups else 0.0,
            "pct_groups_constant": 100.0 * dim_dead[d] / n_groups if n_groups else 0.0,
        }
    return out


def load_arm(dump_dir: Path, stride: int, max_steps: int):
    """Yield (step, [rows]) for sampled step files, newest-last."""
    files = sorted(dump_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    files = [f for f in files if f.stem.isdigit()]
    if stride > 1:
        files = files[::stride]
    if max_steps:
        files = files[-max_steps:]
    for f in files:
        rows = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if rows:
            yield int(f.stem), rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="rollout_dumps root")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1, help="sample every Nth step file")
    ap.add_argument("--max-steps", type=int, default=0, help="keep only the last N sampled files")
    args = ap.parse_args()

    report = {"paper_weights": PAPER_WEIGHTS, "arms": {}}
    root = Path(args.dumps)

    for arm in args.arms:
        d = root / arm
        if not d.is_dir():
            print(f"[skip] {arm}: no such directory", file=sys.stderr)
            continue

        trained_groups, comp_groups = [], []
        n_rows = n_degen = 0
        recon_checked = recon_bad = 0
        worst = 0.0

        for _step, rows in load_arm(d, args.stride, args.max_steps):
            by_prompt = defaultdict(list)
            for r in rows:
                by_prompt[r.get("input", "")].append(r)

            for _prompt, grp in by_prompt.items():
                t_scores, c_scores = [], []
                for r in grp:
                    n_rows += 1
                    tc = trained_components(r)
                    if tc is None:
                        n_degen += 1
                    else:
                        # Self-check: the reconstruction must reproduce the
                        # score verl logged, or this decomposition is measuring
                        # a reward that never ran.
                        if "score" in r:
                            recon_checked += 1
                            delta = abs(sum(tc.values()) - float(r["score"]))
                            worst = max(worst, delta)
                            if delta > 1e-6:
                                recon_bad += 1
                        t_scores.append(tc)
                    c_scores.append(composite_components(r))
                if len(t_scores) >= 2:
                    trained_groups.append(t_scores)
                if len(c_scores) >= 2:
                    comp_groups.append(c_scores)

        # The trained reward is already a plain sum of its terms, so every
        # weight here is 1.0: the weighting is baked into the terms themselves
        # (0.1 for the format bonus, -0.2 per invalid tool call).
        trained_w = dict.fromkeys(TRAINED_DIMS, 1.0)
        report["arms"][arm] = {
            "n_rollouts": n_rows,
            "n_sentinel_excluded": n_degen,
            "reconstruction_check": {
                "rows_checked": recon_checked,
                "rows_mismatched": recon_bad,
                "max_abs_delta": worst,
            },
            "trained_reward": decompose(trained_groups, TRAINED_DIMS, trained_w),
            "composite_reward": decompose(comp_groups, list(PAPER_WEIGHTS), PAPER_WEIGHTS),
        }

        rc = report["arms"][arm]["reconstruction_check"]
        status = "OK" if rc["rows_mismatched"] == 0 else f"MISMATCH x{rc['rows_mismatched']}"
        print(f"[{arm}] {n_rows} rollouts, {n_degen} degenerate, reconstruction {status}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out}")

    # Fail loudly if any arm's reconstruction disagreed with the logged score.
    bad = [a for a, v in report["arms"].items() if v["reconstruction_check"]["rows_mismatched"]]
    if bad:
        print(f"ERROR: reconstruction disagrees with logged score for {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
