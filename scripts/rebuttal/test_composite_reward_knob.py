#!/usr/bin/env python3
"""Behavioural tests for the HCGYM_REWARD_WEIGHTS training knob.

`reward_fn` reads its configuration at import time, so each case runs in a
fresh subprocess with its own environment.  That is deliberate: importlib.reload
would leave module-level constants from a previous case behind and a test could
pass for the wrong reason.

Run:  python scripts/rebuttal/test_composite_reward_knob.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# A multiple-choice rollout that is CORRECT and well-formatted, so the legacy
# path gives accuracy 1.0 + the 0.1 format bonus and every composite dimension
# has something to look at.
MC_SOLUTION = (
    "<think>The patient presents with chest pain radiating to the left arm. "
    "Considering the differential, an acute coronary syndrome is most likely "
    "given the risk factors and ECG changes.</think>\n"
    "Based on the evidence gathered, the correct choice is D.\n"
    "Answer: D"
)

DRIVER = r"""
import json, os, sys
sys.path.insert(0, {repo!r})
sys.path.insert(0, {verl!r})
import reward_fn
out = reward_fn.compute_score(
    data_source="bioagents_medical",
    solution_str={sol!r},
    ground_truth="D",
    extra_info={{"has_options": True, "domain": "cardiology", "raw_answer": "acute coronary syndrome"}},
)
print("RESULT " + json.dumps(out))
"""


def run(env_extra: dict, solution: str = MC_SOLUTION):
    """Score one rollout in a subprocess. Returns (rc, payload|None, stderr)."""
    env = dict(os.environ)
    # Neutralise any inherited arm configuration so a case tests only its own.
    for k in ("HCGYM_REWARD_WEIGHTS", "COSINE_REWARD", "DEGENERATE_FILTER", "DEGENERATE_EXCLUDE"):
        env.pop(k, None)
    env.update(env_extra)
    env["PYTHONNOUSERSITE"] = "1"
    code = DRIVER.format(repo=str(REPO), verl=str(REPO / "scripts" / "verl"), sol=solution)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    payload = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
    return p.returncode, payload, p.stderr


FULL = {"accuracy": 0.25, "format": 0.10, "process": 0.20, "safety": 0.20, "coherence": 0.10, "assertion": 0.15}

failures: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# 1. Default (knob unset) must be byte-identical to the legacy reward: a correct,
#    well-formatted MCQA answer scores accuracy 1.0 + the 0.1 format bonus.
rc, base, err = run({})
check("knob unset -> import succeeds", rc == 0 and base is not None, err[-400:])
if base:
    check("knob unset -> legacy score 1.1", abs(base["score"] - 1.1) < 1e-9, f"got {base['score']}")

# 2. With the knob set the score becomes the composite total, which is bounded
#    by the weights and so cannot still be 1.1.
rc, comp, err = run({"HCGYM_REWARD_WEIGHTS": json.dumps(FULL)})
check("full composite -> runs", rc == 0 and comp is not None, err[-400:])
if comp and base:
    check("full composite -> score differs from legacy", abs(comp["score"] - base["score"]) > 1e-6,
          f"legacy {base['score']} vs composite {comp['score']}")
    check("full composite -> score within sum of weights", 0.0 <= comp["score"] <= sum(FULL.values()) + 1e-9,
          f"got {comp['score']}")
    # 3. The cross-arm readout must not move when the weights do.  This is the
    #    property the whole ablation rests on: arms are compared on `acc`.
    check("composite leaves acc unshaped", comp["acc"] == base["acc"], f"{comp['acc']} vs {base['acc']}")
    check("composite leaves acc_partial unshaped", comp["acc_partial"] == base["acc_partial"])

# 4. Leave-one-out: dropping a dimension must move the score by exactly that
#    dimension's weighted contribution, and dropping an INERT dimension must not
#    move it at all.  `format` is inert by construction here -- with is_final=True
#    format_reward_composite returns 1.0 for any response over 10 characters --
#    so leaving it out shifts the score by exactly its weight, uniformly across
#    every rollout, which GRPO's per-group centering then removes entirely.
loo = {k: v for k, v in FULL.items() if k != "format"}
rc, no_fmt, err = run({"HCGYM_REWARD_WEIGHTS": json.dumps(loo)})
check("leave-one-out format -> runs", rc == 0 and no_fmt is not None, err[-400:])
if no_fmt and comp:
    delta = comp["score"] - no_fmt["score"]
    check("dropping format shifts score by exactly its weight (1.0 * 0.10)",
          abs(delta - 0.10) < 1e-9, f"delta {delta}")

# 5. accuracy-only arm.
rc, acc_only, err = run({"HCGYM_REWARD_WEIGHTS": json.dumps({"accuracy": 1.0})})
check("accuracy-only -> runs", rc == 0 and acc_only is not None, err[-400:])
if acc_only:
    check("accuracy-only -> score in [0,1]", 0.0 <= acc_only["score"] <= 1.0, f"got {acc_only['score']}")

# 6. Misconfiguration must fail loudly at import, never silently define a
#    different arm than the one the launcher named.
rc, _, err = run({"HCGYM_REWARD_WEIGHTS": json.dumps({"acuracy": 1.0})})
check("typo'd dimension -> raises", rc != 0 and "unknown dimensions" in err, err[-200:])

rc, _, err = run({"HCGYM_REWARD_WEIGHTS": "{not json"})
check("malformed JSON -> raises", rc != 0 and "not valid JSON" in err, err[-200:])

rc, _, err = run({"HCGYM_REWARD_WEIGHTS": json.dumps({"accuracy": True})})
check("boolean weight -> raises", rc != 0 and "must be numbers" in err, err[-200:])

rc, _, err = run({"HCGYM_REWARD_WEIGHTS": json.dumps(FULL), "COSINE_REWARD": "1"})
check("composite + cosine -> raises", rc != 0 and "mutually exclusive" in err, err[-200:])

# 7. An empty string is "not configured", not "empty config" -- `sbatch
#    --export=ALL` propagates unset-but-declared variables as empty strings, and
#    that must land on the legacy path rather than error.
rc, blank, err = run({"HCGYM_REWARD_WEIGHTS": ""})
check("empty string -> legacy path", rc == 0 and blank is not None and abs(blank["score"] - 1.1) < 1e-9,
      f"rc={rc} {err[-200:]}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all composite-knob tests passed")
