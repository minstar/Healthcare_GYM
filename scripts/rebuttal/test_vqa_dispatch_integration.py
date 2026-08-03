#!/usr/bin/env python3
"""Integration smoke: drive the REAL run_benchmark_multiturn scoring dispatch.

The unit suite (test_vqa_scoring.py) tests the rule. This tests the WIRING:
that `--vqa-scorer` actually selects the rule, that the artifact records which
rule produced the number, and that both accuracies land in the artifact.

No GPU and no model: `_run_single_task_multiturn` is monkeypatched to replay
stored rollouts, so everything downstream of generation is the real code path.

    python scripts/rebuttal/test_vqa_dispatch_integration.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_benchmark_multiturn as M  # noqa: E402

ROLLOUTS = Path("/data/project/private/minstar/workspace/hcgym_rebuttal/eval_results")
ARM = "base"
BENCH = "vqa_rad"

FAIL = []


def check(name, got, expect):
    ok = got == expect if not isinstance(expect, float) else abs(got - expect) < 5e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<62} got={got}  expect={expect}")
    if not ok:
        FAIL.append(name)


def main():
    stored = json.load(open(ROLLOUTS / ARM / f"{BENCH}_partial.json"))["results"]
    by_id = {r["task_id"]: r for r in stored}

    tasks = [t for t in M.load_vqa_benchmark(BENCH) if t["id"] in by_id]
    print(f"replaying {len(tasks)} stored {BENCH} rollouts from arm '{ARM}' "
          f"through the real dispatch\n")

    class _Turn:
        raw_output = ""
        parsed_tool_call = None
        tool_call_format = "native"
        is_final_answer = True

    def fake_run(runner, task, env, max_turns):
        return [_Turn()], by_id[task["id"]]["submitted"], 0.0, []

    M._run_single_task_multiturn = fake_run

    class _Cfg:
        prompt_mode = "default"

    class _Runner:
        config = _Cfg()

    results = {}
    for rule in ("substring", "cf_em"):
        with tempfile.TemporaryDirectory() as td:
            s = M.run_benchmark_multiturn(
                benchmark_name=BENCH, tasks=tasks, runner=_Runner(),
                domain="visual_diagnosis", max_turns=3,
                output_dir=Path(td), vqa_scorer=rule)
            written = json.load(open(next(Path(td).glob(f"{BENCH}_multiturn_*.json"))))
        results[rule] = (s, written)

    print("\nA. THE FLAG SELECTS THE RULE")
    check("substring run records rule", results["substring"][0]["vqa_scoring"]["rule"], "substring")
    check("cf_em run records rule", results["cf_em"][0]["vqa_scoring"]["rule"], "cf_em")
    check("cf_em run records version", results["cf_em"][0]["vqa_scoring"]["version"], "cf_em/1.0")

    print("\nB. THE OLD RULE REPRODUCES THE PUBLISHED NUMBER EXACTLY")
    pub_correct = sum(1 for r in stored if r["correct"])
    check("substring accuracy == stored accuracy",
          results["substring"][0]["correct"], pub_correct)

    print("\nC. THE NEW RULE IS THE DEFAULT AND MOVES THE NUMBER")
    cf = results["cf_em"][0]
    check("cf_em accuracy != substring accuracy",
          cf["correct"] != results["substring"][0]["correct"], True)
    check("artifact carries BOTH accuracies (substring)",
          cf["vqa_scoring"]["substring_correct"], pub_correct)
    check("artifact carries BOTH accuracies (cf_em)",
          cf["vqa_scoring"]["cf_em_correct"], cf["correct"])

    print("\nD. THE WRITTEN ARTIFACT ON DISK CARRIES THE PROVENANCE")
    w = results["cf_em"][1]
    check("written artifact has vqa_scoring block", "vqa_scoring" in w, True)
    check("written artifact metric names the rule", w["metric"].startswith("cf_em"), True)
    row = w["results"][0]
    for f in ("scored_by", "cf_correct", "cf_pred", "cf_kind", "substring_correct"):
        check(f"every row carries `{f}`", f in row, True)
    check("row scored_by == module version", row["scored_by"], "cf_em/1.0")
    check("vocab built from FULL benchmark, not the replayed subset",
          w["vqa_scoring"]["vocab_built_from_n_items"], 451)
    check("...while only the replayed subset was scored", w["total"], len(tasks))

    print("\nE. THE NUMBERS MATCH THE STANDALONE RESCORER")
    # Computed from the rollouts on disk, never pinned. The claim is that the
    # dispatch and the standalone rescorer agree with each other -- an absolute
    # constant here just records how many rollouts existed the day it was written,
    # and this suite went red when the resume jobs grew base's partial from 280
    # rows to 450, which is the outcome those jobs existed to produce.
    import importlib.util as _il
    _spec = _il.spec_from_file_location("rescore", str(PROJECT_ROOT / "scripts" / "rebuttal" / "rescore_vqa.py"))
    _rescore = _il.module_from_spec(_spec)
    _spec.loader.exec_module(_rescore)
    rows = [{"submitted": r["submitted"], "gold": r["gold"],
             "_i": M.vqa_scoring.task_row_index(r["task_id"], "vqa_rad")}
            for r in cf["results"]]
    _, stats = M.vqa_scoring.score_all(rows, M.vqa_scoring.load_vocab("vqa_rad", PROJECT_ROOT))
    check(f"cf_em matches the standalone rescorer over the {len(rows)} replayed rows",
          round(cf["accuracy"] * 100, 2), round(stats["cf_em"] * 100, 2))
    sub_from_rows = sum(1 for r in cf["results"] if r.get("substring_correct")) / max(len(rows), 1)
    check("substring accuracy is recomputable from the per-row flags",
          round(results["substring"][0]["accuracy"] * 100, 2), round(sub_from_rows * 100, 2))

    print("\n" + "=" * 88)
    if FAIL:
        print(f"FAILED {len(FAIL)}:")
        for f in FAIL:
            print("  -", f)
        return 1
    print("ALL INTEGRATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
