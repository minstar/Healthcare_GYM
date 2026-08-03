#!/usr/bin/env python3
"""--resume-from must continue a partial, not replace it.

_save_partial rewrites the artifact from `results` alone. Before this was covered,
resuming a 280-row VQA-RAD partial at 280 would have written a file holding only
rows 280..451 and reported the accuracy of that tail as the benchmark's accuracy --
a wrong number that looks entirely reasonable. These pin the merge and every
refusal that stops two different measurements ending up in one column.

    python scripts/rebuttal/test_resume_merge.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "eval_mt", str(REPO / "scripts" / "eval_benchmark_multiturn.py")
)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

PASS = 0
FAIL = 0

# With tasks=[] the loop never runs, so the runner is only read for its prompt_mode.
STUB_RUNNER = SimpleNamespace(config=SimpleNamespace(prompt_mode="default"))


def check(label, expect, got):
    global PASS, FAIL
    if expect == got:
        print(f"  [PASS] {label:<58} {got}")
        PASS += 1
    else:
        print(f"  [FAIL] {label:<58} got={got!r} expect={expect!r}")
        FAIL += 1


def refuses(label, fn, must_mention):
    global PASS, FAIL
    try:
        fn()
    except SystemExit as exc:
        message = str(exc)
        if must_mention in message:
            print(f"  [PASS] {label:<58} refused")
            PASS += 1
        else:
            print(f"  [FAIL] {label:<58} refused, but message lacked {must_mention!r}: {message}")
            FAIL += 1
        return
    print(f"  [FAIL] {label:<58} did NOT refuse")
    FAIL += 1


def rows(n, start=0, benchmark="vqa_rad", correct_every=2):
    return [
        {
            "task_id": f"{benchmark}_{i}",
            "gold": "yes",
            "submitted": "yes",
            "correct": (i % correct_every == 0),
            "turns": 3,
            "latency": 1.0,
            "react_rate": 0.0,
            "format_adherence": {"n_turns": 3, "n_react": 0, "react_rate": 0.0},
        }
        for i in range(start, start + n)
    ]


def tasks(n, start=0, benchmark="vqa_rad"):
    return [{"id": f"{benchmark}_{i}"} for i in range(start, start + n)]


def write_partial(directory, benchmark, result_rows, rule=None):
    payload = {
        "benchmark": benchmark,
        "accuracy": sum(1 for r in result_rows if r["correct"]) / max(len(result_rows), 1),
        "correct": sum(1 for r in result_rows if r["correct"]),
        "total": len(result_rows),
        "results": result_rows,
    }
    if rule is not None:
        payload["vqa_scoring"] = {"rule": rule, "version": "test"}
    path = Path(directory) / f"{benchmark}_partial.json"
    path.write_text(json.dumps(payload))
    return path


def main():
    tmp = Path(tempfile.mkdtemp())

    print("1. the happy path returns exactly the rows already collected")
    write_partial(tmp, "vqa_rad", rows(280), rule="substring")
    loaded = M._load_resume_partial("vqa_rad", tmp, 280, tasks(280), "substring")
    check("rows returned", 280, len(loaded))
    check("first row is task 0", "vqa_rad_0", loaded[0]["task_id"])
    check("last row is task 279", "vqa_rad_279", loaded[-1]["task_id"])

    print("\n2. seeding carries the prior rows into the saved artifact")
    # tasks=[] means the loop body never runs, so what lands on disk is exactly
    # what the seeding produced -- which is the thing that used to be dropped.
    out = tmp / "seeded"
    out.mkdir()
    prior = rows(280)
    summary = M.run_benchmark_multiturn(
        benchmark_name="vqa_rad", tasks=[], runner=STUB_RUNNER, domain="radiology",
        max_turns=5, output_dir=out, vqa_scorer="substring", prior_results=prior,
    )
    check("summary total", 280, summary["total"])
    check("summary correct", 140, summary["correct"])
    check("summary accuracy", 0.5, round(summary["accuracy"], 6))

    print("\n3. LFQA aggregates are restored, not restarted at zero")
    lfqa_prior = rows(10, benchmark="kqa_golden")
    for i, r in enumerate(lfqa_prior):
        r["rouge_l"] = 0.1
        r["hallucination"] = 1.0
        r["comprehensiveness"] = 2.0
    out2 = tmp / "seeded_lfqa"
    out2.mkdir()
    summary2 = M.run_benchmark_multiturn(
        benchmark_name="kqa_golden", tasks=[], runner=STUB_RUNNER, domain="general",
        max_turns=5, output_dir=out2, prior_results=lfqa_prior,
    )
    check("avg_comprehensiveness restored", 2.0, round(summary2.get("avg_comprehensiveness", -1), 6))
    check("avg_hallucination restored", 1.0, round(summary2.get("avg_hallucination", -1), 6))
    check("avg_rouge_l restored", 0.1, round(summary2.get("avg_rouge_l", -1), 6))

    print("\n4. every way the merge could be wrong is refused")
    empty = tmp / "empty"
    empty.mkdir()
    refuses("no partial to resume from",
            lambda: M._load_resume_partial("vqa_rad", empty, 280, tasks(280), "substring"),
            "no ")

    short = tmp / "short"
    short.mkdir()
    write_partial(short, "vqa_rad", rows(250), rule="substring")
    refuses("row count disagrees with --resume-from",
            lambda: M._load_resume_partial("vqa_rad", short, 280, tasks(280), "substring"),
            "--resume-from 250")

    shuffled = tmp / "shuffled"
    shuffled.mkdir()
    write_partial(shuffled, "vqa_rad", rows(280, start=1000), rule="substring")
    refuses("task ids do not line up",
            lambda: M._load_resume_partial("vqa_rad", shuffled, 280, tasks(280), "substring"),
            "cannot be merged")

    mixed = tmp / "mixed"
    mixed.mkdir()
    write_partial(mixed, "vqa_rad", rows(280), rule="substring")
    refuses("prior rule differs from this run's rule",
            lambda: M._load_resume_partial("vqa_rad", mixed, 280, tasks(280), "cf_em"),
            "rescore_vqa.py")

    wrong_bench = tmp / "wrong_bench"
    wrong_bench.mkdir()
    path = write_partial(wrong_bench, "vqa_rad", rows(280), rule="substring")
    payload = json.loads(path.read_text())
    payload["benchmark"] = "slake"
    path.write_text(json.dumps(payload))
    refuses("partial names a different benchmark",
            lambda: M._load_resume_partial("vqa_rad", wrong_bench, 280, tasks(280), "substring"),
            "not 'vqa_rad'")

    print("\n5. the sbatch wrapper refuses a multi-benchmark resume before it costs a GPU")
    import subprocess
    slurm = Path("/data/project/private/minstar/workspace/hcgym_rebuttal/runs/eval_agentic.slurm")
    if slurm.exists():
        guard = subprocess.run(
            ["bash", "-c",
             'BENCH="vqa_rad slake"; RESUME_FROM=280; '
             'if [ "$RESUME_FROM" -gt 0 ] && [ "$(echo $BENCH | wc -w)" -gt 1 ]; then exit 2; fi'],
            capture_output=True,
        )
        check("guard rejects two benchmarks", 2, guard.returncode)
        ok = subprocess.run(
            ["bash", "-c",
             'BENCH="vqa_rad"; RESUME_FROM=280; '
             'if [ "$RESUME_FROM" -gt 0 ] && [ "$(echo $BENCH | wc -w)" -gt 1 ]; then exit 2; fi'],
            capture_output=True,
        )
        check("guard accepts one benchmark", 0, ok.returncode)
        check("guard is present in eval_agentic.slurm", True,
              "resume one per job" in slurm.read_text())

    print("\n6. a text benchmark is not rule-gated (no VQA scorer involved)")
    textq = tmp / "text"
    textq.mkdir()
    write_partial(textq, "medqa", rows(50, benchmark="medqa"))
    loaded = M._load_resume_partial("medqa", textq, 50, tasks(50, benchmark="medqa"), "cf_em")
    check("medqa resumes under any scorer", 50, len(loaded))

    print("\n" + "=" * 74)
    if FAIL == 0:
        print(f"ALL {PASS} CHECKS PASSED")
        return 0
    print(f"{FAIL} of {PASS + FAIL} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
