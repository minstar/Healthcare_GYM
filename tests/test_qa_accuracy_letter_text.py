"""`_compute_qa_accuracy` must credit a correct MULTIPLE-CHOICE LETTER.

The benchmark loader stores `answer` as the option TEXT while models answer with
the LETTER, so every multiple-choice task took the free-text branch and scored
0.0 for a correct letter. Two baselines are built on this function and both were
silently destroyed by it:

  * the Reflexion ladder reported solve_rate 0.0% on all 400 medqa tasks and all
    four strategies, against 83.1% for the same model through the eval harness
    (job 61473, eight GPU-hours);
  * STaR's default acceptance signal is this function, so it would have accepted
    no trajectories at all -- a vacuous baseline rather than a visibly broken one.

    python tests/test_qa_accuracy_letter_text.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bioagents.evaluation.agent_runner import AgentRunner  # noqa: E402

PASS = 0
FAIL = 0


def check(label, expect, got):
    global PASS, FAIL
    if expect == got:
        print(f"  [PASS] {label:<56} {got}")
        PASS += 1
    else:
        print(f"  [FAIL] {label:<56} got={got!r} expect={expect!r}")
        FAIL += 1


def scorer():
    # The method is pure over (task, tool_call_log); no model or GPU is needed.
    return AgentRunner.__new__(AgentRunner)


def log(answer):
    return [{"tool_name": "submit_answer", "arguments": {"answer": answer}}]


MC_TASK = {
    "answer": "Tell the attending that he cannot fail to disclose this mistake",
    "correct_answer": "Tell the attending that he cannot fail to disclose this mistake",
    "options": {
        "A": "Disclose the error to the patient and put it in the operative report",
        "B": "Tell the attending that he cannot fail to disclose this mistake",
        "C": "Report the physician to the ethics committee",
        "D": "Refuse to dictate the operative report",
    },
}


def main():
    r = scorer()

    print("1. the defect: a correct LETTER against a TEXT gold")
    check("correct letter scores 1.0", 1.0, r._compute_qa_accuracy(MC_TASK, log("B")))
    check("lowercase letter too", 1.0, r._compute_qa_accuracy(MC_TASK, log("b")))
    check("correct option text still scores 1.0", 1.0,
          r._compute_qa_accuracy(MC_TASK, log(MC_TASK["answer"])))

    print("\n2. wrong answers stay wrong — the fix must not just say yes")
    for bad in ["A", "C", "D", "E", "", "the attending"]:
        check(f"wrong/no answer {bad!r} scores 0.0", 0.0,
              r._compute_qa_accuracy(MC_TASK, log(bad)))

    print("\n3. a LETTER gold with letter answers (the other direction)")
    letter_task = {"answer": "C", "options": MC_TASK["options"]}
    check("matching letter", 1.0, r._compute_qa_accuracy(letter_task, log("C")))
    check("that letter's text", 1.0,
          r._compute_qa_accuracy(letter_task, log(MC_TASK["options"]["C"])))
    check("a different letter", 0.0, r._compute_qa_accuracy(letter_task, log("A")))

    print("\n4. tasks with no options are unaffected (free-text path)")
    free = {"answer": "acute pancreatitis"}
    check("exact free-text match", 1.0, r._compute_qa_accuracy(free, log("acute pancreatitis")))
    check("substring gets partial credit", 0.5,
          r._compute_qa_accuracy(free, log("likely acute pancreatitis, admit")))
    check("unrelated text", 0.0, r._compute_qa_accuracy(free, log("appendicitis")))

    print("\n5. degenerate inputs do not raise")
    check("options is null", 0.0, r._compute_qa_accuracy({"answer": "B", "options": None}, log("A")))
    check("no gold at all", 0.0, r._compute_qa_accuracy({}, log("A")))
    check("no submit_answer call", 0.0, r._compute_qa_accuracy(MC_TASK, []))

    print("\n6. against the real benchmark, end to end")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_mt", str(REPO / "scripts" / "eval_benchmark_multiturn.py"))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    tasks = M.load_textqa_benchmark("medqa")[:200]
    hit = miss = n = 0
    for t in tasks:
        opts = t.get("options") or {}
        letter = next((k for k, v in opts.items() if v == t.get("answer")), None)
        if letter is None:
            continue
        n += 1
        hit += r._compute_qa_accuracy(t, log(letter))
        other = next((k for k in opts if k != letter), None)
        miss += r._compute_qa_accuracy(t, log(other))
    check(f"correct letter credited on all {n} medqa tasks", float(n), float(hit))
    check("wrong letter credited on none", 0.0, float(miss))

    print("\n" + "=" * 74)
    if FAIL == 0:
        print(f"ALL {PASS} CHECKS PASSED")
        return 0
    print(f"{FAIL} of {PASS + FAIL} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
