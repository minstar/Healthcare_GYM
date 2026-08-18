#!/usr/bin/env python3
"""Pin the answer-type coercion in the multi-turn benchmark scorer.

`submit_answer`'s `answer` field is written by the model and nothing upstream
constrains its JSON type, so `submit_answer({"answer": 2})` arrives as an int.
Both scorers assume text -- `_check_answer` calls `.strip()`, `_compute_rouge_l`
calls `.lower()` via rouge_scorer -- and the resulting AttributeError is caught
by the per-task handler, which records the task as INCORRECT with turns=0. A
model that answered correctly is scored wrong, silently, and the zero pollutes
both the accuracy numerator and avg_turns.

It does not cancel across conditions. Over the stored result files, on the same
1061-item slake set, base_strong_tool hit it 15 times and base_react zero -- up
to ~1.4 pp of arm-specific loss, in the direction of whichever arm happens to
emit bare JSON numbers.

The fix coerces once, where the model's answer enters the harness. What this
file checks is that the coercion is actually THERE (by reading the call site)
and that it is SUFFICIENT (by running both scorers on the coerced value).

Run:  python scripts/rebuttal/test_answer_coercion.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGET = REPO / "scripts/eval_benchmark_multiturn.py"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ── 1. the call site itself coerces ──────────────────────────────────────────
# Read it rather than reaching it: getting to line 439 needs a live sglang
# server, an environment and a model. The property under test is textual.
src = TARGET.read_text()
site = re.search(r'submitted_answer\s*=\s*(.+?)\n', src)
check("found the submit_answer assignment", site is not None)
if site:
    expr = site.group(1)
    # The FIRST assignment in the file is the `= ""` initialiser; find the one
    # that reads the tool call.
    tool_sites = [m.group(1) for m in re.finditer(r'submitted_answer\s*=\s*(.*tool_call.*)\n', src)]
    check("exactly one site reads the model's answer", len(tool_sites) == 1, str(tool_sites))
    if tool_sites:
        check("and it wraps the value in str()", tool_sites[0].startswith("str("), tool_sites[0])

# ── 2. the coerced value survives both scorers ───────────────────────────────
spec = importlib.util.spec_from_file_location("_eval_mt", TARGET)
mod = importlib.util.module_from_spec(spec)
sys.modules["_eval_mt"] = mod
try:
    spec.loader.exec_module(mod)
    loaded = True
except Exception as e:  # noqa: BLE001 -- heavy optional deps may be absent
    loaded = False
    print(f"NOTE  could not import the module ({type(e).__name__}: {str(e)[:80]});"
          " skipping the scorer checks, the textual check above still ran")

if loaded:
    _check_answer = mod._check_answer
    _rouge = mod._compute_rouge_l

    # The exact shape that used to crash: an int answer against a string gold.
    check("int answer, coerced, scores correct",
          _check_answer(str(2), "2", {}) is True)
    check("int answer, coerced, scores incorrect when it should",
          _check_answer(str(3), "2", {}) is False)

    # And the raw int still raises -- which is what shows the coercion is doing
    # the work, rather than the scorers having been tolerant all along.
    try:
        _check_answer(2, "2", {})
        check("raw int still raises in _check_answer", False, "it did NOT raise")
    except AttributeError:
        check("raw int still raises in _check_answer", True)

    # LFQA path.
    check("coerced value works through ROUGE-L",
          _rouge(str(2), "2 mg daily") > 0.0)
    try:
        _rouge(2, "2 mg daily")
        check("raw int still raises in _compute_rouge_l", False, "it did NOT raise")
    except AttributeError:
        check("raw int still raises in _compute_rouge_l", True)

    # A float and a bool are the other JSON scalars a model can emit here.
    for val in (2.0, True):
        try:
            _check_answer(str(val), "2", {})
            check(f"{type(val).__name__} answer does not crash once coerced", True)
        except Exception as e:  # noqa: BLE001
            check(f"{type(val).__name__} answer does not crash once coerced", False, repr(e))

    # Empty and None must still be falsy, not the string "None".
    check("missing answer stays empty, not 'None'", _check_answer(str(""), "2", {}) is False)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("answer coercion: all checks passed")
