#!/usr/bin/env python3
"""Pin the STaR rollout filename rule.

Six of the 4,777 task ids in the pool carry a path separator
(``tri_bal_overdose/ingestion_1_092``). Interpolating the id straight into a
filename produced a path into a directory that was never created:

    FileNotFoundError: .../rollouts/task_tri_bal_overdose/ingestion_1_092__k0.json

Generation died at the write, after every rollout had already been sampled, and
the retry loop re-ran the whole iteration into the same wall -- 45 of the arm's
60 attempts.

The interesting property is not that the separator is gone; it is that the fix
does not trade a loud crash for a silent one. Under naive replacement ``a/b`` and
``a_b`` both become ``a_b`` and one rollout would quietly overwrite the other, so
uniqueness over the real pool is what this test spends most of its assertions on.

Run:  python scripts/rebuttal/test_star_rollout_filename.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("_star_generate", REPO / "scripts/rebuttal/star_generate.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_star_generate"] = _mod
try:
    _spec.loader.exec_module(_mod)
except SystemExit:  # the module has a __main__ guard that argparses
    pass

rollout_filename = _mod._rollout_filename

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def rec(task_id: str, k: int = 0) -> dict:
    return {"task_id": task_id, "star": {"sample_idx": k}}


# The id that actually broke the arm.
REAL = "tri_bal_overdose/ingestion_1_092"
name = rollout_filename(rec(REAL))
check("separator id yields a flat filename", "/" not in name and "\\" not in name, name)

tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / name).write_text("x")
check("and the write actually succeeds", (tmp / name).exists(), name)

# Uniqueness: the property a naive replacement would lose.
check("a/b does not collide with a_b",
      rollout_filename(rec("a/b")) != rollout_filename(rec("a_b")),
      f"{rollout_filename(rec('a/b'))} vs {rollout_filename(rec('a_b'))}")
check("two different separator ids stay distinct",
      rollout_filename(rec("x/y")) != rollout_filename(rec("x/z")))
check("sample_idx still distinguishes samples of one task",
      rollout_filename(rec("a/b", 0)) != rollout_filename(rec("a/b", 1)))

# Already-safe ids must keep their exact historical filename, so nothing on disk
# is renamed by this change.
check("safe id keeps its historical filename",
      rollout_filename(rec("medqa_00123")) == "task_medqa_00123__k0.json",
      rollout_filename(rec("medqa_00123")))

# Hostile inputs.
check("path traversal is neutralised", ".." not in rollout_filename(rec("../../etc/passwd")),
      rollout_filename(rec("../../etc/passwd")))
check("empty id still produces a filename", rollout_filename(rec("")).startswith("task_task-"),
      rollout_filename(rec("")))
check("a 400-char id fits inside the 255-byte limit",
      len(rollout_filename(rec("z" * 400)).encode()) < 255,
      str(len(rollout_filename(rec("z" * 400)))))

# The whole pool, not a sample: every id must map to its own filename.
pool = json.loads((REPO / "data/domains/full_4modality_clean/tasks.json").read_text())
seen: dict[str, str] = {}
collision = None
for t in pool:
    n = rollout_filename(rec(t["id"]))
    if n in seen and seen[n] != t["id"]:
        collision = (seen[n], t["id"], n)
        break
    seen[n] = t["id"]
check(f"all {len(pool)} pool ids map to unique filenames", collision is None, str(collision))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("star rollout filename: all checks passed")
