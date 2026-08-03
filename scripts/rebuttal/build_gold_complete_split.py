#!/usr/bin/env python3
"""Build a training pool in which every TRAIN task has a gradeable gold answer.

The problem
-----------
``scripts/verl/convert_tasks_to_parquet.py`` writes

    "ground_truth": correct_answer if has_options else raw_answer

so an OPEN-ENDED task whose answer lives in ``correct_answer`` gets an empty
ground_truth.  ``raw_answer`` is not merely blank on those tasks -- the key is
absent entirely and the empty string comes from the ``.get(..., "")`` default.
``reward_fn.compute_score`` then takes

    if not ground_truth or not solution_str:
        base_reward = 0.0
        is_correct = False

so every rollout on such a prompt scores identically.  GRPO centres its
advantage per prompt group, so an identical score across the group means an
advantage of exactly zero: the prompt costs a full rollout budget and returns no
gradient.  Measured on stored rollouts, 98.0% of these groups are dead, and they
are 45.9% of the train split (1,556 / 3,390) -- about half of all training
compute produced nothing.

What this script does
---------------------
Two policies, both of which leave TEST untouched:

  --policy drop      remove all 1,556 gold-less train tasks.  This is the clean
                     intervention: nothing is added, nothing is rewritten, the
                     only change is that unscoreable prompts stop consuming
                     rollouts.

  --policy recover   additionally repair the tasks whose ``correct_answer`` is a
                     usable short answer, materialising the cleaned gold into
                     ``raw_answer`` so the existing converter picks it up with no
                     change to its mapping.  Strictly a superset of `drop`.

TEST IS NEVER FILTERED, on purpose.  Its 495 gold-less rows cap val accuracy at
355/850 = 0.4176, and every published curve was measured against that cap.
Removing them would silently redefine the metric mid-study and make the new runs
incomparable with the five arms already on disk.  The cap is a reporting matter
(divide by 0.4176), not something to fix by changing the yardstick.

Why most of the gold is NOT recoverable
---------------------------------------
Of the 1,556, only three families carry an answer that can serve as a token-F1
target.  The rest are rejected, and each rejection was checked rather than
assumed:

  * 328 have an empty ``correct_answer`` -- nothing to recover.
  * 569 ``*_evid_*`` tasks hold a TRUNCATED PubMed passage, median 128 words
    (p90 128, max 256).  As an F1 target a 128-word gold makes precision
    essentially unreachable; only the short ones are kept.
  * 167 ``ehr_*`` golds all literally begin ``Task type: `` and restate the
    prompt in 5-7 words.  They are tautologies, not answers.

Verified counts on full_4modality_clean (train), by family:

    family    total   correct_answer empty
    dx           70        70
    ehr         209        42
    evid        569         0
    labeled     342        52
    triage      366       164

Usage
-----
    python scripts/rebuttal/build_gold_complete_split.py --policy drop \
        --pool data/domains/full_4modality_clean \
        --out  data/domains/full_4modality_gold_drop

    python scripts/rebuttal/build_gold_complete_split.py --policy recover \
        --out data/domains/full_4modality_gold_recover

Deterministic and idempotent: no timestamps, no randomness, input order
preserved, and every dropped/repaired id is recorded in MANIFEST.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Cleaning rule for a recovered gold ────────────────────────────────────────
# Order matters and each step has a reason; see clean_gold().

# Self-RAG control tokens. Replaced with a SPACE, never with "": they are glued
# to real content ("Retrieval]Diagnosis:", "cirrhosis.[Utility:4]"), and
# reward_fn tokenises with a bare .lower().split() that never strips
# punctuation, so deleting them outright would fuse two content words into one
# token that the model can never produce.
_CONTROL_RE = re.compile(
    r"\[(?:No\s+Retrieval|Retrieval|Continue\s+to\s+Use\s+Evidence|Irrelevant|Relevant"
    r"|Partially\s+supported|Fully\s+supported|No\s+support[^\]]*|Utility:\d)\]",
    re.IGNORECASE,
)

# Angle-bracketed type tags: <Diagnosis>, <ICD Code>, <Interacting Drug>.
_TYPETAG_RE = re.compile(r"<[A-Z][A-Za-z ]{0,30}>")

# Template scaffold keys. The labeled families are two templates, so the
# scaffold word appears in most golds of the family; leaving it in would hand
# every rollout free F1 for typing "Diagnosis:".
_SCAFFOLD_RE = re.compile(
    r"\b(?:Diagnosis|Updated diagnosis|Key assessments|Key management"
    r"|First-line treatment|Interaction|Severity|Management)\s*:",
    re.IGNORECASE,
)

_PUNCT_RE = re.compile(r"[^\w\s-]")
_WS_RE = re.compile(r"\s+")

# A gold longer than this cannot function as an F1 target: precision is bounded
# by |gold| and the answer span is a sentence or two.
MAX_GOLD_WORDS = 40


def clean_gold(raw: str) -> str:
    """Normalise a recovered gold into a token-F1 target.

    The final lowercase/punctuation pass is not cosmetic: reward_fn scores with
    ``set(ground_truth.lower().split())`` against ``set(answer_span.lower().split())``
    and strips no punctuation, so a gold token "nervosa." can never match a
    predicted "nervosa".
    """
    g = _CONTROL_RE.sub(" ", raw)
    g = _TYPETAG_RE.sub(" ", g)
    g = _SCAFFOLD_RE.sub(" ", g)
    g = _PUNCT_RE.sub(" ", g)
    return _WS_RE.sub(" ", g).strip().lower()


def family(task_id: str) -> str:
    """Which generator produced this task. Decides recoverability."""
    if "_evid_" in task_id:
        return "evid"
    if task_id.startswith(("tri_", "triage_", "scaled_")):
        return "triage"
    if task_id.startswith(("ob_", "di_", "psy_", "psych_")):
        return "labeled"
    if task_id.startswith("ehr"):
        return "ehr"
    if task_id.startswith("dx_"):
        return "dx"
    return "other"


def is_gold_less(task: dict) -> bool:
    """True when the converter would emit an empty ground_truth for this task."""
    if task.get("options"):
        return False  # multiple choice: gold comes from correct_answer, which is present
    return not (task.get("raw_answer") or "").strip()


def recover(task: dict) -> tuple[str | None, str]:
    """Return (cleaned_gold, reason). cleaned_gold is None when unrecoverable."""
    raw = (task.get("correct_answer") or "").strip()
    if not raw:
        return None, "no correct_answer"

    fam = family(task["id"])
    if fam in ("ehr", "dx", "other"):
        # ehr golds are all "Task type: ..." restatements of the prompt; dx and
        # other have no non-empty correct_answer in this pool at all.
        return None, f"family {fam} carries no answer"

    cleaned = clean_gold(raw)
    if not cleaned:
        return None, "empty after cleaning"
    n = len(cleaned.split())
    if n > MAX_GOLD_WORDS:
        # Overwhelmingly the evid family: a truncated passage, not an answer.
        # Bucketed, not reported per exact length -- the manifest is a summary and
        # 70 single-count reasons hide the one number that matters.
        return None, f"too long (>{MAX_GOLD_WORDS} words)"
    return cleaned, f"recovered from {fam}"


def build(pool: Path, out: Path, policy: str, dry_run: bool) -> dict:
    tasks = json.loads((pool / "tasks.json").read_text())
    splits = json.loads((pool / "split_tasks.json").read_text())
    by_id = {t["id"]: t for t in tasks}

    train_ids = splits["train"]
    kept, dropped, repaired = [], [], []
    reasons = collections.Counter()
    new_by_id = {tid: dict(t) for tid, t in by_id.items()}

    # Dedup key for repaired rows only. Two generators emit byte-identical
    # (prompt, gold) pairs under different id prefixes (scaled_* vs triage_*);
    # keeping both would double-weight those items. Untouched tasks are left
    # exactly as the input pool had them -- deduplicating them here would be a
    # second, unrelated intervention.
    seen_repaired: dict[tuple[str, str], str] = {}

    for tid in train_ids:
        task = by_id.get(tid)
        if task is None:
            dropped.append(tid)
            reasons["missing from tasks.json"] += 1
            continue

        if not is_gold_less(task):
            kept.append(tid)
            continue

        if policy == "drop":
            dropped.append(tid)
            reasons["gold-less (policy=drop)"] += 1
            continue

        cleaned, why = recover(task)
        if cleaned is None:
            dropped.append(tid)
            reasons[why] += 1
            continue

        key = ((task.get("ticket") or task.get("raw_question") or "").strip(), cleaned)
        if key in seen_repaired:
            dropped.append(tid)
            reasons["duplicate of a repaired task"] += 1
            continue
        seen_repaired[key] = tid

        # Materialise into raw_answer so the EXISTING converter picks it up with
        # no change to its mapping. correct_answer is left untouched as the audit
        # trail, and gold_source records the provenance so metrics can be sliced
        # by it later.
        t = new_by_id[tid]
        t["raw_answer"] = cleaned
        t["gold_source"] = "correct_answer_recovered"
        kept.append(tid)
        repaired.append(tid)
        reasons[why] += 1

    # TEST is copied through verbatim. See the module docstring: filtering it
    # would redefine the validation metric and break comparison with every run
    # already on disk.
    new_splits = {k: (kept if k == "train" else list(v)) for k, v in splits.items()}

    kept_ids = set(kept) | {i for k, v in splits.items() if k != "train" for i in v}
    new_tasks = [new_by_id[t["id"]] for t in tasks if t["id"] in kept_ids]

    manifest = {
        "source_pool": str(pool),
        "policy": policy,
        "train_in": len(train_ids),
        "train_out": len(kept),
        "train_dropped": len(dropped),
        "train_repaired": len(repaired),
        "test_unchanged": len(splits.get("test", [])),
        "drop_reasons": dict(sorted(reasons.items())),
        "repaired_ids": sorted(repaired),
        "dropped_ids": sorted(dropped),
    }

    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "tasks.json").write_text(json.dumps(new_tasks, ensure_ascii=False, indent=1))
        (out / "split_tasks.json").write_text(json.dumps(new_splits, ensure_ascii=False, indent=1))
        (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(PROJECT_ROOT / "data/domains/full_4modality_clean"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", choices=("drop", "recover"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = build(Path(args.pool), Path(args.out), args.policy, args.dry_run)

    print(f"policy={m['policy']}  train {m['train_in']} -> {m['train_out']} "
          f"(dropped {m['train_dropped']}, repaired {m['train_repaired']})")
    print(f"test unchanged: {m['test_unchanged']}")
    for k, v in m["drop_reasons"].items():
        print(f"   {k:<40} {v}")

    # A pool that still contains an ungradeable train task defeats the purpose.
    pool_tasks = {t["id"]: t for t in json.loads((Path(args.pool) / "tasks.json").read_text())}
    residual = 0
    for tid in json.loads((Path(args.pool) / "split_tasks.json").read_text())["train"]:
        if tid in set(m["repaired_ids"]):
            continue
        if tid in set(m["dropped_ids"]):
            continue
        t = pool_tasks.get(tid)
        if t is not None and is_gold_less(t):
            residual += 1
    if residual:
        print(f"ERROR: {residual} train tasks kept without a gold answer")
        return 1
    print("ok: every kept train task has a gold answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
