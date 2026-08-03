#!/usr/bin/env python3
"""Repoint `_image_path` in a task pool at the restored image directory.

The pool's image paths are absolute under a CSI volume
(`/mnt/aiplatform/csi-volumes/pvc-.../BIOAgents/datasets/medical_images`) that is no
longer mounted. `restore_data.sh D_train` rebuilds those exact files from HuggingFace,
by split index, under `datasets/medical_images/` inside this repo.

Rewriting the paths is preferable to recreating the dead mount with a symlink: it needs
no root, it survives being moved to another machine, and the pool then states plainly
where its images actually are.

Refuses to write unless every rewritten path resolves on disk, so a pool can never end
up half-remapped — a task whose image silently fails to load trains on the text alone
and quietly stops being a multimodal example.

    python scripts/rebuttal/remap_image_paths.py \
        --pool data/domains/full_4modality_clean [--dry-run]
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LIVE = REPO / "datasets" / "medical_images"
# Everything up to and including the directory that holds the per-dataset image trees.
DEAD_MARKER = "/datasets/medical_images/"


def remap(pool_dir: Path, live_root: Path, dry_run: bool) -> int:
    tasks_path = pool_dir / "tasks.json"
    if not tasks_path.exists():
        print(f"[fatal] no tasks.json under {pool_dir}", file=sys.stderr)
        return 2

    tasks = json.loads(tasks_path.read_text())

    rewritten, already_live, unresolved = 0, 0, []
    suffix_counts = Counter()

    for task in tasks:
        old = task.get("_image_path")
        if not old:
            continue
        if DEAD_MARKER not in old:
            print(f"[fatal] unrecognized image path shape: {old}", file=sys.stderr)
            return 2

        suffix = old.split(DEAD_MARKER, 1)[1]
        new = str(live_root / suffix)
        suffix_counts[suffix.split("/")[0]] += 1

        if old == new:
            already_live += 1
        else:
            rewritten += 1
        if not os.path.exists(new):
            unresolved.append(new)
        task["_image_path"] = new

    total = rewritten + already_live
    print(f"pool                : {pool_dir}")
    print(f"tasks with an image : {total}")
    print(f"  rewritten         : {rewritten}")
    print(f"  already correct   : {already_live}")
    for dataset, n in sorted(suffix_counts.items()):
        print(f"  {dataset:<16}: {n}")

    if unresolved:
        print(f"\n[fatal] {len(unresolved)} remapped paths do not exist on disk; refusing to write.")
        for path in unresolved[:5]:
            print(f"    missing: {path}")
        if len(unresolved) > 5:
            print(f"    ... and {len(unresolved) - 5} more")
        print("\nRun `restore_data.sh D_train` first, then re-run this.")
        return 1

    print(f"\nall {total} images resolve under {live_root}")

    if dry_run:
        print("[dry-run] nothing written")
        return 0

    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {tasks_path}")

    manifest_path = pool_dir / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["image_path_remap"] = {
            "live_root": str(live_root),
            "tasks_remapped": rewritten,
            "tasks_with_image": total,
            "per_dataset": dict(sorted(suffix_counts.items())),
            "note": (
                "Images were rebuilt from HuggingFace by split index after the original "
                "CSI volume was unmounted; paths repointed at the in-repo copy."
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"updated {manifest_path}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/domains/full_4modality_clean")
    ap.add_argument("--live-root", default=str(DEFAULT_LIVE))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pool = Path(args.pool)
    if not pool.is_absolute():
        pool = REPO / pool
    return remap(pool, Path(args.live_root), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
