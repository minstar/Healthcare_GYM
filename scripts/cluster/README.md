# `scripts/cluster/` — the authors' Slurm harness

**This is not an entry point.** Every `.slurm` file here names
`--partition=pt2_preemptible` and absolute paths on one specific B200 / sm100
fleet. `sbatch`-ing one somewhere else fails in ways that are hard to diagnose.
It is committed because it is the only record of *how* the reported runs were
produced, and because most of what made them run at all is encoded in it rather
than in the portable code under `scripts/`.

```
runs/     the sbatch templates, the launchers, the autoretry loop, the watchdog
tests/    134 + 48 + 17 checks over the launchers' and watchdog's decisions
docs/     what each fix was and how it was found
requirements-frozen.txt
```

## Paths

`runs/common_env.sh` resolves three roots, each overridable from the environment,
defaulting to the layout the reported runs used:

| variable | default | what it holds |
|---|---|---|
| `RUN_ROOT` | `/data/project/private/minstar/workspace/hcgym_rebuttal` | data, checkpoints, logs, venv |
| `HCGYM_ROOT` | `…/minstar/Healthcare_GYM` | this repo |
| `VERL_ROOT` | `…/workspace/verl_ttopd` | the verl fork |

The resolved values are echoed on every run, because `sbatch --export=ALL`
propagates whatever is in the submitting shell and a stale `RUN_ROOT` would
silently redirect a run rather than fail.

`RUN_ROOT` is also what makes the harness testable: `tests/test_watchdog.sh`
points it at a tree whose `eval_results/` is empty so the eval assertions test the
watchdog's cap logic rather than the launcher's dedupe against live results.

Credentials, if any tool wants them, come from `$HCGYM_ENV_FILE`, default
`$RUN_ROOT/.env`. Nothing in the training or eval path needs one — local weights,
a local sglang server, a local FTS5 index — so an absent file is not an error.

## The environment is not reproducible from `requirements-frozen.txt` alone

Two steps were done by hand; both are recorded in that file's header. The short
version: flash-attn must be the prebuilt wheel matching this exact torch ABI, and
`causal-conv1d` must *not* be installed — it builds, imports, and then dies at
load with an undefined `c10::cuda` symbol. Verifying an install is not the same as
verifying a load.

## Running the tests

Nothing is submitted and nothing is cancelled; `squeue`/`sbatch`/`sacct`/`pgrep`
are replaced by fixture-backed stubs on `PATH`.

```bash
cd "$RUN_ROOT"      # or wherever runs/ and tests/ live together
bash tests/test_arms.sh           # 48  — arm resolution in train_hcgym.slurm
bash tests/test_watchdog.sh       # 134 — the watchdog's decisions
bash tests/test_scratch_prune.sh  # 17  — the /tmp reaper, which deletes directories
```

`tests/mutate.sh` is a mutation harness: it edits scripts in place to prove the
tests fail when the behaviour they assert is broken. Read it before running it.

## Things that are load-bearing and look like they are not

- **`--open-mode=append`** on every `.slurm`. The partition preempts with
  `PreemptMode=REQUEUE` and `GraceTime=0`, so a preempted job restarts under the
  same job id and reopens the same log; Slurm's default open mode truncates it.
  This was found the expensive way — `q4b_grpo`'s checkpoints run contiguously
  from step 10 to 630 with an identical config either side, but its validation
  curve survives only for steps 10–190 and 390–600. The middle third was
  overwritten by its own requeues.
- **`attention_backend=triton` *and* `mm_attention_backend=triton_attn`.** They
  look like a duplicate and are not: one covers text attention, the other the
  vision tower, and verl hardcodes `fa3` for both, which raises on Blackwell.
  Dropping the first re-breaks text attention while multimodal keeps working.
- **`unset ROCR_VISIBLE_DEVICES`.** Slurm here exports it alongside
  `CUDA_VISIBLE_DEVICES` with an identical mask, and verl's worker refuses to
  start when both are set.
- **No `expandable_segments`.** verl colocates actor and rollout, so sglang frees
  its KV cache through TorchMemorySaver, which refuses to run under expandable
  segments and takes the scheduler down with it.
- **`free_cache_engine=False`.** At `gpu_memory_utilization=0.3` on a 192GB card
  there is nothing to reclaim, and the release/resume path dies on an inplace
  write to an `inference_mode` tensor.
- **The trailing space in `pgrep -f "autoretry.sh ${tag}_${arm} "`.** Without it
  the pattern also matches `q9b_grpo_cosine` and wrongly skips `q9b_grpo`.

## What the arms are

`runs/train_hcgym.slurm` carries three, as an additive ladder so each component
can be priced on its own:

| arm | what it adds |
|---|---|
| `grpo` | — |
| `grpo_cosine` | + cosine length-controlled reward |
| `ttopd` | + turn-level KL against a frozen teacher, bidirectional sign flip, top-K filtering |

`ttopd` reproduces **what was measured**, which is not everything the method
description names: outcome-conditioned hints, turn truncation and the EMA teacher
were all inert in the reported runs. `docs/HINT_FIX.md` has the evidence, and
`scripts/rebuttal/verify_hint_injection.py` fails on the old revision and passes
on the fixed fork. `runs/train_ttopd_hints.slurm` is the separate arm that turns
the hints on.
