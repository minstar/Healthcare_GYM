# `runs/watchdog.sh` — unattended driver for the job matrix

Healthcare AI GYM. Keeps the training/eval matrix moving on a
partition where almost every node is reserved by someone else, without a human
watching it and without crowding out the other workstreams sharing the slice.

- Script: `/data/project/private/minstar/workspace/hcgym_rebuttal/runs/watchdog.sh`
- Tests: `/data/project/private/minstar/workspace/hcgym_rebuttal/tests/test_watchdog.sh` (134 assertions, submits nothing)
- Decision log: `logs/watchdog.log`

---

## Start and stop

```bash
cd /data/project/private/minstar/workspace/hcgym_rebuttal

# dry run first, always — decides everything, acts on nothing
WD_ONCE=1 WD_DRY_RUN=1 ./runs/watchdog.sh

# overnight
nohup ./runs/watchdog.sh > /dev/null 2>&1 &
echo $! > logs/.watchdog.pid

# watch it
tail -f logs/watchdog.log

# stop it
kill "$(cat logs/.watchdog.pid)"        # logs "STOP received signal, exiting"
```

Stopping the watchdog does **not** stop anything it started. The autoretry loops
and the queued jobs keep going; the watchdog is only the thing that decides what
to start next. To stop an arm entirely, kill its `autoretry.sh` loop.

Single instance is enforced with `flock` on `logs/.watchdog.lock`. A second copy
logs `LOCK another watchdog already holds …` and exits before deciding anything.

---

## What it decides, what it delegates

The launchers are the source of truth for what is submittable. They already
encode backbone readiness, checkpoint usability, `.done` markers, already-queued
dedup and running-loop dedup. Re-deriving any of that here would give two answers
that drift apart. So each tick the watchdog **asks** them via their own `PLAN=1`
mode and treats the answer as the candidate list.

| Delegated — never reimplemented | To |
|---|---|
| which arms exist, in what order, which are launchable | `launch_backbones.sh PLAN=1` |
| backbone weight completeness, `.done` markers, live-loop dedup | `launch_backbones.sh` |
| whether an arm has a checkpoint worth evaluating | `launch_evals.sh` → `resolve_ckpt.sh --check` |
| eval already queued / already scored | `launch_evals.sh` |
| resubmitting after preemption | `autoretry.sh` |

Decided here, because it is not available from the launchers:

- **whether the cluster view is trustworthy enough to act on at all**
- the poll cadence
- whether there is room (concurrency caps, in-flight accounting)
- whether the validation gate is open, and therefore which selector the backbone
  launcher gets
- whether a failure is worth retrying, and quarantining the ones that are not —
  the one thing `autoretry.sh` cannot do, and the reason this file exists
- whether an arm has finished training, and therefore whether scoring it now
  would freeze a mid-training number into `eval_results/`
- slot ordering and single-instance locking

A candidate is filtered here for exactly three reasons of its own: it is
**quarantined**, it is **already in flight**, or (for a trained arm) it has **not
finished training**. Every other skip is the launcher's call and is echoed into
the decision log verbatim as a `DELEGATE` line, so the overnight history shows
why something was not offered without this script knowing.

---

## The one rule that matters: never judge an arm from a superseded log

The first version derived each experiment's state from its newest log **file** and
never consulted the queue it had computed moments earlier. Against the queue as it
actually stood, that misfired on tick 1:

```
tick=1 STATUS     train:q4b_grpo = FATAL :: import error … in slurm_hcgym-q4b_grpo_60585.log
tick=1 QUARANTINE !!!! train:q4b_grpo WILL NOT BE RELAUNCHED
tick=1 QUARANTINE      [dry-run] would SIGTERM autoretry loop pid=857833 for q4b_grpo
```

while `60612 hcgym-q4b_grpo PENDING` — the *retry*, with the venv already fixed —
was sitting in the queue. Both `q4b` arms and three of the four base evals were
quarantined off dead evidence, and with the default `WD_ON_FATAL=stop` both live
autoretry loops were kill targets. Because `q4b` is the gate backbone, that held
the entire remaining matrix behind `GATE 9 non-q4b arm(s) … waiting on the gate`.

`classify()` now consults the queue **first**, and draws this distinction:

- **a live slurm JOB for this experiment → `LIVE`, do not triage.**
  The newest log on disk belongs to an *earlier* attempt. Whatever killed it has
  already been superseded by a later submission. If the new attempt reproduces
  the fault it writes its own log and is triaged then — one retry cycle later,
  which is the correct price for never parking a healthy arm.
- **a live autoretry LOOP but no job → triage anyway.**
  This is precisely the window in which autoretry is about to resubmit, and
  catching a config error here is the entire point. Deferring on a live loop
  would disable the feature, because the loop is alive by construction.

Same tick, after the fix:

```
tick=1 STATUS train:q4b_grpo = LIVE :: job 60612 PENDING (log 60585 is an earlier attempt and is NOT evidence about this one)
```

Covered by tests `S15` (superseded log) and `S15b` (loop alone still triages).

---

## Fail closed when the cluster view is unavailable

`squeue` failing and `squeue` returning nothing are the same empty string, and the
watchdog's in-flight set **and** `launch_evals.sh`'s already-queued dedup both came
from that one call, so they failed together. Measured, with a stub `squeue`
exiting 1 while jobs 60627–60630 were genuinely `PENDING`:

```
ticks run: 3 / sbatch calls: 9
  3 hcgym-eval-base, 3 hcgym-eval-base_react, 3 hcgym-eval-base_strong_tool
```

At the default 300 s interval that is 36 duplicate 8-hour GPU jobs per hour, each
pair racing on the same `eval_results/<tag>` directory.

Now the exit status is checked, and a tick that cannot see the queue triages
nothing and launches nothing:

```
tick=1 DEGRADED cannot read the queue :: squeue exited 1: slurm_load_jobs error: Socket timed out on send/recv operation
tick=1 DEGRADED   triaging nothing and launching nothing this tick …
tick=1 DEGRADED   consecutive degraded ticks: 1
tick=1 END      launched train=0 eval=0, held=0 (degraded)
```

Same scenario after the fix: **0 sbatch calls**. It recovers on its own the tick
the controller answers again (`RECOVER queue readable again after N degraded tick(s)`).

`sacct` is treated the same way, but per item rather than per tick: a failed
`sacct` yields `DEFER`, never a strike. Belt and braces were added at both other
layers that share the same `squeue` — `launch_evals.sh` and `autoretry.sh` now
refuse to submit when the queue is unreadable instead of failing open.

---

## Concurrency caps, and why those numbers

Measured, not assumed:

```
$ sinfo -p pt2_preemptible -h -N -o '%n' | wc -l     →  71
$ scontrol show reservation
  <reservation A>  NodeCnt=49  EndTime=2026-07-31
  <reservation B>  NodeCnt=16  EndTime=2026-07-31
→ usable: [171, 181, 186, 188, 189, 190]   count: 6
```

This account is on neither reservation's user list. Those six nodes (48 GPUs) are
shared with other workstreams' jobs, which must never be touched.

Two caps, because the two job classes cost wildly different amounts and one
number would have to be set for the worse case:

| Cap | Default | Unit | Why |
|---|---|---|---|
| `WD_MAX_TRAIN` | 2 | **whole nodes** | `train_hcgym.slurm` is `--nodes=1 --gres=gpu:8 --exclusive`, so one arm = one node. 2 of 6 leaves two thirds of the slice for everything else. |
| `WD_MAX_EVAL` | 4 | **GPUs** | `eval_agentic.slurm` takes 1 GPU, except `q27b_*` which `launch_evals.sh` submits with `--gres=gpu:4`. |

Worst case with the defaults: 2 exclusive nodes (16 GPUs) + 4 eval GPUs = **20 of
48 GPUs**, touching at most 3 of the 6 nodes.

`WD_MAX_EVAL` really does count GPUs now. The first version documented it as a GPU
budget but counted one slot per job name, so three `q27b` evals could take 12 GPUs
— 1.5 whole nodes — under a cap that read `3`. 4 is also the smallest value that
can ever admit a `q27b` eval at all; below it the watchdog says so explicitly
rather than idling forever:

```
HOLD eval q27b_grpo :: needs 4 gpu(s) but WD_MAX_EVAL=3 — it can NEVER be admitted; raise WD_MAX_EVAL to at least 4
```

`hcgym-smoke` counts against the training cap (it occupies a node) but is excluded
from triage via `WD_IGNORE`.

---

## The validation gate

Until the cheap 4B arm proves the stack end to end, only 4B arms launch. The gate
string is exactly:

```
training/global_step:
```

`verl`'s console logger builds each line in
`verl/utils/logger/aggregate_logger.py::concat_dict_to_str`, which keeps only
`numbers.Number` values, and `verl/trainer/ppo/ray_trainer.py` calls it from
exactly two places:

- **line 1509** — `logger.log(data=val_metrics, …)`, the validation pass that runs
  *before* training. `val_metrics` carries only `val-core/*` keys.
- **line 1937** — end of one training iteration. `metrics` gets
  `"training/global_step"` at **line 1907**.

The key exists in only one non-experimental code path:

```
verl/trainer/ppo/ray_trainer.py:1907:   "training/global_step": self.global_steps,
```

(other hits are `verl/experimental/*` trainers, which `main_ppo` does not use).
Line 1907 sits **after** `_update_actor()` at line 1773 in straight-line code, so
Python cannot reach it unless rollout generation, reward scoring, advantage
computation and the optimizer step all returned without raising.
`self.global_steps` is an `int`, so it survives the numeric filter and is printed.

Rejected alternatives, each of which means "started" rather than "progressed":

| Candidate | Why not |
|---|---|
| `step:` | **also printed by the pre-training validation pass** at line 1509, so it opens the gate before a single optimizer step. The exact false positive the gate exists to prevent. |
| `[run] exp=…` | the sbatch preamble |
| `Loading weights` | job 60585's `ImportError` fired *after* a 100 % weight load |
| `server healthy` | sglang came up; no gradient was applied |
| a checkpoint on disk | `save_freq=10`, so a correct run shows nothing for ten steps, and a resumed run already has one |

The marker is sticky (`.wd_gate_open`); delete it to re-close. While closed, the
watchdog reports the size of what is waiting:
`GATE 9 non-q4b arm(s) are otherwise launchable and waiting on the gate`.

---

## Failure taxonomy

| Class | Trigger | Action |
|---|---|---|
| `LIVE` | a live slurm job for this exp, or `sacct` says live, or a loop between submissions with no log | nothing; clears any strike counter |
| `DONE` | `.autoretry_<exp>.done`, a `[done] <exp>` / `[done] eval <tag>` sentinel, or `sacct` `COMPLETED` | nothing; consumes no slot; clears strikes |
| `RETRY` | `sacct` in `PREEMPTED\|NODE_FAIL\|TIMEOUT\|REQUEUED\|CANCELLED\|OUT_OF_MEMORY`, or a matching log line | leave it — `autoretry.sh` owns it; clears strikes |
| `DEFER` | `sacct` itself failed | nothing, **no strike**, try again next tick |
| `FATAL` | a log matches a fatal signature | quarantine; SIGTERM its loop (argv-checked) |
| `UNKNOWN` | failed, no known signature | one strike per **distinct job id**; quarantine at `WD_UNKNOWN_STRIKES` (3) |
| `NEW` | no log, no job | nothing |

**Fatal signatures.** Both anchors were taken verbatim from logs already in this
run root, not invented: job 60585 `ImportError: cannot import name
'AutoModelForCausalLMWithValueHead' from 'trl'`, and eval job 60592
`error: argument --benchmarks: invalid choice: 'mmlu_college_bio'`. Retrying these
is pointless — the next allocation runs the identical command against the
identical filesystem. The bar for membership is high because a false positive
parks an arm. Deliberately **not** listed:

- `CUDA out of memory` — sglang prints OOM during memory profiling on runs that go
  on to succeed. A genuine OOM still fails the job and is caught by the strike
  counter instead: slower, safe.
- `AssertionError` — far too broad; verl asserts on recoverable conditions too.

**Quarantine lifecycle.** Keyed to the job id it was derived from. A **newer**
attempt that does not reproduce the signature clears it automatically
(`UNQUARANTINE … job 60862 is newer`), so a fix does not need a human to un-stick
the matrix. Strikes are cleared the moment an arm looks healthy, so they cannot
accumulate over a whole night. Every tick restates what is parked:

```
HELD 1 quarantined, needing a human: train:q9b_grpo :: rm …/.wd_quarantine_<kind>_<name> to release
```

---

## Why evals wait for training to finish

`launch_evals.sh` asks `resolve_ckpt.sh --check`, which answers "there is something
loadable here" — and with `trainer.save_freq=10` that is true from step 10, while
the arm still has fifty steps to go. Scoring then writes `eval_results/<tag>`, and
`launch_evals.sh` skips a populated `eval_results/<tag>` **forever after**. The
rebuttal table would quietly carry step-10 numbers for arms that trained to
completion, and nothing in the log would say so.

The watchdog knows the training state, so it holds a trained arm's eval until that
arm is `DONE`:

```
HOLD eval q9b_grpo :: arm q9b_grpo is LIVE, not DONE — save_freq=10 means a checkpoint exists
from step 10, and scoring it now would populate eval_results/q9b_grpo and make launch_evals
skip the finished model forever
```

Untrained `base*` rows (`base`, `base_strong_tool`, `base_react`, `base_reflexion`)
need no checkpoint and are never held. `WD_EVAL_REQUIRE_DONE=0` disables the check.

---

## Safety properties

- **It never cancels a slurm job.** Not "only hcgym ones" — there is no job-cancel
  code path at all, and the test suite asserts that command's name appears nowhere
  in the file.
- **Other workstreams are invisible.** Only `squeue -u $WD_USER` is read, and every
  job whose name does not begin with `hcgym-` is dropped at the source. Verified:
  `pro4full-*`, `pivotrl-*`, `saas-*` produce `INFLIGHT train 0/2 [] eval 0/3 []`
  and are never named in the log.
- **The only process it may signal** is an `autoretry.sh` loop for a quarantined
  experiment. The pid must pass an **argv** check — some argv element ends in
  `/autoretry.sh` and the *next* element is exactly the experiment name — read
  from `/proc/<pid>/cmdline`, and must not be this process or one of its ancestors.
  A substring glob over the joined command line was not enough: it matched, and
  would have SIGTERMed, an operator shell whose command merely *mentioned* the arm.
  `WD_ON_FATAL=warn` disables even the argv-checked kill.
- **The lock fd does not leak.** `exec 9>>lock` is not close-on-exec, so the 72 h
  autoretry loops inherited it and a watchdog that died at 01:00 could not restart
  until they exited — refused the lock by its own grandchildren. Every
  child-spawning site now closes it (`9>&-`).
- **The decision log is bounded** (`WD_LOG_MAX_BYTES`, default 32 MiB, rotates to
  `watchdog.log.1`). Measured growth ≈ 2 MiB/day at the default interval; tick
  wall time ≈ 2 s per 300 s interval, so there is no CPU spin.

---

## Environment reference

| Variable | Default | Meaning |
|---|---|---|
| `WD_INTERVAL` | `300` | seconds between ticks |
| `WD_MAX_TRAIN` | `2` | training cap, in whole exclusive nodes |
| `WD_MAX_EVAL` | `4` | eval cap, in GPUs |
| `WD_GATE_TAG` | `q4b` | gate backbone; also the selector while the gate is closed |
| `WD_GATE_STRING` | `training/global_step:` | the string that opens the gate |
| `WD_UNKNOWN_STRIKES` | `3` | distinct failed job ids before an unclassifiable arm is quarantined |
| `WD_ON_FATAL` | `stop` | `stop` = SIGTERM the quarantined arm's loop; `warn` = log only |
| `WD_EVAL_REQUIRE_DONE` | `1` | hold a trained arm's eval until that arm is DONE |
| `WD_IGNORE` | `smoke diag probe` | job stems counted against the cap but never triaged |
| `WD_LOG_MAX_BYTES` | `33554432` | rotate the decision log past this size |
| `WD_ONCE` / `WD_MAX_TICKS` | `0` | run one / N ticks then exit |
| `WD_DRY_RUN` | `0` | decide and log, act on nothing |
| `WD_USER` | `$USER`, else `id -un` | the slurm user whose queue is read |
| `WD_LOGDIR` / `WD_STATE_DIR` | `logs/` | slurm logs / watchdog state (split for testing) |
| `WD_PROC` | `/proc` | overridable so the argv guard can be tested with fixtures |

---

## What code review changed

Three independent adversarial code-review passes attacked the first version.
Everything below was reproduced against the live cluster before being fixed.

| # | Severity | Defect | Fix | Test |
|---|---|---|---|---|
| 1 | blocker | `classify()` judged an arm from the newest log file and never consulted the queue — quarantined both `q4b` arms and 3 base evals off superseded logs on tick 1, and SIGTERMed both live loops | queue consulted first; a live job means `LIVE`, a loop alone still triages | `S15`, `S15b` |
| 2 | blocker | a failing `squeue` read as "nothing is running" → 3 duplicate 8-hour eval jobs per tick, forever | exit status checked; a degraded tick decides nothing. Same guard added to `launch_evals.sh` and `autoretry.sh` | `S16`, `S24` |
| 3 | blocker (results) | evals fired against `global_step_10` mid-training, then `eval_results/<tag>` blocked the real eval forever | a trained arm's eval waits for `DONE` | `S20`, `S20b` |
| 4 | major | `stop_loop()`'s `/proc` guard was a substring glob — it SIGTERMed an operator shell that merely mentioned the arm | argv match from `/proc/<pid>/cmdline` + ancestor refusal | `S19`, `S19b` |
| 5 | major | a `slurmdbd` outage charged strikes to a healthy RUNNING arm until it was quarantined | `sacct` failure → `DEFER`, no strike | `S18` |
| 6 | major | `WD_MAX_EVAL` documented as GPUs but counted jobs; 3 × `q27b` = 12 GPUs under a cap reading "3" | genuine GPU accounting; default raised to 4 so `q27b` is admissible; impossible conditions named | `S21`, `S21b`, `S21c` |
| 7 | major | `USER` unset (cron/systemd/container) killed `live_jobs()` under `set -u`; error went to the discarded stderr and the tick ran against an empty queue | `WD_USER` resolved once, with `id -un` fallback | `S17` |
| 8 | major | the flock fd was inherited by the 72 h autoretry loops, so a restarted watchdog was refused its own lock | `9>&-` at every child-spawning site | `S23` |
| 9 | minor | strikes only ever incremented, accumulating over a whole night | cleared on `LIVE`/`DONE`/`RETRY` and on unquarantine | `S22` |
| 10 | minor | `watchdog.log` had no size bound | rotation at `WD_LOG_MAX_BYTES` | — |
| 11 | (found here) | `live_loops()` counted any process whose *joined* command line mentioned a loop, silently withholding that arm's slot | argv-validated when `/proc` is readable | `S2c` |

`autoretry.sh` was replaced by **atomic rename**, not edited in place: pids 857833
and 858050 were executing it, and bash reads a script incrementally by byte
offset, so an in-place edit corrupts the running loops. The rename leaves them on
the old inode; new loops get the fixed file. The guard was proven with stubs
before installation — `MODE=fail` → 0 sbatch calls, `MODE=busy` → 0,
`MODE=idle` → submits normally.

---

## Known limitations — things I did NOT fix, and why

**1. `launch_backbones.sh:154` dedups with the same joined-cmdline `pgrep -f`.**
A long-lived process whose command line contains `autoretry.sh <exp> ` makes the
*launcher* skip that arm, so it is never offered to the watchdog at all. The
watchdog's own accounting is now argv-accurate (`S2c`), but this check is
delegated by design and fixing it means duplicating the `/proc` argv walk into a
second script. The failure mode is a stall, not damage, it needs an operator shell
that literally spells out the arm, and it self-heals when that shell exits.
Diagnose with `pgrep -af "autoretry.sh <exp> "`; if the only hit is not a real
loop, close that shell. Worth doing properly if it ever bites.

**2. A `FATAL` quarantine still needs a human to clear it.** Deliberate. The whole
reason this script exists is that `autoretry.sh` would otherwise burn 60
submissions — four hours of a six-node slice — reproducing one `ImportError`.
Auto-expiring a fatal quarantine on a timer re-creates exactly that. Three things
make it safe now: the misclassification that made it dangerous is gone (#1), a
newer non-fatal job id still clears it automatically, and every tick prints a
`HELD` line naming what is parked and the `rm` that releases it, so an overnight
log cannot go quiet about it.

**3. `WD_MAX_TRAIN` stays at 2.** One code-review pass framed the GPU footprint as
too large. With the eval cap now counting GPUs correctly the true worst case is 20
of 48 GPUs across at most 3 of 6 nodes — not the 28/48 computed against the old
job-counting cap. Lowering it further would make an 11-arm matrix unable to finish
before the deadline. It is a policy number and it is an env var.

**4. A residual race between the watchdog's `squeue` snapshot and the launcher's
own `squeue` call.** The watchdog can verify the queue is readable and have the
controller fail two seconds later inside `launch_evals.sh`. That window cannot be
closed from the watchdog; it is closed *inside* `launch_evals.sh` and
`autoretry.sh`, which now fail closed themselves. Three independent layers must
now all fail in the same few seconds to produce a duplicate.

**5. `eval_agentic.slurm` hardcodes `PORT=31000`.** Two evals scheduled onto the
same node would collide on the sglang port — which the GPU-counted eval cap makes
possible (four 1-GPU evals can land together). Not fixed: it is outside the
reviewed file, no code-review pass raised it, and it needs its own verification
that I have not done. Flagged as the top follow-up.

**6. `autoretry.sh:52` still picks its latest log with `ls -t` (mtime), while the
watchdog picks by job id.** They can disagree when a preempted job's log is
touched after a newer job starts. Not fixed: it only affects `autoretry.sh`'s own
sentinel check, the sentinel is written once at the end of a successful run, and I
did not want two behavioural changes in one atomic replacement of a file that two
live loops depend on.

**7. `scripts/rebuttal/run_reflexion_eval.py` is uncommitted in the
`Healthcare_GYM` repo.** It was modified at 23:10, over an hour before this
session began, and I did not touch it. Nothing in the repo was changed by this
work — the watchdog, the launchers and the tests all live in `hcgym_rebuttal/`,
which is not a git repository — so there was nothing for me to commit, and
committing someone else's in-progress edit under this change would be wrong.

---

## Test output

```
$ ./tests/test_watchdog.sh
…
passed 134   failed 0
```

Nothing is submitted and nothing is cancelled. `squeue`, `sacct`, `pgrep` and
`sbatch` are stubs on `PATH` reading fixture files; `WD_LOGDIR`, `WD_STATE_DIR`
and `WD_PROC` are per-scenario scratch directories, so the real run root is never
written to. The discovery half of the delegation is **real** — every scenario
actually executes `launch_backbones.sh` / `launch_evals.sh` in read-only `PLAN=1`
mode, so the candidate lists under test are the launchers' actual answers and
cannot drift from them.
