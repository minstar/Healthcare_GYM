#!/bin/bash
# Unattended driver for the Healthcare_GYM rebuttal job matrix.
#
#   nohup ./watchdog.sh > /dev/null 2>&1 &        # overnight
#   WD_ONCE=1 WD_DRY_RUN=1 ./watchdog.sh          # one tick, decide but do not act
#
# ─── WHAT THIS DECIDES vs WHAT IT DELEGATES ───────────────────────────────────
#
# The launchers are the source of truth for what is submittable. They already
# encode backbone readiness, checkpoint usability, .done markers, already-queued
# dedup and running-loop dedup. Re-deriving any of that here would give two
# answers that drift apart. So each tick this script ASKS them, via their own
# PLAN=1 mode, and treats the answer as the candidate list.
#
#   DELEGATED — never reimplemented here
#     what training arms exist, in what order, and which are launchable
#                                             -> launch_backbones.sh PLAN=1
#     whether a backbone's weights are complete
#                                             -> launch_backbones.sh
#     whether an arm already has a .done marker or a live autoretry loop
#                                             -> launch_backbones.sh
#     whether an arm has a checkpoint worth evaluating
#                                             -> launch_evals.sh (resolve_ckpt.sh --check)
#     whether an eval is already queued or already scored
#                                             -> launch_evals.sh
#     resubmitting a preempted training job   -> autoretry.sh
#
#   DECIDED HERE — genuinely new, not available from the launchers
#     WHEN to ask at all (the poll interval)
#     WHETHER the cluster view is trustworthy enough to act on at all
#     WHETHER there is room  (concurrency caps, in-flight accounting)
#     WHETHER the validation gate is open, and therefore whether to hand the
#         backbone launcher the narrow "q4b" selector or the full "all"
#     WHETHER a failure is worth retrying at all, and quarantining the ones that
#         are not — this is the one thing autoretry.sh cannot do, and the reason
#         this file exists
#     WHETHER an arm has finished training, and therefore whether scoring it now
#         would freeze a mid-training number into eval_results/
#     the order in which candidates consume the remaining slots (launcher order)
#     single-instance locking
#
# A candidate is filtered here for exactly THREE reasons of its own: it is
# quarantined, it is already in flight, or (for a trained arm) it has not finished
# training. Every other skip is the launcher's call and is passed through to the
# log verbatim.
#
# ─── THE ONE RULE THAT MATTERS: NEVER JUDGE AN ARM FROM A SUPERSEDED LOG ──────
#
# The first version of this script derived every experiment's state from its
# newest log FILE and never looked at the queue. Against the queue as it actually
# stood that misfired on tick 1: q4b_grpo's newest log was job 60585, which died
# on an ImportError, while job 60612 for the same arm was already PENDING with the
# venv fixed. It quarantined both q4b arms and three base evals off already-dead
# evidence and (WD_ON_FATAL=stop) SIGTERMed the two live autoretry loops. Since
# q4b is the gate backbone, that held the entire remaining matrix.
#
# So classify() now consults the queue FIRST, and the distinction it draws is:
#
#   a live slurm JOB for this experiment  -> LIVE, do not triage.
#       The newest log on disk belongs to an EARLIER attempt. Whatever killed
#       that attempt has already been superseded by a submission someone (or
#       autoretry) made afterwards. If the new attempt reproduces the fault it
#       will write its own log and be triaged then — one retry cycle later, which
#       is the correct price for never parking a healthy arm.
#
#   a live autoretry LOOP but no job      -> TRIAGE ANYWAY.
#       This is the window in which autoretry is about to resubmit, and catching
#       a config error here is the entire reason this script exists. Deferring on
#       a live loop would disable the feature, because the loop is always alive.
#
# ─── FAIL CLOSED WHEN THE CLUSTER VIEW IS UNAVAILABLE ─────────────────────────
#
# `squeue` failing and `squeue` returning nothing are the same empty string, and
# the watchdog's in-flight set AND launch_evals.sh's already-queued dedup both
# come from that one call, so they fail together. Read as "nothing is running",
# a controller hiccup made this script submit three duplicate 8-hour eval jobs
# per tick — 36/hour at the default interval — each pair racing on the same
# eval_results/<tag> directory.
#
# Therefore squeue's EXIT STATUS is checked, and a tick that cannot see the queue
# triages nothing and launches nothing. It logs why and waits. sacct is treated
# the same way per item: a failed sacct yields DEFER, never a strike.
#
# ─── THE VALIDATION GATE ──────────────────────────────────────────────────────
#
# Until the cheap 4B arm proves the stack end to end, only 4B arms are launched.
# The gate string is
#
#     training/global_step:
#
# verl's console logger builds each line in verl/utils/logger/aggregate_logger.py
# (`concat_dict_to_str`, which keeps only numeric metrics) and the trainer calls
# it from exactly two places in verl/trainer/ppo/ray_trainer.py:
#
#   line 1509  logger.log(data=val_metrics, ...)  validation BEFORE training.
#              val_metrics carries only val-core/* keys — no training/* key.
#   line 1937  logger.log(data=metrics, ...)      end of one training iteration.
#              `metrics` gets "training/global_step" at line 1907.
#
# So the string exists in one code path only, and that path sits AFTER
# _update_actor() (line 1773) in a straight-line function. Python cannot reach
# line 1907 unless rollout generation, reward scoring, advantage computation and
# the optimizer step all returned without raising. `self.global_steps` is an int,
# so it survives the numbers.Number filter and is actually printed.
#
# That is why this string and not something easier:
#   "[run] exp=..."         the sbatch preamble — says the job started, nothing more
#   "Loading weights"       says weights loaded; the ImportError in job 60585 hit
#                           AFTER a 100% weight load, so this proves nothing
#   "server healthy"        says sglang came up, not that a gradient was applied
#   "step:"                 ALSO printed by the pre-training validation pass at
#                           line 1509, so it opens the gate before a single
#                           optimizer step has run — the exact false positive
#                           this gate exists to prevent
#   a checkpoint on disk    save_freq=10, so a correct run shows nothing for ten
#                           steps; too slow, and a resumed run has one already
#
# The marker is sticky (.wd_gate_open): the pipeline only has to be proven once.
# Delete that file to re-close the gate.
#
# ─── FAILURE TRIAGE ───────────────────────────────────────────────────────────
#
# autoretry.sh resubmits every 180s up to 60 times. That is right for preemption
# (REQUEUE, GraceTime=0, verl resumes from disk) and catastrophic for a config
# error — job 60585 died on `ImportError: cannot import name
# 'AutoModelForCausalLMWithValueHead' from 'trl'` and was already on attempt 3
# when this was written. Sixty of those is four hours of a six-node slice burned
# reproducing the same traceback.
#
# So: sacct's terminal state is authoritative for preemption; otherwise the log
# is scanned for signatures that cannot fix themselves; anything else gets
# WD_UNKNOWN_STRIKES attempts before being quarantined as unclassifiable.
# Quarantine is keyed to the job id it was derived from, so a NEWER attempt that
# does not reproduce the signature clears it automatically — a fix does not need
# a human to un-stick the matrix. Strikes are cleared the moment an arm looks
# healthy again, so they cannot accumulate across a whole night.
#
# ─── WHY EVALS WAIT FOR TRAINING TO FINISH ────────────────────────────────────
#
# launch_evals.sh asks resolve_ckpt.sh --check, which answers "there is something
# loadable here" — and with trainer.save_freq=10 that is true from step 10, while
# the arm still has fifty steps to go. Scoring then writes eval_results/<tag>,
# and launch_evals.sh skips a populated eval_results/<tag> forever after. The
# rebuttal table would quietly carry step-10 numbers for arms that trained to
# completion. The watchdog knows the training state, so it holds a trained arm's
# eval until that arm is DONE. Untrained rows (base*) are never held.
# WD_EVAL_REQUIRE_DONE=0 disables this.
#
# ─── SAFETY ───────────────────────────────────────────────────────────────────
#
#   * it never cancels a slurm job. Not "only hcgym ones" — it has no job-cancel
#     code path at all, and tests/test_watchdog.sh asserts that command's name
#     appears nowhere in this file, which is why that name is not written out
#     even in this comment.
#   * only ever reads `squeue -u $WD_USER`, and ignores every job whose name does
#     not begin with hcgym-  (pro4full-*, pivotrl-*, saas-* are invisible to it)
#   * the only process it may signal is an autoretry.sh loop for a quarantined
#     experiment. The pid must pass an ARGV check — some argv element ends in
#     /autoretry.sh and the NEXT element is exactly the experiment name — and must
#     not be an ancestor of this process. A substring match on the joined command
#     line was not enough: it matched, and would have SIGTERMed, an operator shell
#     whose command line merely mentioned the arm. WD_ON_FATAL=warn disables even
#     the argv-checked kill.
#   * flock single-instance, and the lock fd is closed across every child it
#     spawns — otherwise the 72h autoretry loops inherit it and a restarted
#     watchdog is refused the lock by its own grandchildren.
#
set -uo pipefail
export LC_ALL=C

# Overridable so the harness can run against another checkout or a fixture tree;
# the default is the layout every run in the paper used.
RUN_ROOT="${RUN_ROOT:-/data/project/private/minstar/workspace/hcgym_rebuttal}"

# Where slurm logs and autoretry's .done/.count markers live. Overridable so the
# test suite can point the whole decision path at a fixture directory.
WD_LOGDIR="${WD_LOGDIR:-${RUN_ROOT}/logs}"
# Where this script keeps its own state (decision log, lock, quarantine, strikes).
WD_STATE_DIR="${WD_STATE_DIR:-${WD_LOGDIR}}"
WD_LOG="${WD_LOG:-${WD_STATE_DIR}/watchdog.log}"
WD_LOCK="${WD_LOCK:-${WD_STATE_DIR}/.watchdog.lock}"

# Under cron/systemd/a container there is no login shell, so USER may be unset —
# and with `set -u` that killed live_jobs() inside a command substitution, which
# looked exactly like an empty queue. Resolve it once, here, and loudly.
WD_USER="${WD_USER:-${USER:-}}"
[ -n "$WD_USER" ] || WD_USER="$(id -un 2>/dev/null || true)"

# /proc, overridable so the argv guard on the only kill path can be tested with
# fixtures instead of live processes.
WD_PROC="${WD_PROC:-/proc}"

WD_LAUNCH_BACKBONES="${WD_LAUNCH_BACKBONES:-${RUN_ROOT}/runs/launch_backbones.sh}"
WD_LAUNCH_EVALS="${WD_LAUNCH_EVALS:-${RUN_ROOT}/runs/launch_evals.sh}"

WD_INTERVAL="${WD_INTERVAL:-300}"

# Concurrency caps — the reason this watchdog cannot crowd out the user's other
# workstreams. pt2_preemptible has 71 nodes; 65 are held by two other teams'
# reservations (49 + 16) until 2026-07-31 and this account is on neither user
# list, leaving SIX usable nodes (171, 181, 186, 188, 189, 190) which are also
# carrying this account's other workstreams.
#
# Two caps rather than one, because the two job classes cost wildly different
# amounts and a single number would have to be set for the worse case:
#   WD_MAX_TRAIN  train_hcgym.slurm is --nodes=1 --gres=gpu:8 --exclusive, so this
#                 counts WHOLE NODES. 2 of 6 leaves two thirds of the slice.
#   WD_MAX_EVAL   counts GPUs, and it really does count GPUs: eval_agentic.slurm
#                 takes 1 GPU except for q27b_*, which launch_evals.sh submits
#                 with --gres=gpu:4. Counting one slot per JOB (the first version)
#                 let three q27b evals take 12 GPUs under a cap that read "3".
#                 4 is half a node, and is also the smallest value that can ever
#                 admit a q27b eval at all.
#
# Worst case with the defaults: 2 exclusive nodes (16 GPUs) + 4 eval GPUs = 20 of
# the 48 GPUs on the six-node slice, touching at most 3 of the 6 nodes.
WD_MAX_TRAIN="${WD_MAX_TRAIN:-2}"
WD_MAX_EVAL="${WD_MAX_EVAL:-4}"

# Validation gate. WD_GATE_TAG is a backbone tag understood by launch_backbones.sh
# and is ALSO the selector handed to it while the gate is closed.
WD_GATE_TAG="${WD_GATE_TAG:-q4b}"
WD_GATE_STRING="${WD_GATE_STRING:-training/global_step:}"

# Failures that match no known signature are retried this many times (counted per
# distinct job id, not per tick) before being quarantined as unclassifiable.
WD_UNKNOWN_STRIKES="${WD_UNKNOWN_STRIKES:-3}"

# stop = SIGTERM the quarantined arm's autoretry loop so it stops resubmitting.
# warn = log loudly and leave the loop alone.
WD_ON_FATAL="${WD_ON_FATAL:-stop}"

# Hold a trained arm's eval until that arm has actually finished training.
WD_EVAL_REQUIRE_DONE="${WD_EVAL_REQUIRE_DONE:-1}"

# Job-name stems that are not part of the matrix; classified for the log but
# never acted on.
WD_IGNORE="${WD_IGNORE:-smoke diag probe}"

# Rotate the decision log past this size so a 72h unattended run cannot grow it
# without bound (measured ~2 MiB/day at the default interval).
WD_LOG_MAX_BYTES="${WD_LOG_MAX_BYTES:-33554432}"

WD_ONCE="${WD_ONCE:-0}"
WD_DRY_RUN="${WD_DRY_RUN:-0}"
WD_MAX_TICKS="${WD_MAX_TICKS:-0}"

TICK=0
DEGRADED_TICKS=0

mkdir -p "$WD_STATE_DIR" "$WD_LOGDIR" 2>/dev/null

# Scratch file used to capture a child's stderr without swallowing it.
WD_ERR="${WD_STATE_DIR}/.wd_stderr.$$"
: > "$WD_ERR" 2>/dev/null
trap 'rm -f "$WD_ERR" 2>/dev/null' EXIT

# ── logging ───────────────────────────────────────────────────────────────────
# Append-only, one decision per line, every line carrying its reason. Written to
# both the file and stdout so `tail -f` and `nohup ... > x` both work.
log() { # log <LEVEL> <message...>
    local lvl="$1"; shift
    printf '%s tick=%-4s %-10s %s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$TICK" "$lvl" "$*" \
        | tee -a "$WD_LOG"
}

rotate_log() {
    [ -f "$WD_LOG" ] || return 0
    local sz; sz="$(stat -c %s "$WD_LOG" 2>/dev/null || echo 0)"
    [ "${sz:-0}" -gt "$WD_LOG_MAX_BYTES" ] 2>/dev/null || return 0
    mv -f "$WD_LOG" "${WD_LOG}.1" 2>/dev/null
    log ROTATE "decision log passed ${WD_LOG_MAX_BYTES} bytes; previous log is ${WD_LOG}.1"
}

# ── single instance ───────────────────────────────────────────────────────────
exec 9>>"$WD_LOCK" || { echo "cannot open lock ${WD_LOCK}" >&2; exit 1; }
if ! flock -n 9; then
    log LOCK "another watchdog already holds ${WD_LOCK} :: exiting without deciding anything"
    exit 0
fi
printf 'pid=%s started=%s\n' "$$" "$(date '+%F %T')" >&9

trap 'log STOP "received signal, exiting"; exit 0' INT TERM

# ── slurm view (this user, hcgym-* only) ──────────────────────────────────────
LIVE_STATES="PENDING|RUNNING|COMPLETING|CONFIGURING|SUSPENDED|REQUEUED|RESIZING|SIGNALING|STAGE_OUT"

SLURM_OK=1
SLURM_ERR=""
SQUEUE_RAW=""

# Take one snapshot of the queue and say plainly whether it can be trusted.
# Called directly (never in a command substitution) so its globals survive.
slurm_snapshot() {
    SLURM_OK=1; SLURM_ERR=""; SQUEUE_RAW=""
    if [ -z "$WD_USER" ]; then
        SLURM_OK=0; SLURM_ERR="cannot determine the slurm user (USER unset and \`id -un\` failed); set WD_USER"
        return 0
    fi
    if ! command -v squeue >/dev/null 2>&1; then
        SLURM_OK=0; SLURM_ERR="squeue is not on PATH"
        return 0
    fi
    local rc
    SQUEUE_RAW="$(squeue -u "$WD_USER" -h -o '%i|%j|%T' 2>"$WD_ERR")"; rc=$?
    if [ "$rc" -ne 0 ]; then
        SLURM_OK=0
        SLURM_ERR="squeue exited ${rc}: $(head -1 "$WD_ERR" 2>/dev/null)"
        SQUEUE_RAW=""
    fi
    return 0
}

# Emits "<kind>|<name>|<jobid>|<state>" for live hcgym jobs. kind is eval|train.
# Anything not matching ^hcgym- is dropped here and is invisible to the rest of
# the script — that is the guarantee that other workstreams are never touched.
live_jobs() {
    printf '%s\n' "$SQUEUE_RAW" | while IFS='|' read -r id name state; do
        [ -n "${name:-}" ] || continue
        case "$name" in hcgym-*) ;; *) continue ;; esac
        printf '%s' "$state" | grep -qE "^(${LIVE_STATES})$" || continue
        case "$name" in
            hcgym-eval-*) printf 'eval|%s|%s|%s\n'  "${name#hcgym-eval-}" "$id" "$state" ;;
            *)            printf 'train|%s|%s|%s\n' "${name#hcgym-}"      "$id" "$state" ;;
        esac
    done
}

# ── argv-accurate process inspection ──────────────────────────────────────────
# The only place this script signals anything, so the check is on argv, not on
# the joined command line. A glob over the joined line matched — and would have
# killed — a shell that merely MENTIONED "/autoretry.sh <exp> " in its command.
proc_argv() { # proc_argv <pid> -> one argv element per line
    local f="${WD_PROC}/$1/cmdline"
    [ -r "$f" ] || return 1
    tr '\0' '\n' < "$f"
}

is_autoretry_for() { # is_autoretry_for <pid> <exp>
    local pid="$1" exp="$2" prev="" a rc=1
    while IFS= read -r a; do
        case "$prev" in
            */autoretry.sh|autoretry.sh) [ "$a" = "$exp" ] && { rc=0; break; } ;;
        esac
        prev="$a"
    done < <(proc_argv "$pid" 2>/dev/null)
    return $rc
}

# Never signal ourselves or anything that spawned us.
is_self_or_ancestor() { # is_self_or_ancestor <pid>
    local target="$1" p=$$ ppid guard=0
    while [ "$guard" -lt 64 ]; do
        [ "$p" = "$target" ] && return 0
        [ "$p" -le 1 ] 2>/dev/null && return 1
        ppid="$(sed -n 's/^PPid:[[:space:]]*//p' "${WD_PROC}/${p}/status" 2>/dev/null | head -1)"
        [ -n "$ppid" ] || return 1
        p="$ppid"; guard=$((guard + 1))
    done
    return 1
}

# Experiments with a live autoretry loop. An arm between two submissions has no
# job in squeue but has definitely claimed a slot, so it must count.
live_loops() {
    local pid cmd exp
    pgrep -af "autoretry.sh " 2>/dev/null | while read -r pid cmd; do
        [ "$pid" = "$$" ] && continue
        case "$cmd" in *pgrep*) continue ;; esac
        # printf '%s\n', not '%s': without the newline sed emits an unterminated
        # line and two concurrent loops concatenate into one bogus experiment
        # name ("q4b_grpo" + "q4b_ttopd" -> "q4b_grpoq4b_ttopd"), which then
        # counts as a third occupied slot.
        exp="$(printf '%s\n' "$cmd" \
          | sed -n 's|.*/autoretry\.sh[[:space:]]\{1,\}\([A-Za-z0-9_][A-Za-z0-9_]*\)[[:space:]].*|\1|p')"
        [ -n "$exp" ] || continue
        # pgrep matches the JOINED command line, so an operator shell that merely
        # mentions the loop matches too. Counting that as an occupied slot is not
        # destructive but it is not harmless either: it withholds that arm's slot
        # for as long as the shell lives. When /proc can be read for this pid,
        # require a real argv match. A pid with no /proc entry cannot be a live
        # process holding a slot in the first place, so the fallback is safe.
        if proc_argv "$pid" >/dev/null 2>&1 && ! is_autoretry_for "$pid" "$exp"; then
            continue
        fi
        printf '%s\n' "$exp"
    done
}

# ── per-experiment log lookup ─────────────────────────────────────────────────
# `slurm_hcgym-<stem>_<jobid>.log`. A plain glob of "…-q9b_grpo_*.log" would also
# match "…-q9b_grpo_cosine_60612.log" — the same trailing-token collision
# launch_backbones.sh documents — so the job id is required to be all digits.
latest_log() { # latest_log <stem>   -> path (newest job id), empty if none
    local stem="$1" f base best_id="" best=""
    for f in "${WD_LOGDIR}"/slurm_hcgym-"${stem}"_*.log; do
        [ -e "$f" ] || continue
        base="$(basename "$f")"
        [[ "$base" =~ ^slurm_hcgym-"${stem}"_([0-9]+)\.log$ ]] || continue
        local id="${BASH_REMATCH[1]}"
        if [ -z "$best_id" ] || [ "$id" -gt "$best_id" ]; then best_id="$id"; best="$f"; fi
    done
    printf '%s' "$best"
}

log_jobid() { local b; b="$(basename "${1:-}")"; b="${b##*_}"; printf '%s' "${b%.log}"; }

# sacct, with its failure distinguished from its silence. Sets globals rather
# than echoing, because a command substitution would discard JS_OK.
JS_STATE=""; JS_OK=1; JS_ERR=""
job_state() { # job_state <jobid>
    JS_STATE=""; JS_OK=1; JS_ERR=""
    [ -n "${1:-}" ] || return 0
    if ! command -v sacct >/dev/null 2>&1; then
        JS_OK=0; JS_ERR="sacct is not on PATH"; return 0
    fi
    local out rc
    out="$(sacct -j "$1" -X -P --format=State --noheader 2>"$WD_ERR")"; rc=$?
    if [ "$rc" -ne 0 ]; then
        JS_OK=0; JS_ERR="sacct exited ${rc}: $(head -1 "$WD_ERR" 2>/dev/null)"; return 0
    fi
    JS_STATE="$(printf '%s\n' "$out" | head -1 | awk '{print $1}')"
    return 0
}

# ── failure signatures ────────────────────────────────────────────────────────
# "<extended regex>:::<plain-English reason>". The delimiter is ::: and not | —
# these patterns use | for alternation, and splitting on it truncated every
# multi-alternative pattern to its first branch.
#
# Retrying these is pointless: the next allocation runs the identical command
# against the identical filesystem and reproduces them exactly. The first two are
# not hypothetical — both were taken from logs already in this run root:
#   job 60585  ImportError: cannot import name 'AutoModelForCausalLMWithValueHead' from 'trl'
#   job 60592  eval_benchmark_multiturn.py: error: argument --benchmarks: invalid
#              choice: 'mmlu_college_bio'
#
# The bar for membership is high, because a false positive here parks an arm.
# Deliberately NOT listed:
#   CUDA out of memory  sglang prints OOM during memory profiling on runs that go
#                       on to succeed, so matching it would park healthy arms. A
#                       genuine OOM still fails the job and is caught by the
#                       unknown-signature strike counter instead — slower, safe.
#   AssertionError      far too broad; verl asserts on recoverable conditions too.
FATAL_PATTERNS=(
    "ImportError: cannot import name:::import error — the venv cannot satisfy the code"
    "ModuleNotFoundError: No module named:::missing python module"
    "error: argument [^:]*: invalid choice:::unknown benchmark or CLI value rejected by argparse"
    "^\[fatal\]:::the run script's own preflight refused to continue"
    "FileNotFoundError:::a required input file does not exist"
    "hydra\.errors\.|omegaconf\.errors\.|Could not override|is not in struct:::malformed trainer config"
)

# Worth retrying: the allocation was taken away, not the configuration rejected.
RETRY_PATTERNS=(
    "DUE TO PREEMPTION:::preempted"
    "DUE TO NODE FAILURE:::node failure"
    "DUE TO TIME LIMIT:::hit the 4h wall clock; verl resumes from its checkpoint"
    "CANCELLED AT .*\*\*\*:::killed by slurm mid-step (requeue)"
    "srun: error: .*: task .*: Killed:::task killed by slurm"
)
RETRY_STATES="PREEMPTED|NODE_FAIL|TIMEOUT|REQUEUED|CANCELLED|OUT_OF_MEMORY"

# In-flight view for the current tick. Global (not tick-local) so classify() and
# the launch loops read exactly the same maps.
declare -A LIVE_TRAIN=() LIVE_EVAL=() LIVE_TRAIN_ID=() LIVE_EVAL_ID=() LOOP_ALIVE=()
declare -A TRAIN_STATE=()

# Classify one item. Sets CLS / CLS_WHY / CLS_LOG / CLS_JOB.
#
# Order is the whole point. The queue is consulted BEFORE any log, because a log
# is only evidence about the attempt that wrote it, and a live job means a later
# attempt already exists.
classify() { # classify <kind> <name>
    local kind="$1" name="$2" stem live_id live_desc
    case "$kind" in eval) stem="eval-${name}" ;; *) stem="$name" ;; esac
    CLS=""; CLS_WHY=""; CLS_LOG=""; CLS_JOB=""

    if [ "$kind" = train ] && [ -f "${WD_LOGDIR}/.autoretry_${name}.done" ]; then
        CLS=DONE; CLS_WHY="autoretry wrote .autoretry_${name}.done"; return
    fi

    if [ "$kind" = train ]; then
        live_id="${LIVE_TRAIN_ID[$name]:-}"; live_desc="${LIVE_TRAIN[$name]:-}"
    else
        live_id="${LIVE_EVAL_ID[$name]:-}";  live_desc="${LIVE_EVAL[$name]:-}"
    fi

    # A live slurm job supersedes every log on disk. Do not triage it.
    if [ -n "$live_id" ]; then
        CLS=LIVE; CLS_JOB="$live_id"
        CLS_LOG="$(latest_log "$stem")"
        local older=""
        if [ -n "$CLS_LOG" ]; then
            local lid; lid="$(log_jobid "$CLS_LOG")"
            [ "${lid:-0}" -lt "$live_id" ] 2>/dev/null && \
                older=" (log ${lid} is an earlier attempt and is NOT evidence about this one)"
        fi
        CLS_WHY="${live_desc}${older}"
        return
    fi

    CLS_LOG="$(latest_log "$stem")"
    if [ -z "$CLS_LOG" ]; then
        # No log and no job. A loop with no job is between submissions.
        if [ -n "${LOOP_ALIVE[$name]:-}" ] && [ "$kind" = train ]; then
            CLS=LIVE; CLS_WHY="autoretry loop, between submissions, no log yet"
        else
            CLS=NEW; CLS_WHY="no log yet"
        fi
        return
    fi
    CLS_JOB="$(log_jobid "$CLS_LOG")"

    if grep -qE "^\[done\] (eval )?${name}( |$)" "$CLS_LOG" 2>/dev/null; then
        CLS=DONE; CLS_WHY="completion sentinel in $(basename "$CLS_LOG")"; return
    fi

    job_state "$CLS_JOB"
    if [ "$JS_OK" != 1 ]; then
        # Cannot see the accounting DB. Judging from a partial log here is how a
        # perfectly healthy RUNNING arm collected three strikes and got parked.
        CLS=DEFER
        CLS_WHY="cannot determine job ${CLS_JOB}'s state (${JS_ERR}) — deferring, no strike"
        return
    fi
    local st="$JS_STATE"
    if printf '%s' "$st" | grep -qE "^(${LIVE_STATES})$"; then
        CLS=LIVE; CLS_WHY="job ${CLS_JOB} is ${st} per sacct"; return
    fi
    # sacct is authoritative for "the allocation was taken away". Checked before
    # the log so a stray traceback printed during a preemption teardown cannot be
    # mistaken for a configuration error.
    if printf '%s' "$st" | grep -qE "^(${RETRY_STATES})"; then
        CLS=RETRY; CLS_WHY="job ${CLS_JOB} state=${st} — autoretry owns this"; return
    fi

    local entry pat why
    for entry in "${FATAL_PATTERNS[@]}"; do
        pat="${entry%%:::*}"; why="${entry##*:::}"
        if grep -qE "$pat" "$CLS_LOG" 2>/dev/null; then
            CLS=FATAL
            CLS_WHY="${why}; matched /${pat}/ in $(basename "$CLS_LOG")"
            return
        fi
    done
    for entry in "${RETRY_PATTERNS[@]}"; do
        pat="${entry%%:::*}"; why="${entry##*:::}"
        if grep -qE "$pat" "$CLS_LOG" 2>/dev/null; then
            CLS=RETRY; CLS_WHY="${why} — autoretry owns this"; return
        fi
    done

    if [ "$st" = COMPLETED ]; then
        CLS=DONE; CLS_WHY="job ${CLS_JOB} COMPLETED (no sentinel)"; return
    fi
    CLS=UNKNOWN
    CLS_WHY="job ${CLS_JOB} state=${st:-unrecorded}, no known signature in $(basename "$CLS_LOG")"
}

# ── quarantine ────────────────────────────────────────────────────────────────
qfile()  { printf '%s/.wd_quarantine_%s_%s' "$WD_STATE_DIR" "$1" "$2"; }
sfile()  { printf '%s/.wd_strikes_%s_%s'    "$WD_STATE_DIR" "$1" "$2"; }

quarantined() { [ -f "$(qfile "$1" "$2")" ]; }

# Clear a quarantine when a NEWER attempt exists that no longer shows the
# signature, so a fix un-sticks the matrix without anyone editing state by hand.
quarantine_stale() { # quarantine_stale <kind> <name> <current jobid>
    local q; q="$(qfile "$1" "$2")"
    [ -f "$q" ] || return 1
    local was; was="$(sed -n 's/^jobid=//p' "$q" | head -1)"
    [ -n "${3:-}" ] && [ -n "$was" ] && [ "$3" -gt "$was" ] 2>/dev/null
}

quarantine() { # quarantine <kind> <name> <why> <log> <jobid>
    local kind="$1" name="$2" why="$3" lg="$4" job="$5" q
    q="$(qfile "$kind" "$name")"
    {
        printf 'jobid=%s\n' "$job"
        printf 'when=%s\n'  "$(date '+%F %T')"
        printf 'log=%s\n'   "$lg"
        printf 'why=%s\n'   "$why"
    } > "$q"
    log QUARANTINE "!!!! ${kind}:${name} WILL NOT BE RELAUNCHED :: ${why}"
    log QUARANTINE "     evidence: ${lg}"
    if [ -n "$lg" ] && [ -f "$lg" ]; then
        local entry pat hit
        for entry in "${FATAL_PATTERNS[@]}"; do
            pat="${entry%%:::*}"
            hit="$(grep -oE "$pat.*" "$lg" 2>/dev/null | head -1)"
            [ -n "$hit" ] && { log QUARANTINE "     >>> ${hit}"; break; }
        done
    fi
    log QUARANTINE "     clear it with: rm ${q}"
    [ "$kind" = train ] && stop_loop "$name"
    return 0
}

# SIGTERM the arm's autoretry loop so it stops resubmitting a config error.
# Guarded three ways: WD_ON_FATAL, an ARGV match in /proc (not a substring match
# on the joined command line), and a refusal to signal this process or any of its
# ancestors. No slurm job is cancelled — the loop is a plain user process, and any
# job it already submitted is left to finish or fail on its own.
stop_loop() { # stop_loop <exp>
    local exp="$1" pid
    if [ "$WD_ON_FATAL" != stop ]; then
        log QUARANTINE "     WD_ON_FATAL=${WD_ON_FATAL}: leaving the autoretry loop alone (it will keep resubmitting)"
        return 0
    fi
    for pid in $(pgrep -f "autoretry.sh ${exp} " 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        if ! proc_argv "$pid" >/dev/null 2>&1; then
            log QUARANTINE "     pid ${pid}: cannot read ${WD_PROC}/${pid}/cmdline, refusing to signal"
            continue
        fi
        if ! is_autoretry_for "$pid" "$exp"; then
            log QUARANTINE "     pid ${pid}: argv is not an autoretry.sh invocation for ${exp}, refusing to signal"
            continue
        fi
        if is_self_or_ancestor "$pid"; then
            log QUARANTINE "     pid ${pid}: is this watchdog or one of its ancestors, refusing to signal"
            continue
        fi
        if [ "$WD_DRY_RUN" = 1 ]; then
            log QUARANTINE "     [dry-run] would SIGTERM autoretry loop pid=${pid} for ${exp}"
        else
            kill -TERM "$pid" 2>/dev/null \
              && log QUARANTINE "     stopped autoretry loop pid=${pid} for ${exp}" \
              || log QUARANTINE "     failed to signal pid=${pid}"
        fi
    done
}

bump_strike() { # bump_strike <kind> <name> <jobid> -> echoes new count
    local s; s="$(sfile "$1" "$2")"
    local n=0 was=""
    if [ -f "$s" ]; then n="$(cut -d'|' -f1 "$s")"; was="$(cut -d'|' -f2 "$s")"; fi
    # Count distinct failed job ids, not ticks: polling the same corpse ten times
    # is one failure, and this is what makes repeated ticks idempotent.
    if [ "$was" != "${3:-}" ]; then n=$((n + 1)); printf '%s|%s\n' "$n" "${3:-}" > "$s"; fi
    printf '%s' "$n"
}

# An arm that looks healthy again must not carry its old strikes: without this
# they accumulate over a whole night and one late hiccup parks a recovered arm.
clear_strike() { # clear_strike <kind> <name>
    local s; s="$(sfile "$1" "$2")"
    [ -f "$s" ] || return 0
    rm -f "$s"
    log STRIKE "${1}:${2} strike counter cleared — it is healthy again"
}

# ── the gate ──────────────────────────────────────────────────────────────────
GATE_WHY=""
gate_open() {
    local marker="${WD_STATE_DIR}/.wd_gate_open"
    if [ -f "$marker" ]; then
        GATE_WHY="already proven — $(head -1 "$marker")"; return 0
    fi
    local f base
    for f in "${WD_LOGDIR}"/slurm_hcgym-"${WD_GATE_TAG}"_*.log; do
        [ -e "$f" ] || continue
        base="$(basename "$f")"
        case "$base" in slurm_hcgym-eval-*) continue ;; esac
        if grep -qF "$WD_GATE_STRING" "$f" 2>/dev/null; then
            printf '%s printed "%s" at %s\n' "$base" "$WD_GATE_STRING" "$(date '+%F %T')" > "$marker"
            GATE_WHY="${base} printed \"${WD_GATE_STRING}\" — an optimizer step completed"
            return 0
        fi
    done
    GATE_WHY="no ${WD_GATE_TAG}_* log contains \"${WD_GATE_STRING}\" — no optimizer step has completed yet"
    return 1
}

# ── ask the launchers what they would submit ──────────────────────────────────
# plan_train <selector> [quiet] -> "tag:arm" per line.
# The launcher's own [skip] reasons are echoed into the decision log verbatim, so
# the overnight history shows WHY something was not offered without this script
# having to know. "quiet" suppresses that for the gate's book-keeping count,
# which would otherwise duplicate every skip line.
plan_train() {
    local quiet="${2:-}"
    while IFS= read -r line; do
        case "$line" in
            "[plan] "*) set -- $line; printf '%s\n' "$2" ;;
            "[skip] "*) [ -n "$quiet" ] || log DELEGATE "launch_backbones: ${line#\[skip\] }" >&2 ;;
        esac
    done < <(PLAN=1 "$WD_LAUNCH_BACKBONES" "$1" 2>&1 9>&-)
}

plan_eval() { # plan_eval -> "tag" per line
    while IFS= read -r line; do
        case "$line" in
            "[plan] "*) set -- $line; printf '%s\n' "$2" ;;
            "[skip] "*) log DELEGATE "launch_evals: ${line#\[skip\] }" >&2 ;;
        esac
    done < <(PLAN=1 "$WD_LAUNCH_EVALS" 2>&1 9>&-)
}

# 9>&- everywhere a child is spawned: the flock is held on fd 9, bash does not
# set close-on-exec on it, and launch_backbones.sh nohups 72h autoretry loops.
# Without this the loops inherit the lock and a restarted watchdog is refused
# entry by its own grandchildren until they exit.
run_launcher() { # run_launcher <script> <selector>
    if [ "$WD_DRY_RUN" = 1 ]; then
        log LAUNCH "[dry-run] would run $(basename "$1") $2"
        return 0
    fi
    local out; out="$("$1" "$2" 2>&1 9>&-)"
    printf '%s\n' "$out" | while IFS= read -r l; do
        [ -n "$l" ] && log LAUNCH "  | $l"
    done
}

# GPUs one eval condition costs. launch_evals.sh submits q27b_* with --gres=gpu:4
# and everything else with 1; this must track that case statement.
eval_gpus() { case "$1" in q27b_*) printf 4 ;; *) printf 1 ;; esac; }

# An eval tag that names a training arm must wait for that arm. base* rows are
# untrained prompting conditions and never wait.
eval_needs_training() { case "$1" in base*) return 1 ;; *) return 0 ;; esac; }

# ── one tick ──────────────────────────────────────────────────────────────────
tick() {
    TICK=$((TICK + 1))
    rotate_log
    log BEGIN "────────────────────────────────────────────────────────"

    # 0. Can we see the cluster at all? An empty answer from a BROKEN squeue is
    #    byte-identical to an empty answer from an idle one, and acting on the
    #    second reading when the first is true submits duplicates of jobs that
    #    are already queued. So this is checked before anything else is decided.
    slurm_snapshot
    if [ "$SLURM_OK" != 1 ]; then
        DEGRADED_TICKS=$((DEGRADED_TICKS + 1))
        log DEGRADED "cannot read the queue :: ${SLURM_ERR}"
        log DEGRADED "  triaging nothing and launching nothing this tick — an unreadable queue is"
        log DEGRADED "  indistinguishable from an empty one, and launch_evals.sh dedups on the same"
        log DEGRADED "  call, so acting now would duplicate jobs that are already queued"
        log DEGRADED "  consecutive degraded ticks: ${DEGRADED_TICKS}"
        log END "launched train=0 eval=0, held=0 (degraded)"
        return 0
    fi
    [ "$DEGRADED_TICKS" -gt 0 ] && \
        log RECOVER "queue readable again after ${DEGRADED_TICKS} degraded tick(s)"
    DEGRADED_TICKS=0

    # 1. what is alive right now
    local jobs loops
    jobs="$(live_jobs)"
    loops="$(live_loops)"

    LIVE_TRAIN=(); LIVE_EVAL=(); LIVE_TRAIN_ID=(); LIVE_EVAL_ID=(); LOOP_ALIVE=()
    TRAIN_STATE=()
    local kind name id state exp
    while IFS='|' read -r kind name id state; do
        [ -n "${kind:-}" ] || continue
        case "$kind" in
            train) LIVE_TRAIN["$name"]="job ${id} ${state}"; LIVE_TRAIN_ID["$name"]="$id" ;;
            eval)  LIVE_EVAL["$name"]="job ${id} ${state}";  LIVE_EVAL_ID["$name"]="$id" ;;
        esac
    done <<< "$jobs"
    while read -r exp; do
        [ -n "${exp:-}" ] || continue
        LOOP_ALIVE["$exp"]=1
        # union with squeue, so a loop and its own job are one slot, not two
        [ -n "${LIVE_TRAIN[$exp]:-}" ] || LIVE_TRAIN["$exp"]="autoretry loop, between submissions"
    done <<< "$loops"

    local n_train=${#LIVE_TRAIN[@]} names_t="" names_e="" n_eval=0
    [ "$n_train" -gt 0 ] && names_t="${!LIVE_TRAIN[*]}"
    if [ "${#LIVE_EVAL[@]}" -gt 0 ]; then
        names_e="${!LIVE_EVAL[*]}"
        for name in "${!LIVE_EVAL[@]}"; do n_eval=$((n_eval + $(eval_gpus "$name"))); done
    fi
    log INFLIGHT "train ${n_train}/${WD_MAX_TRAIN} node(s) [${names_t}]  eval ${n_eval}/${WD_MAX_EVAL} gpu(s) [${names_e}]"

    # 2. triage. Three discovery sources, unioned, so nothing falls through a gap:
    #    every slurm log, every .done marker (an arm whose logs were cleaned is
    #    still complete), and every live job (a fresh job has no log yet).
    local f base stem ig skip key items=""
    remember() { case " $items " in *" $1 "*) ;; *) items="${items} $1" ;; esac; }
    for f in "${WD_LOGDIR}"/slurm_hcgym-*.log; do
        [ -e "$f" ] || continue
        base="$(basename "$f")"
        stem="${base#slurm_hcgym-}"; stem="${stem%_*.log}"
        case "$stem" in
            eval-*) remember "eval:${stem#eval-}" ;;
            *)      remember "train:${stem}" ;;
        esac
    done
    for f in "${WD_LOGDIR}"/.autoretry_*.done; do
        [ -e "$f" ] || continue
        base="$(basename "$f")"; base="${base#.autoretry_}"
        remember "train:${base%.done}"
    done
    for name in "${!LIVE_TRAIN[@]}"; do remember "train:${name}"; done
    for name in "${!LIVE_EVAL[@]}";  do remember "eval:${name}";  done

    for key in $items; do
        kind="${key%%:*}"; name="${key#*:}"
        skip=0
        for ig in $WD_IGNORE; do [ "$name" = "$ig" ] && skip=1; done
        [ "$skip" = 1 ] && continue

        classify "$kind" "$name"
        [ "$kind" = train ] && TRAIN_STATE["$name"]="$CLS"
        case "$CLS" in
            LIVE)  log STATUS "${kind}:${name} = LIVE     :: ${CLS_WHY}"; clear_strike "$kind" "$name" ;;
            DONE)  log STATUS "${kind}:${name} = DONE     :: ${CLS_WHY}"; clear_strike "$kind" "$name" ;;
            RETRY) log STATUS "${kind}:${name} = RETRY    :: ${CLS_WHY}"; clear_strike "$kind" "$name" ;;
            DEFER) log STATUS "${kind}:${name} = DEFER    :: ${CLS_WHY}"; continue ;;
            NEW)   log STATUS "${kind}:${name} = NEW      :: ${CLS_WHY}" ;;
            FATAL)
                if quarantined "$kind" "$name"; then
                    log STATUS "${kind}:${name} = FATAL    :: already quarantined, still held"
                else
                    log STATUS "${kind}:${name} = FATAL    :: ${CLS_WHY}"
                    quarantine "$kind" "$name" "$CLS_WHY" "$CLS_LOG" "$CLS_JOB"
                fi
                continue
                ;;
            UNKNOWN)
                local n; n="$(bump_strike "$kind" "$name" "$CLS_JOB")"
                if [ "$n" -ge "$WD_UNKNOWN_STRIKES" ]; then
                    if quarantined "$kind" "$name"; then
                        log STATUS "${kind}:${name} = FATAL    :: already quarantined, still held"
                    else
                        log STATUS "${kind}:${name} = UNKNOWN  :: strike ${n}/${WD_UNKNOWN_STRIKES} — giving up"
                        quarantine "$kind" "$name" \
                            "${WD_UNKNOWN_STRIKES} distinct failures with no recognised signature (last: ${CLS_WHY})" \
                            "$CLS_LOG" "$CLS_JOB"
                    fi
                    continue
                fi
                log STATUS "${kind}:${name} = UNKNOWN  :: strike ${n}/${WD_UNKNOWN_STRIKES}, letting autoretry try again :: ${CLS_WHY}"
                ;;
        esac
        # A newer attempt that is not fatal clears an old quarantine.
        if quarantine_stale "$kind" "$name" "$CLS_JOB"; then
            rm -f "$(qfile "$kind" "$name")"
            clear_strike "$kind" "$name"
            log UNQUARANTINE "${kind}:${name} released :: job ${CLS_JOB} is newer and shows no fatal signature"
        fi
    done

    # 2b. Say out loud, every tick, what is being held — an overnight log that
    #     goes quiet because everything is parked reads the same as one that is
    #     quiet because everything is fine.
    local q qn=0 qlist=""
    for q in "${WD_STATE_DIR}"/.wd_quarantine_*; do
        [ -e "$q" ] || continue
        base="$(basename "$q")"; base="${base#.wd_quarantine_}"
        qn=$((qn + 1)); qlist="${qlist} ${base/_/:}"
    done
    [ "$qn" -gt 0 ] && log HELD "${qn} quarantined, needing a human:${qlist} :: rm ${WD_STATE_DIR}/.wd_quarantine_<kind>_<name> to release"

    # 3. the gate decides which selector the backbone launcher gets
    local selector
    if gate_open; then
        selector=all
        log GATE "OPEN   :: ${GATE_WHY}"
    else
        selector="$WD_GATE_TAG"
        log GATE "CLOSED :: ${GATE_WHY}"
        log GATE "         only ${WD_GATE_TAG} arms may launch; the rest of the matrix is held"
    fi

    # 4. training
    local launched=0 held=0 sel tag arm
    if [ "$n_train" -ge "$WD_MAX_TRAIN" ]; then
        log HOLD "train :: at cap ${n_train}/${WD_MAX_TRAIN} — not asking launch_backbones for more"
    else
        while read -r sel; do
            [ -n "${sel:-}" ] || continue
            tag="${sel%%:*}"; arm="${sel#*:}"
            # Defence in depth against a double submit. launch_backbones.sh dedups
            # on the autoretry loop, which is the right check and normally enough;
            # this covers the window where a loop has died but its job has not,
            # and it costs nothing because the in-flight set is already computed.
            if [ -n "${LIVE_TRAIN[${tag}_${arm}]:-}" ]; then
                log SKIP "train ${sel} :: already in flight (${LIVE_TRAIN[${tag}_${arm}]})"
                held=$((held + 1)); continue
            fi
            if quarantined train "${tag}_${arm}"; then
                log SKIP "train ${sel} :: quarantined — $(sed -n 's/^why=//p' "$(qfile train "${tag}_${arm}")" | head -1)"
                held=$((held + 1)); continue
            fi
            if [ "$((n_train + launched))" -ge "$WD_MAX_TRAIN" ]; then
                log HOLD "train ${sel} :: cap reached ($((n_train + launched))/${WD_MAX_TRAIN})"
                held=$((held + 1)); continue
            fi
            log LAUNCH "train ${sel} :: room ($((n_train + launched))/${WD_MAX_TRAIN}), gate ${selector}, launch_backbones offered it"
            run_launcher "$WD_LAUNCH_BACKBONES" "$sel"
            launched=$((launched + 1))
        done < <(plan_train "$selector")
    fi
    # While the gate is closed, report the size of what is waiting on it.
    if [ "$selector" != all ]; then
        local waiting; waiting="$(plan_train all quiet | grep -cv "^${WD_GATE_TAG}:")"
        log GATE "         ${waiting:-0} non-${WD_GATE_TAG} arm(s) are otherwise launchable and waiting on the gate"
    fi

    # 5. evals — not gated by the validation gate. The gate protects the TRAINING
    #    matrix from fanning out on an unproven stack; an eval needs no training
    #    step, and launch_evals.sh already refuses any condition without a usable
    #    checkpoint. They ARE gated on their own arm having finished training.
    local elaunched=0 egpus=0 w
    if [ "$n_eval" -ge "$WD_MAX_EVAL" ]; then
        log HOLD "eval :: at cap ${n_eval}/${WD_MAX_EVAL} gpu(s) — not asking launch_evals for more"
    else
        while read -r tag; do
            [ -n "${tag:-}" ] || continue
            if [ -n "${LIVE_EVAL[$tag]:-}" ]; then
                log SKIP "eval ${tag} :: already in flight (${LIVE_EVAL[$tag]})"
                held=$((held + 1)); continue
            fi
            if quarantined eval "$tag"; then
                log SKIP "eval ${tag} :: quarantined — $(sed -n 's/^why=//p' "$(qfile eval "$tag")" | head -1)"
                held=$((held + 1)); continue
            fi
            if [ "$WD_EVAL_REQUIRE_DONE" = 1 ] && eval_needs_training "$tag" \
               && [ "${TRAIN_STATE[$tag]:-}" != DONE ]; then
                log HOLD "eval ${tag} :: arm ${tag} is ${TRAIN_STATE[$tag]:-not started}, not DONE — save_freq=10 means a checkpoint exists from step 10, and scoring it now would populate eval_results/${tag} and make launch_evals skip the finished model forever"
                held=$((held + 1)); continue
            fi
            w="$(eval_gpus "$tag")"
            if [ "$w" -gt "$WD_MAX_EVAL" ]; then
                log HOLD "eval ${tag} :: needs ${w} gpu(s) but WD_MAX_EVAL=${WD_MAX_EVAL} — it can NEVER be admitted; raise WD_MAX_EVAL to at least ${w}"
                held=$((held + 1)); continue
            fi
            if [ "$((n_eval + egpus + w))" -gt "$WD_MAX_EVAL" ]; then
                log HOLD "eval ${tag} :: cap reached (${w} more would be $((n_eval + egpus + w))/${WD_MAX_EVAL} gpu(s))"
                held=$((held + 1)); continue
            fi
            log LAUNCH "eval ${tag} :: room ($((n_eval + egpus))+${w}/${WD_MAX_EVAL} gpu(s)), launch_evals offered it"
            run_launcher "$WD_LAUNCH_EVALS" "$tag"
            elaunched=$((elaunched + 1)); egpus=$((egpus + w))
        done < <(plan_eval)
    fi

    log END "launched train=${launched} eval=${elaunched}, held=${held}"
}

# ── main ──────────────────────────────────────────────────────────────────────
log START "pid=$$ user=${WD_USER:-<unresolved>} interval=${WD_INTERVAL}s caps train=${WD_MAX_TRAIN}node eval=${WD_MAX_EVAL}gpu gate=${WD_GATE_TAG} dry_run=${WD_DRY_RUN}"
while :; do
    tick
    [ "$WD_ONCE" = 1 ] && break
    [ "$WD_MAX_TICKS" -gt 0 ] && [ "$TICK" -ge "$WD_MAX_TICKS" ] && break
    sleep "$WD_INTERVAL" 9>&-
done
log EXIT "after ${TICK} tick(s)"
