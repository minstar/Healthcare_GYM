#!/bin/bash
# Keep one hcgym training arm alive across preemption.
#
# pt2_preemptible preempts with REQUEUE and GraceTime=0, so a run gets killed
# mid-step with no warning. verl resumes from its latest checkpoint, so the only
# thing missing is something to resubmit. That is this.
#
#   ./autoretry.sh <EXP> <BACKBONE> <ARM> [extra --export k=v pairs...]
#
#   SCRIPT=<path> ./autoretry.sh ...   drive a different sbatch template, e.g.
#                                      runs/train_ttopd_hints.slurm or runs/train_opsd.slurm
#
# Stops when the job's log prints the "[done] <EXP>" sentinel, when the retry
# budget is spent, or at the deadline.
set -uo pipefail

EXP="${1:?usage: autoretry.sh EXP BACKBONE ARM [EXTRA_EXPORTS]}"
BACKBONE="${2:?}"
ARM="${3:?}"
EXTRA="${4:-}"

# Overridable so the harness can run against another checkout or a fixture tree;
# the default is the layout every run in the paper used.
RUN_ROOT="${RUN_ROOT:-/data/project/private/minstar/workspace/hcgym_rebuttal}"
# Which sbatch template to drive. train_hcgym.slurm covers grpo / grpo_cosine /
# ttopd; the ttopd_hints and opsd arms live in their own files and need the same
# preemption resilience, so allow an override rather than a second copy of this
# loop. ARM is still exported for all of them — the standalone templates ignore it.
SCRIPT="${SCRIPT:-${RUN_ROOT}/runs/train_hcgym.slurm}"
if [ ! -f "$SCRIPT" ]; then
    echo "[autoretry] no such sbatch template: $SCRIPT" >&2
    exit 2
fi

# Preflight: resolve the arm against THIS template before entering the retry loop.
#
# SCRIPT is an environment override, so `./autoretry.sh q9b_star <bb> star
# "SCRIPT=..."` silently passes the path as the extra-vars string and leaves the
# default template in place. train_hcgym.slurm then fatals on ARM=star -- correctly
# -- and the loop resubmits that same fatal 60 times. Every template honours
# DRY_RUN and exits in seconds without touching a GPU, so ask it once, here.
#
# PREFLIGHT=0 skips this, for a template whose dry run needs something absent.
if [ "${PREFLIGHT:-1}" = "1" ]; then
    if ! _pf=$(DRY_RUN=1 BACKBONE="$BACKBONE" ARM="$ARM" EXP="$EXP" timeout 90 bash "$SCRIPT" 2>&1); then
        echo "[autoretry] preflight FAILED — $(basename "$SCRIPT") cannot run ARM=${ARM}." >&2
        echo "$_pf" | grep -E "^\[fatal\]|^\[warn\]" | head -5 >&2
        echo "[autoretry] refusing to submit; nothing has been queued." >&2
        exit 2
    fi
    echo "[autoretry] preflight ok: $(basename "$SCRIPT") resolves ARM=${ARM}"
fi
STATE="${RUN_ROOT}/logs/.autoretry_${EXP}"
LOGDIR="${RUN_ROOT}/logs"
MAX_RETRIES="${MAX_RETRIES:-60}"
DEADLINE=$(( $(date +%s) + ${DEADLINE_HOURS:-72} * 3600 ))
POLL="${POLL:-180}"

# Under cron/systemd/a container there is no login shell and USER is unset, which
# under `set -u` would kill the squeue check below inside a command substitution.
SQ_USER="${USER:-$(id -un 2>/dev/null || true)}"

mkdir -p "$LOGDIR"
echo "[autoretry] exp=${EXP} arm=${ARM} budget=${MAX_RETRIES} deadline=$(date -d @${DEADLINE} '+%F %T')"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if [ -f "${STATE}.done" ]; then
        echo "[autoretry] ${EXP} already complete"; exit 0
    fi

    # Anything queued or running for this experiment means there is nothing to do.
    #
    # FAIL CLOSED. A squeue that cannot reach the controller returns an empty
    # string, exactly like a squeue that reached it and found nothing — and this
    # check is the only thing standing between a poll and a duplicate submission.
    # Two of these running the same EXP write one checkpoint directory from two
    # --exclusive nodes of a six-node slice. So the exit status is checked, and an
    # unreadable queue means wait, not submit.
    if ! QSTATE="$(squeue -u "$SQ_USER" -h -n "hcgym-${EXP}" -o %T 2>&1)"; then
        echo "[autoretry] ${EXP} cannot read the queue (${QSTATE%%$'\n'*}); waiting rather than risking a duplicate"
        sleep "$POLL"; continue
    fi
    if printf '%s' "$QSTATE" | grep -qE "RUNNING|PENDING|COMPLETING|CONFIGURING|SUSPENDED"; then
        sleep "$POLL"; continue
    fi

    LATEST=$(ls -t "${LOGDIR}"/slurm_hcgym-${EXP}_*.log 2>/dev/null | head -1)
    if [ -n "${LATEST:-}" ] && grep -q "^\[done\] ${EXP}" "$LATEST"; then
        touch "${STATE}.done"
        echo "[autoretry] ${EXP} finished — sentinel found in $(basename "$LATEST")"
        exit 0
    fi

    N=$(cat "${STATE}.count" 2>/dev/null || echo 0)
    if [ "$N" -ge "$MAX_RETRIES" ]; then
        echo "[autoretry] ${EXP} retry budget exhausted after ${N} submissions"; exit 1
    fi
    echo $((N + 1)) > "${STATE}.count"

    echo "[autoretry] submitting ${EXP} (attempt $((N + 1)))"
    sbatch -J "hcgym-${EXP}" \
           --export="ALL,BACKBONE=${BACKBONE},ARM=${ARM},EXP=${EXP}${EXTRA:+,${EXTRA}}" \
           "$SCRIPT" 2>&1 | tee -a "${LOGDIR}/autoretry_${EXP}.log"
    sleep "$POLL"
done

echo "[autoretry] ${EXP} hit the deadline without completing"
exit 1
