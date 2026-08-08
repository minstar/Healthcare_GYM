#!/bin/bash
# prune_stale_scratch deletes directories on a shared node. These pin the cases
# where deleting is wrong: a live job's scratch, this job's own scratch across a
# requeue, and -- the one that matters -- an squeue that failed to answer.
set -uo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PASS=0
FAIL=0

check() {
    local label="$1" expect="$2" got="$3"
    if [ "$expect" = "$got" ]; then
        printf '  [PASS] %-52s %s\n' "$label" "$got"
        PASS=$((PASS + 1))
    else
        printf '  [FAIL] %-52s got=%s expect=%s\n' "$label" "$got" "$expect"
        FAIL=$((FAIL + 1))
    fi
}

# Pull the function out of common_env.sh rather than restating it, so the test
# cannot pass against a copy that has drifted from what the jobs actually run.
eval "$(sed -n '/^prune_stale_scratch() {/,/^}/p' "$SELF_DIR/../runs/common_env.sh")"
if ! declare -F prune_stale_scratch >/dev/null; then
    echo "[fatal] could not extract prune_stale_scratch from runs/common_env.sh"
    exit 2
fi

ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT

# squeue is shadowed per-scenario via this file.
SQUEUE_OUT="$ROOT/squeue_out"
SQUEUE_RC="$ROOT/squeue_rc"
squeue() { cat "$SQUEUE_OUT" 2>/dev/null; return "$(cat "$SQUEUE_RC")"; }
export -f squeue 2>/dev/null || true

# Directories are AGED past the grace window by default, because a freshly created
# one is deliberately never pruned -- see scenario 9. A scenario that wants to
# exercise the grace window passes "fresh" as $3.
scenario() {
    rm -rf "${ROOT:?}"/hcgym_* "${ROOT:?}"/keepme
    mkdir -p "$ROOT/hcgym_1001" "$ROOT/hcgym_1002" "$ROOT/hcgym_9999" "$ROOT/keepme"
    if [ "${3:-aged}" != "fresh" ]; then
        touch -d '2 hours ago' "$ROOT"/hcgym_* 2>/dev/null || true
    fi
    printf '%s\n' "$1" > "$SQUEUE_OUT"
    printf '%s' "${2:-0}" > "$SQUEUE_RC"
}

alive() { [ -d "$ROOT/hcgym_$1" ] && echo yes || echo no; }

echo "1. a queued job's scratch survives, a finished job's does not"
scenario "1001"
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "queued job 1001 kept"          yes "$(alive 1001)"
check "finished job 1002 removed"     no  "$(alive 1002)"
check "finished job 9999 removed"     no  "$(alive 9999)"

echo
echo "2. squeue failing must never be read as 'nothing is running'"
scenario "" 1
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "1001 kept when squeue failed"  yes "$(alive 1001)"
check "1002 kept when squeue failed"  yes "$(alive 1002)"
check "9999 kept when squeue failed"  yes "$(alive 9999)"

echo
echo "3. this job's own scratch survives even if squeue has not listed it yet"
scenario "7777"
SLURM_JOB_ID=1002 prune_stale_scratch "$ROOT"
check "own scratch 1002 kept"         yes "$(alive 1002)"
check "unrelated 1001 removed"        no  "$(alive 1001)"

echo
echo "4. array-style ids (12345_3) protect the underlying job"
scenario "1001_3"
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "array parent 1001 kept"        yes "$(alive 1001)"

echo
echo "5. nothing outside the hcgym_<digits> shape is ever touched"
scenario "1001"
mkdir -p "$ROOT/hcgym_notanumber" "$ROOT/hcgym_12ab"
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "non-hcgym sibling untouched"   yes "$([ -d "$ROOT/keepme" ] && echo yes || echo no)"
check "hcgym_notanumber untouched"    yes "$([ -d "$ROOT/hcgym_notanumber" ] && echo yes || echo no)"
check "hcgym_12ab untouched"          yes "$([ -d "$ROOT/hcgym_12ab" ] && echo yes || echo no)"

echo
echo "6. interactive scratch (hcgym_local_<pid>) is never pruned"
scenario "1001"
mkdir -p "$ROOT/hcgym_local_$$"
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "hcgym_local_<pid> kept"        yes "$([ -d "$ROOT/hcgym_local_$$" ] && echo yes || echo no)"

echo
echo "7. the scratch name common_env.sh builds off-Slurm matches that shape"
# Evaluate the real TRITON_HOME expression from common_env.sh in isolation --
# sourcing the whole file here would run prune_stale_scratch against the real /tmp.
_expr=$(sed -n 's/^export TRITON_HOME="\(.*\)"$/\1/p' "$SELF_DIR/../runs/common_env.sh")
_off=$(unset SLURM_JOB_ID; bash -c 'basename "'"$_expr"'"')
_on=$(SLURM_JOB_ID=60769 bash -c 'basename "'"$_expr"'"')
is_prunable() { case "${1##hcgym_}" in *[!0-9]*) echo no ;; *) echo yes ;; esac; }
check "expression was extracted"          yes "$([ -n "$_expr" ] && echo yes || echo no)"
check "off-Slurm scratch ($_off) kept"    no  "$(is_prunable "$_off")"
check "in-job scratch ($_on) prunable"    yes "$(is_prunable "$_on")"

echo
echo "8. a PARTIAL squeue answer must not authorise a prune"
# The dangerous case is not an empty answer, it is a truthful-looking short one.
# squeue emits some ids and then fails; every running job missing from that list
# would otherwise have its scratch deleted underneath it, and ray spills into
# TMPDIR. Reading the exit status off the pipeline would report `sort`'s status,
# which is 0 whatever squeue did.
scenario "1001" 1
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "1002 kept when squeue listed 1001 then failed" yes "$(alive 1002)"
check "9999 kept when squeue listed 1001 then failed" yes "$(alive 9999)"
check "1001 kept too"                                 yes "$(alive 1001)"

echo
echo "9. a freshly created scratch is never pruned, even if squeue omits it"
# A job that starts between the squeue snapshot and this loop is absent from the
# list but has a brand-new directory. Absence from squeue alone must not be enough.
scenario "1001" 0 fresh
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "fresh 1002 kept despite being unlisted" yes "$(alive 1002)"
check "fresh 9999 kept despite being unlisted" yes "$(alive 9999)"
# ...and the same directory, aged, IS collected, so the grace window is not a leak.
scenario "1001"
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "the same 1002, aged, is collected"      no  "$(alive 1002)"

echo
echo "10. a directory named hcgym_ with no id is left alone"
scenario "1001"
mkdir -p "$ROOT/hcgym_"
touch -d '2 hours ago' "$ROOT/hcgym_" 2>/dev/null || true
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "hcgym_ (empty id) untouched" yes "$([ -d "$ROOT/hcgym_" ] && echo yes || echo no)"

echo
echo "11. an empty scratch root is a no-op, not an error"
rm -rf "${ROOT:?}"/hcgym_* "${ROOT:?}"/keepme
printf '1001\n' > "$SQUEUE_OUT"; printf '0' > "$SQUEUE_RC"
SLURM_JOB_ID=1001 prune_stale_scratch "$ROOT"
check "exit status on empty root"     0 "$?"

echo
echo "12. the function must not abort a caller running under 'set -e'"
# Scenarios 1-11 call prune_stale_scratch from THIS script, which sets
# `set -uo pipefail` -- no -e. Every real caller (train_hcgym.slurm and friends)
# sets `set -euo pipefail`, and that difference is not cosmetic: under -e a bare
# `x=$(pipeline)` whose pipeline exits non-zero terminates the script AT THAT
# LINE, so the two guards immediately below each assignment -- the ones that
# exist to handle precisely that failure -- were unreachable.
#
# What that cost: `squeue` answering with an empty list is normal on a login node.
# It killed common_env.sh mid-source, so train_hcgym.slurm's DRY_RUN produced zero
# bytes and exited 1; autoretry.sh read that as "this arm cannot run", printed
# `preflight FAILED`, and refused to submit. Every arm in the campaign silently
# became unresubmittable, with no error naming the real cause.
#
# So this scenario runs the function under the CALLER's flags, in a subshell, and
# only checks that control reaches the line after the call.
# $1 = what squeue prints, $2 = what squeue exits with. The stub runs in a child
# process, so the two values go in as a prefix assignment on `bash` itself, which
# bash exports for the duration of that command.
CHILD="$ROOT/under_set_e.sh"
cat > "$CHILD" <<'CHILD_EOF'
set -euo pipefail
squeue() { printf '%s' "$FAKE_OUT"; return "$FAKE_RC"; }
eval "$(sed -n '/^prune_stale_scratch() {/,/^}/p' "$1")"
prune_stale_scratch "$2"
echo SURVIVED
CHILD_EOF
run_under_set_e() {
    FAKE_OUT="$1" FAKE_RC="$2" bash "$CHILD" \
        "$SELF_DIR/../runs/common_env.sh" "$ROOT" 2>/dev/null
}
mkdir -p "$ROOT/hcgym_1001"
check "survives an EMPTY squeue answer"      SURVIVED "$(run_under_set_e ''            0)"
check "survives a squeue that FAILED"        SURVIVED "$(run_under_set_e ''            1)"
check "survives a squeue with no numeric id" SURVIVED "$(run_under_set_e 'CLUSTER_DOWN' 0)"
check "still works on a normal answer"       SURVIVED "$(run_under_set_e '1001'        0)"

echo
echo "================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "ALL $PASS CHECKS PASSED"
    exit 0
fi
echo "$FAIL of $((PASS + FAIL)) CHECKS FAILED"
exit 1
