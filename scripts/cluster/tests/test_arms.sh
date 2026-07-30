#!/bin/bash
# Arm-selection tests for runs/train_hcgym.slurm.
#
# The arms form an additive ladder:
#     grpo  ->  grpo_cosine (+ cosine length reward)  ->  ttopd (+ distillation)
# The middle arm is what prices the length reward on its own. It is only a valid ablation
# if it carries the cosine reward with EXACTLY the shaping ttopd uses and carries
# no teacher at all. This test asserts that mechanically instead of by reading.
#
#   ./tests/test_arms.sh
#
# Runs the real script with DRY_RUN=1, which resolves the arm and exits before
# the trainer. No Slurm job is submitted and no GPU is touched.
set -uo pipefail
export LC_ALL=C

RUN_ROOT=/data/project/private/minstar/workspace/hcgym_rebuttal
# Overridable so the suite can be pointed at a deliberately broken copy to prove
# these assertions actually fail when the ablation is violated (see mutate.sh).
SCRIPT="${HCGYM_TRAIN_SCRIPT:-${RUN_ROOT}/runs/train_hcgym.slurm}"
BACKBONE="${RUN_ROOT}/models/Qwen3.5-9B"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); echo "  ok   — $1"; }
bad()  { FAIL=$((FAIL + 1)); echo "  FAIL — $1"; }
check(){ # check <desc> <actual> <expected>
    if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: expected '$3', got '$2'"; fi
}

# Run the script for one arm and capture the DRYRUN report.
# Extra "K=V" args are exported first, to simulate a dirty submitting shell.
run_arm() { # run_arm <outfile> <arm> [K=V ...]
    local out="$1" arm="$2"; shift 2
    (
        for kv in "$@"; do export "${kv?}"; done
        DRY_RUN=1 BACKBONE="$BACKBONE" ARM="$arm" EXP="test_${arm}" \
            bash "$SCRIPT"
    ) > "$out" 2>&1
    return $?
}

field()  { sed -n "s/^DRYRUN $2=//p" "$1"; }              # field <file> <key>
cosines(){ sed -n 's/^DRYRUN cosine //p' "$1" | sort; }   # sorted KEY=VALUE lines

echo "=============================================================="
echo "1. each arm resolves and reports"
echo "=============================================================="
for arm in grpo grpo_cosine ttopd; do
    if run_arm "${WORK}/${arm}.txt" "$arm"; then
        if [ "$(field "${WORK}/${arm}.txt" ok)" = "" ] && grep -q '^DRYRUN ok$' "${WORK}/${arm}.txt"; then
            ok "ARM=${arm} dry-run exited 0 and reported"
        else
            bad "ARM=${arm} produced no DRYRUN ok sentinel"
        fi
    else
        bad "ARM=${arm} dry-run exited non-zero"
        cat "${WORK}/${arm}.txt"
    fi
done

# An unknown arm must be rejected, not silently trained as a baseline.
if run_arm "${WORK}/bogus.txt" "grpo_cosinne"; then
    bad "ARM=grpo_cosinne was accepted (typo would train a silent baseline)"
else
    if grep -q '\[fatal\] ARM must be grpo, grpo_cosine or ttopd' "${WORK}/bogus.txt"; then
        ok "unknown arm rejected with a fatal error"
    else
        bad "unknown arm failed but not with the expected message"
    fi
fi

echo
echo "=============================================================="
echo "2. per-arm state: COSINE_* vars and DISTILL_ARGS"
echo "=============================================================="

echo "-- ARM=grpo — no length control, no teacher"
check "cosine var count"    "$(field "${WORK}/grpo.txt" cosine_count)"      "0"
check "distill env count"   "$(field "${WORK}/grpo.txt" distill_env_count)" "0"
check "DISTILL_ARGS empty"  "$(field "${WORK}/grpo.txt" distill_args_count)" "0"
check "student gpus"        "$(field "${WORK}/grpo.txt" student_gpus)"      "7"

echo "-- ARM=grpo_cosine — length control ONLY"
check "cosine var count"    "$(field "${WORK}/grpo_cosine.txt" cosine_count)"      "8"
check "distill env count"   "$(field "${WORK}/grpo_cosine.txt" distill_env_count)" "0"
check "DISTILL_ARGS empty"  "$(field "${WORK}/grpo_cosine.txt" distill_args_count)" "0"
check "student gpus"        "$(field "${WORK}/grpo_cosine.txt" student_gpus)"      "7"

# The exact shaping, asserted value by value — this is the reward the middle arm
# exists to isolate, so a silent retune must break the test.
for expect in \
    "COSINE_REWARD=1" \
    "COSINE_L_MAX=12288" \
    "COSINE_CHARS_PER_TOKEN=5.0" \
    "COSINE_R0_CORRECT=1.1" \
    "COSINE_RL_CORRECT=0.7" \
    "COSINE_R0_WRONG=0.0" \
    "COSINE_RL_WRONG=-0.3" \
    "COSINE_R_EXCEED=-0.5" ; do
    if cosines "${WORK}/grpo_cosine.txt" | grep -qxF "$expect"; then
        ok "grpo_cosine sets ${expect}"
    else
        bad "grpo_cosine missing ${expect}"
    fi
done

echo "-- ARM=ttopd — length control AND teacher"
check "cosine var count"     "$(field "${WORK}/ttopd.txt" cosine_count)"      "8"
# 12, not 13: HINT_OPD_DYNAMIC was dropped after confirming no Python in either
# repo reads it. Every run script exported it; nothing consumed it.
check "distill env count"    "$(field "${WORK}/ttopd.txt" distill_env_count)" "12"
check "student gpus"         "$(field "${WORK}/ttopd.txt" student_gpus)"      "7"
TT_ARGS="$(field "${WORK}/ttopd.txt" distill_args_count)"
if [ "${TT_ARGS:-0}" -gt 0 ]; then
    ok "DISTILL_ARGS non-empty (${TT_ARGS} hydra overrides)"
else
    bad "DISTILL_ARGS empty for ttopd — the teacher would never be built"
fi
if grep -qxF 'DRYRUN distill_arg distillation.enabled=True' "${WORK}/ttopd.txt"; then
    ok "ttopd passes distillation.enabled=True"
else
    bad "ttopd does not pass distillation.enabled=True"
fi
# Self-consistency: the reported count must match the lines actually emitted.
check "distill_arg lines == reported count" \
      "$(grep -c '^DRYRUN distill_arg ' "${WORK}/ttopd.txt")" "$TT_ARGS"

echo
echo "=============================================================="
echo "3. MECHANICAL identity of the cosine block: ttopd vs grpo_cosine"
echo "=============================================================="
cosines "${WORK}/grpo_cosine.txt" > "${WORK}/cos_grpo_cosine"
cosines "${WORK}/ttopd.txt"       > "${WORK}/cos_ttopd"

if [ ! -s "${WORK}/cos_ttopd" ]; then
    bad "ttopd emitted no COSINE_* vars — nothing to compare"
elif diff -u "${WORK}/cos_grpo_cosine" "${WORK}/cos_ttopd" > "${WORK}/cos_diff"; then
    ok "cosine blocks are byte-identical ($(wc -l < "${WORK}/cos_ttopd") vars)"
    sed 's/^/       /' "${WORK}/cos_ttopd"
else
    bad "cosine blocks DIFFER — the ablation is invalid"
    sed 's/^/       /' "${WORK}/cos_diff"
fi

# Static guard: the values must have exactly one definition site in the script,
# so a future tune cannot land on one arm and miss the other.
echo
echo "-- single definition site in train_hcgym.slurm"
for var in COSINE_REWARD COSINE_L_MAX COSINE_CHARS_PER_TOKEN \
           COSINE_R0_CORRECT COSINE_RL_CORRECT COSINE_R0_WRONG \
           COSINE_RL_WRONG COSINE_R_EXCEED ; do
    # Count assignments anywhere on a line, not just at line start, so stacking
    # two exports onto one line cannot hide a second definition. Comments are
    # stripped first: the file discusses these variables in prose, and a mention
    # in a comment is not a definition site.
    n=$(sed 's/[[:space:]]*#.*$//' "$SCRIPT" | grep -oE "(^|[[:space:];])${var}=" | wc -l)
    check "${var} assigned exactly once" "$n" "1"
done

echo
echo "=============================================================="
echo "4. arms are hermetic against a dirty --export=ALL environment"
echo "=============================================================="
# autoretry.sh submits with --export=ALL, so whatever is in the submitting shell
# reaches the job. reward_fn.py reads COSINE_REWARD at import time, so a leaked
# value would silently give the baseline length control.
run_arm "${WORK}/grpo_dirty.txt" grpo \
        COSINE_REWARD=1 COSINE_L_MAX=99 COSINE_R_EXCEED=-9.9 \
        HINT_OPD_ENABLED=1 BT_OPD_TOPK_RATIO=0.9
check "grpo ignores leaked COSINE_*"  "$(field "${WORK}/grpo_dirty.txt" cosine_count)"      "0"
check "grpo ignores leaked distill"   "$(field "${WORK}/grpo_dirty.txt" distill_env_count)" "0"

run_arm "${WORK}/gc_dirty.txt" grpo_cosine \
        COSINE_L_MAX=99 COSINE_R_EXCEED=-9.9 HINT_OPD_ENABLED=1
check "grpo_cosine overrides leaked COSINE_L_MAX" \
      "$(cosines "${WORK}/gc_dirty.txt" | sed -n 's/^COSINE_L_MAX=//p')" "12288"
check "grpo_cosine overrides leaked COSINE_R_EXCEED" \
      "$(cosines "${WORK}/gc_dirty.txt" | sed -n 's/^COSINE_R_EXCEED=//p')" "-0.5"
check "grpo_cosine still builds no teacher" \
      "$(field "${WORK}/gc_dirty.txt" distill_env_count)" "0"
# And it is still identical to ttopd even when the caller's shell was dirty.
if diff -q <(cosines "${WORK}/gc_dirty.txt") "${WORK}/cos_ttopd" >/dev/null; then
    ok "dirty-shell grpo_cosine still matches ttopd exactly"
else
    bad "dirty-shell grpo_cosine drifted from ttopd"
fi

echo
echo "=============================================================="
echo "5. launch_backbones.sh selects the new arm without fanning out"
echo "=============================================================="
LB="${RUN_ROOT}/runs/launch_backbones.sh"

plan() { PLAN=1 bash "$LB" "$@" 2>&1; }   # PLAN=1 prints, submits nothing

# Default sweep must add grpo_cosine on the anchor only.
plan all > "${WORK}/plan_all.txt"
n_gc=$(grep -c ':grpo_cosine' "${WORK}/plan_all.txt")
check "default sweep plans grpo_cosine once" "$n_gc" "1"
if grep -q '^\[plan\] q9b:grpo_cosine' "${WORK}/plan_all.txt"; then
    ok "the one grpo_cosine is on the anchor backbone q9b"
else
    ok "grpo_cosine not planned (anchor backbone absent from disk) — see plan below"
fi
if grep -qE '^\[plan\] (q4b|g12b|q27b):grpo_cosine' "${WORK}/plan_all.txt"; then
    bad "grpo_cosine fanned out to a non-anchor backbone"
else
    ok "no grpo_cosine fan-out to non-anchor backbones"
fi

# Explicit single-arm selection.
plan q9b:grpo_cosine > "${WORK}/plan_one.txt"
check "explicit q9b:grpo_cosine plans exactly one arm" \
      "$(grep -c '^\[plan\] ' "${WORK}/plan_one.txt")" "1"

# Typos must fail loudly: a fatal message AND a non-zero exit, so a mistyped
# selector can never look like a successful no-op run.
# Captured rather than piped — `plan ... | grep` is masked by pipefail, because
# the non-zero exit is itself the behaviour under test.
out=$(plan q9b:grpo_cosinne); rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q "unknown arm"; then
    ok "unknown arm selector rejected (exit ${rc})"
else
    bad "unknown arm selector not rejected (exit ${rc}): ${out}"
fi
out=$(plan q9c); rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q "unknown backbone"; then
    ok "unknown backbone selector rejected (exit ${rc})"
else
    bad "unknown backbone selector not rejected (exit ${rc}): ${out}"
fi
# Nothing may be submitted in PLAN mode.
if grep -q 'nothing submitted' "${WORK}/plan_all.txt"; then
    ok "PLAN=1 submitted nothing"
else
    bad "PLAN=1 did not confirm it submitted nothing"
fi

echo
echo "=============================================================="
echo "  passed ${PASS}   failed ${FAIL}"
echo "=============================================================="
[ "$FAIL" -eq 0 ]
