#!/bin/bash
# Mutation check for tests/test_arms.sh.
#
# test_arms.sh passing is only evidence if it would FAIL when the ablation is
# actually broken. This injects each way grpo_cosine could silently stop being a
# clean "GRPO + length control only" arm, and asserts the suite catches it.
#
#   ./tests/mutate.sh
#
# Every mutation is applied to a COPY in a temp dir. The real script is never
# modified. No Slurm job is submitted.
set -uo pipefail
export LC_ALL=C

RUN_ROOT=/data/project/private/minstar/workspace/hcgym_rebuttal
REAL="${RUN_ROOT}/runs/train_hcgym.slurm"
SUITE="${RUN_ROOT}/tests/test_arms.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0

# expect_caught <name> <sed program> <substring the suite must report>
expect_caught() {
    local name="$1" prog="$2" want="$3"
    local mutant="${WORK}/mutant.slurm" out="${WORK}/out.txt"

    sed "$prog" "$REAL" > "$mutant"
    if cmp -s "$mutant" "$REAL"; then
        echo "  ERROR — mutation '${name}' changed nothing (pattern stale)"
        FAIL=$((FAIL + 1)); return
    fi

    HCGYM_TRAIN_SCRIPT="$mutant" bash "$SUITE" > "$out" 2>&1
    local rc=$?

    # The substring must appear on a FAIL line specifically. Matching it
    # anywhere would let a passing "ok — ..." line masquerade as the catch.
    local hit
    hit=$(grep -F "$want" "$out" | grep -m1 'FAIL' | sed 's/^ *//')

    if [ "$rc" -eq 0 ]; then
        echo "  FAIL  — '${name}' was NOT caught (suite still passed)"
        FAIL=$((FAIL + 1))
    elif [ -n "$hit" ]; then
        echo "  ok    — '${name}' caught by: ${hit}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  — '${name}' failed the suite, but not via the expected check"
        grep -m5 'FAIL' "$out" | sed 's/^/           /'
        FAIL=$((FAIL + 1))
    fi
}

echo "=============================================================="
echo "mutation check — does test_arms.sh actually have teeth?"
echo "=============================================================="

# 1. The exact hazard the task calls out: someone retunes the length reward on
#    one arm only. Here ttopd re-exports a different L_max after the shared call.
#    The address range confines the edit to the ttopd branch — without it the
#    edit lands on both arms, which stays symmetric and is not the hazard.
expect_caught "ttopd retunes COSINE_L_MAX behind grpo_cosine's back" \
    '/^    ttopd)$/,/^        ;;$/ s/^        cosine_reward_on$/        cosine_reward_on\n        export COSINE_L_MAX=8192/' \
    "cosine blocks DIFFER"

# 2. grpo_cosine silently acquires a teacher — it would no longer isolate
#    length control, and would just be a second TT-OPD run.
expect_caught "grpo_cosine gains the distillation stack" \
    '/^    grpo_cosine)$/,/^        ;;$/ s/^        distillation_off$/        distillation_on/' \
    "DISTILL_ARGS empty"

# 3. grpo_cosine loses the cosine reward — it would silently be a duplicate of
#    the plain grpo baseline while still being reported as the ablation.
expect_caught "grpo_cosine loses the cosine reward" \
    '/^    grpo_cosine)$/,/^        ;;$/ s/^        cosine_reward_on$/        cosine_reward_off/' \
    "cosine var count"

# 4. The GPU-topology invariant drifts on one arm.
expect_caught "student GPU count drifts" \
    's/^STUDENT_GPUS="\${STUDENT_GPUS:-7}"$/STUDENT_GPUS="${STUDENT_GPUS:-8}"/' \
    "student gpus"

# 5. A second definition site for a cosine value appears (the duplication hazard
#    the single-definition-site check exists to prevent).
expect_caught "a second COSINE_R_EXCEED definition site appears" \
    's/^    export COSINE_R_EXCEED=-0.5$/    export COSINE_R_EXCEED=-0.5\n    export COSINE_R_EXCEED=-0.5/' \
    "COSINE_R_EXCEED assigned exactly once"

# 6. The arm stops being hermetic, so a stale COSINE_REWARD in the submitting
#    shell leaks into the plain grpo baseline via --export=ALL.
expect_caught "grpo stops scrubbing inherited COSINE_* vars" \
    '/^    grpo)$/,/^        ;;$/ s/^        cosine_reward_off$/        :/' \
    "grpo ignores leaked COSINE_"

echo "=============================================================="
echo "  mutations caught ${PASS}   missed ${FAIL}"
echo "=============================================================="
[ "$FAIL" -eq 0 ]
