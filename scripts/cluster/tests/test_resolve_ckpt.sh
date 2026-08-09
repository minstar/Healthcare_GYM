#!/bin/bash
# resolve_ckpt.sh decides WHICH weights get served. Getting it wrong does not
# crash -- it reports a number under the wrong step's label, which is worse.
#
# Two behaviours are pinned here:
#
#   * a run root resolves to its NEWEST step (right for a preempted run that is
#     still training), and
#   * a named `global_step_N` resolves to THAT step and never substitutes a
#     neighbour (right for evaluation, where the step is chosen from a measured
#     validation curve -- q9b_grpo peaks at 670 and trains on to 1452, so
#     "latest" and "best" are 782 steps apart).
#
# Every scenario runs in --check mode, which stops before the merge, so no test
# here loads a model or needs a GPU.
#
# Run:  bash tests/test_resolve_ckpt.sh
set -uo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESOLVE="$SELF_DIR/../runs/resolve_ckpt.sh"
PASS=0
FAIL=0

check() {
    local label="$1" expect="$2" got="$3"
    if [ "$expect" = "$got" ]; then
        printf '  [PASS] %-54s %s\n' "$label" "$got"
        PASS=$((PASS + 1))
    else
        printf '  [FAIL] %-54s got=%s expect=%s\n' "$label" "$got" "$expect"
        FAIL=$((FAIL + 1))
    fi
}

ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT

# A step with FSDP shards and the weightless huggingface/ dir verl really writes.
# The weightless config is the whole reason this script exists: a naive
# "find a config.json" search would hand it straight to the server.
make_sharded() {
    local d="$1/actor"
    mkdir -p "$d/huggingface"
    for r in 0 1 2 3 4 5 6; do : > "$d/model_world_size_7_rank_${r}.pt"; done
    echo '{}' > "$d/huggingface/config.json"
}
# A step someone has already merged.
make_merged() {
    local d="$1/actor/merged"
    mkdir -p "$d"
    echo '{}' > "$d/config.json"
    : > "$d/model-00001-of-00001.safetensors"
}
# A step that exists but holds nothing loadable -- what a preemption mid-save leaves.
make_empty() { mkdir -p "$1/actor"; }

# stdout is the resolved path; stderr is the log. Keep them apart: the log names
# the step it chose, which is what several scenarios below actually assert on.
run() {  # $1.. = args; sets OUT, ERR, RC
    OUT=$("$RESOLVE" "$@" 2>"$ROOT/err"); RC=$?
    ERR=$(cat "$ROOT/err")
}
# The step it SETTLED on, not the first one it mentioned. Scenario 5 skips a
# half-written newest step and logs that skip before it logs the step it chose,
# so keying on the first "step N" in the log reads back the rejected one.
chose_step() { grep 'shards present' <<< "$ERR" | grep -oE 'step [0-9]+' | tail -1 | awk '{print $2}'; }

echo "1. a run root resolves to its NEWEST step"
RUN="$ROOT/run_a"
for s in 40 670 1452; do make_sharded "$RUN/global_step_${s}"; done
run "$RUN" --check
check "exit 2 (shards present, not merged)"  2    "$RC"
check "chose the newest step"                1452 "$(chose_step)"

echo
echo "2. a named step resolves to THAT step, not the newest"
run "$RUN/global_step_670" --check
check "exit 2"                               2   "$RC"
check "chose the named step"                 670 "$(chose_step)"

echo
echo "3. a named step that is already merged returns its path directly"
make_merged "$RUN/global_step_670"
run "$RUN/global_step_670" --check
check "exit 0"                                          0     "$RC"
check "returned the merged dir"                         merged "$(basename "$OUT")"
check "and it is under the step that was asked for"     yes   "$([[ "$OUT" == *global_step_670* ]] && echo yes || echo no)"

echo
echo "4. a named step with nothing loadable FAILS — it must not substitute a sibling"
# This is the regression that matters. run_b's newest step is empty and step 40
# is fully populated, so a fallback would silently answer with step 40 while the
# caller believes it is reading step 990.
RUN_B="$ROOT/run_b"
make_sharded "$RUN_B/global_step_40"
make_empty   "$RUN_B/global_step_990"
run "$RUN_B/global_step_990" --check
check "exit is non-zero"                     yes "$([ "$RC" -ne 0 ] && echo yes || echo no)"
check "stdout is empty (no path returned)"   yes "$([ -z "$OUT" ] && echo yes || echo no)"
check "step 40 was NOT substituted"          yes "$([[ "$OUT" != *global_step_40* ]] && echo yes || echo no)"
check "the message names the step asked for" yes "$(grep -q 'step 990' <<< "$ERR" && echo yes || echo no)"

echo
echo "5. a run root still falls back past an unusable newest step"
# The same layout as scenario 4, addressed as a run root instead. Here falling
# back IS correct: nobody named a step, and a half-written newest step is exactly
# what preemption mid-save leaves behind.
run "$RUN_B" --check
check "exit 2"                               2  "$RC"
check "fell back to the older usable step"   40 "$(chose_step)"

echo
echo "6. a weightless huggingface/ is never accepted as loadable"
# verl writes huggingface/ with config+tokenizer and no weights. If this were
# accepted, sglang would load a config with no model behind it.
RUN_C="$ROOT/run_c"
make_sharded "$RUN_C/global_step_10"
run "$RUN_C/global_step_10" --check
check "did not return the weightless hf dir" yes "$([ "$OUT" != *huggingface ] && echo yes || echo no)"
check "reported shards instead"              2   "$RC"

echo
echo "7. a run root with no steps and no weights fails cleanly"
mkdir -p "$ROOT/run_d"
run "$ROOT/run_d" --check
check "exit 1"                               1   "$RC"
check "stdout empty"                         yes "$([ -z "$OUT" ] && echo yes || echo no)"

echo
echo "8. a nonexistent path fails without touching anything"
run "$ROOT/nope" --check
check "exit 1"                               1   "$RC"
check "says which path"                      yes "$(grep -q 'no such directory' <<< "$ERR" && echo yes || echo no)"

echo
echo "9. a merged dir whose config MISDECLARES its dtype is rebuilt, not served"
# The failure this scenario exists for: verl's merger writes bf16 weights but
# records dtype float32 in config.json (transformers 5 renamed from_config's
# `torch_dtype` argument, so the bf16 request is dropped). sglang sizes the
# Qwen3.5 Mamba conv-state cache from the DECLARED dtype, loads the model onto
# the GPU, and only then aborts in causal_conv1d_fwd. Fifteen jobs died that way,
# and an earlier version of this check looked only at the tensors, so it passed
# the broken directory through a second time.
#
# Real safetensors here, not the empty placeholder the scenarios above use --
# the check reads headers, and a header it cannot parse is deliberately treated
# as "no opinion" rather than "wrong".
PY_BIN="$SELF_DIR/../.venv/bin/python"
make_real_merged() {   # $1 = step dir, $2 = dtype to DECLARE in config.json
    local d="$1/actor/merged"
    mkdir -p "$d"
    "$PY_BIN" - "$d" "$2" <<'PY' 2>/dev/null
import json, sys, torch
from safetensors.torch import save_file
d, declared = sys.argv[1], sys.argv[2]
save_file({"w": torch.zeros(4, dtype=torch.bfloat16)}, f"{d}/model.safetensors")
json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"],
           "dtype": declared, "text_config": {"dtype": declared}},
          open(f"{d}/config.json", "w"))
PY
}

if [ ! -x "$PY_BIN" ]; then
    echo "  SKIP — no venv python at $PY_BIN"
elif ! "$PY_BIN" -c "import safetensors, torch" 2>/dev/null; then
    echo "  SKIP — safetensors/torch unavailable in the venv"
else
    RUN_E="$ROOT/run_e"
    make_sharded "$RUN_E/global_step_50"          # shards, so a rebuild is possible
    make_real_merged "$RUN_E/global_step_50" float32
    run "$RUN_E/global_step_50" --check
    check "misdeclared dtype is not served"     2   "$RC"
    check "stdout empty (no path handed back)"  yes "$([ -z "$OUT" ] && echo yes || echo no)"
    check "the message says what is wrong"      yes "$(grep -q 'misdeclares dtype' <<< "$ERR" && echo yes || echo no)"
    check "--check left the directory alone"    yes "$([ -d "$RUN_E/global_step_50/actor/merged" ] && echo yes || echo no)"

    # ...and the same directory with a truthful config is served immediately.
    make_real_merged "$RUN_E/global_step_50" bfloat16
    run "$RUN_E/global_step_50" --check
    check "truthful dtype is served"            0      "$RC"
    check "and it returns the merged dir"       merged "$(basename "$OUT")"
fi

echo
echo "================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "ALL $PASS CHECKS PASSED"
    exit 0
fi
echo "$FAIL of $((PASS + FAIL)) CHECKS FAILED"
exit 1
