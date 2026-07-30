#!/bin/bash
# Resolve a verl training run's checkpoint directory to something sglang can load,
# merging FSDP shards into HF format when that has not been done yet.
#
#   ./resolve_ckpt.sh <checkpoints/exp>          # print a loadable path, or fail
#   ./resolve_ckpt.sh <checkpoints/exp> --check  # report only, never merge
#
# WHY THIS EXISTS. verl's default `save_contents: ["model","optimizer","extra"]`
# writes sharded `model_world_size_<N>_rank_<R>.pt` files, plus a `huggingface/`
# subdirectory that holds ONLY the config and tokenizer — no weights. A naive
# "find a config.json" search therefore finds `huggingface/` and hands it to the
# server, which fails to load, or worse silently serves the wrong thing. Checking
# for weights is the whole point; do not simplify this back to a config.json test.
set -uo pipefail

CKPT_ROOT="${1:?usage: resolve_ckpt.sh <checkpoint dir> [--check]}"
MODE="${2:-merge}"
VERL_ROOT=/data/project/private/minstar/workspace/verl_ttopd
PY=/data/project/private/minstar/workspace/hcgym_rebuttal/.venv/bin/python

log() { echo "[resolve] $*" >&2; }

has_weights() {
    local d="$1"
    [ -f "${d}/config.json" ] || return 1
    ls "${d}"/*.safetensors >/dev/null 2>&1 && return 0
    ls "${d}"/pytorch_model*.bin >/dev/null 2>&1 && return 0
    return 1
}

[ -d "$CKPT_ROOT" ] || { log "no such directory: ${CKPT_ROOT}"; exit 1; }

# Newest global_step_* first, so a preempted run evaluates its latest complete step.
mapfile -t STEPS < <(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'global_step_*' \
                     | sed 's/.*global_step_//' | sort -rn | head -20)
if [ "${#STEPS[@]}" -eq 0 ]; then
    if has_weights "$CKPT_ROOT"; then echo "$CKPT_ROOT"; exit 0; fi
    log "no global_step_* under ${CKPT_ROOT} and no weights at its root"
    exit 1
fi

for step in "${STEPS[@]}"; do
    STEP_DIR="${CKPT_ROOT}/global_step_${step}"
    ACTOR="${STEP_DIR}/actor"
    [ -d "$ACTOR" ] || ACTOR="$STEP_DIR"

    # Already merged by a previous call?
    if has_weights "${ACTOR}/huggingface"; then echo "${ACTOR}/huggingface"; exit 0; fi
    if has_weights "${ACTOR}/merged"; then echo "${ACTOR}/merged"; exit 0; fi

    SHARDS=$(ls "${ACTOR}"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
    if [ "$SHARDS" -eq 0 ]; then
        log "step ${step}: no shards and no merged weights, skipping"
        continue
    fi

    if [ "$MODE" = "--check" ]; then
        log "step ${step}: ${SHARDS} shards present, not yet merged (--check, not merging)"
        exit 2
    fi

    log "step ${step}: merging ${SHARDS} FSDP shards -> HF format"
    if "$PY" -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR" \
            --target_dir "${ACTOR}/merged" \
            --trust-remote-code >&2; then
        if has_weights "${ACTOR}/merged"; then echo "${ACTOR}/merged"; exit 0; fi
        log "step ${step}: merge reported success but wrote no weights"
    else
        log "step ${step}: merge failed"
    fi
    # Fall through and try an older step rather than giving up: a step can be
    # half-written when preemption lands mid-save.
done

log "no loadable checkpoint under ${CKPT_ROOT}"
exit 1
