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

# Does a merged directory carry the dtypes the architecture needs?
#
# verl's FSDP merger used to cast every tensor to bf16. FSDP keeps master weights
# in fp32, so the shards look uniform and carry no record of what the model
# wanted -- and Qwen3.5 wants `linear_attn.A_log` and `linear_attn.norm.weight`
# in fp32, because sglang sizes the Mamba conv-state cache from them. A bf16
# merge loads, then dies in the kernel with
#   Expected conv_states_.scalar_type() == input_type to be true, but got false
# after the model is already on the GPU. That is a slow, expensive way to find out.
#
# So compare against the reference before serving. No DTYPE_REFERENCE means no
# opinion: accept whatever is there, as before.
dtypes_ok() {
    local merged="$1" ref="${DTYPE_REFERENCE:-}"
    # The config check needs no reference; the tensor check does.
    # stdout is this script's return channel -- it carries the resolved path and
    # nothing else. The probe's diagnostics go to stderr; its own stdout is
    # discarded so a stray print can never be mistaken for a model directory.
    "$PY" - "$merged" "$ref" >/dev/null <<'EOF'
import sys, glob, json, collections, os
from safetensors import safe_open

NAME = {"BF16": "bfloat16", "F32": "float32", "F16": "float16", "F64": "float64"}

def dtypes(d):
    """Header-only dtype map. An unreadable file yields no opinion.

    A truncated or corrupt safetensors is a different problem than a wrong
    dtype, and re-merging on top of it would hide it. Returning {} lets the
    caller fall through to the loader, which reports it properly.
    """
    out = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        try:
            with safe_open(f, framework="pt") as g:
                for k in g.keys():
                    out[k] = g.get_slice(k).get_dtype()
        except Exception:  # noqa: BLE001
            return {}
    return out

merged_dir, ref_dir = sys.argv[1], sys.argv[2]
merged = dtypes(merged_dir)
if not merged:
    sys.exit(0)                     # nothing to inspect; do not block

# 1. config.json must DECLARE the dtype the weights actually are. sglang sizes
#    the Qwen3.5 Mamba conv-state cache from the declared value, so a config
#    saying float32 over bf16 weights loads onto the GPU and then aborts in
#    causal_conv1d_fwd. The declaration is what broke fifteen jobs, and only the
#    tensors were being checked here, which is why it got through twice.
declared_wrong = []
cfg_path = os.path.join(merged_dir, "config.json")
if os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path))
    dominant = NAME.get(collections.Counter(merged.values()).most_common(1)[0][0])
    for scope in ("", "text_config", "vision_config"):
        node = cfg.get(scope) if scope else cfg
        if not isinstance(node, dict):
            continue
        got = node.get("dtype", node.get("torch_dtype"))
        if got is not None and dominant is not None and got != dominant:
            declared_wrong.append(f"{scope or '<root>'}.dtype={got} but the weights are {dominant}")
if declared_wrong:
    print("config.json misdeclares dtype: " + "; ".join(declared_wrong), file=sys.stderr)
    sys.exit(1)

# 2. individual tensors must match the reference, where one is given.
if not ref_dir:
    sys.exit(0)
ref = dtypes(ref_dir)
if not ref:
    sys.exit(0)
bad = [k for k, v in ref.items() if k in merged and merged[k] != v]
if bad:
    print(f"{len(bad)} tensor(s) differ from the reference, e.g. "
          f"{bad[0]}: {merged[bad[0]]} vs {ref[bad[0]]}", file=sys.stderr)
    sys.exit(1)
EOF
}

[ -d "$CKPT_ROOT" ] || { log "no such directory: ${CKPT_ROOT}"; exit 1; }

# A specific step may be named directly:
#
#   ./resolve_ckpt.sh checkpoints/q9b_grpo                   # latest step
#   ./resolve_ckpt.sh checkpoints/q9b_grpo/global_step_670   # exactly this one
#
# The two are not interchangeable. "Latest" is the right default for a preempted
# run that is still training, but a checkpoint chosen for evaluation is chosen by
# a measured validation curve, not by recency -- q9b_grpo peaks at 670 and trains
# on to 1452. So when a step is named it is the ONLY candidate: falling back to a
# neighbour would publish a number under the wrong step's label, which is worse
# than failing.
SINGLE=0
if [[ "$(basename "$CKPT_ROOT")" == global_step_* ]]; then
    STEP_DIRS=("$CKPT_ROOT")
    SINGLE=1
else
    # Newest first, so a preempted run evaluates its latest complete step.
    mapfile -t STEP_DIRS < <(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'global_step_*' \
                             | sed 's/.*global_step_//' | sort -rn | head -20 \
                             | sed "s|^|${CKPT_ROOT}/global_step_|")
fi
if [ "${#STEP_DIRS[@]}" -eq 0 ]; then
    if has_weights "$CKPT_ROOT"; then echo "$CKPT_ROOT"; exit 0; fi
    log "no global_step_* under ${CKPT_ROOT} and no weights at its root"
    exit 1
fi

for STEP_DIR in "${STEP_DIRS[@]}"; do
    step="${STEP_DIR##*/global_step_}"
    ACTOR="${STEP_DIR}/actor"
    [ -d "$ACTOR" ] || ACTOR="$STEP_DIR"

    # Already merged by a previous call?
    if has_weights "${ACTOR}/huggingface"; then echo "${ACTOR}/huggingface"; exit 0; fi
    if has_weights "${ACTOR}/merged"; then
        if dtypes_ok "${ACTOR}/merged"; then
            echo "${ACTOR}/merged"; exit 0
        fi
        # --check is called from the login node by launch_evals.sh's submission
        # guard, and it promises to report without changing anything. Say the
        # merge is needed; let the compute node do the moving.
        if [ "$MODE" = "--check" ]; then
            log "step ${step}: merged/ has the wrong dtypes for this architecture and will be rebuilt in the job"
            exit 2
        fi
        # Renamed, not removed: it is hours of compute, it is the only evidence of
        # what the old merger produced, and re-merging needs only the shards, which
        # are untouched. Whoever wants the space back can take it deliberately.
        STALE="${ACTOR}/merged.stale-$(date +%Y%m%d_%H%M%S)"
        log "step ${step}: merged/ has the wrong dtypes for this architecture; moving it to $(basename "$STALE") and re-merging"
        mv "${ACTOR}/merged" "$STALE" || { log "step ${step}: could not move the stale merge aside"; exit 1; }
    fi

    SHARDS=$(ls "${ACTOR}"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
    if [ "$SHARDS" -eq 0 ]; then
        if [ "$SINGLE" = "1" ]; then
            log "step ${step} was named explicitly but has no shards and no merged weights"
            exit 1
        fi
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
        if has_weights "${ACTOR}/merged"; then
            # Hold a FRESH merge to the same standard as an existing one. Accepting
            # it on has_weights alone made the gate unsatisfiable: a merger that
            # cannot produce a passing config renames merged/ aside, re-merges for
            # hours, accepts the byte-identical result, and the next job repeats it.
            # That loop left 94 merged.stale-* directories and 2.2 TB on disk, and
            # re-merged q27b_grpo/global_step_380 sixteen times. Failing loudly here
            # costs one merge; passing costs every future job.
            if dtypes_ok "${ACTOR}/merged"; then
                echo "${ACTOR}/merged"; exit 0
            fi
            log "step ${step}: the FRESH merge still fails the dtype check — the merger"
            log "  cannot produce a servable checkpoint here, so re-merging will not help."
            log "  Fix verl/model_merger before retrying; nothing further is renamed."
            exit 1
        fi
        log "step ${step}: merge reported success but wrote no weights"
    else
        log "step ${step}: merge failed"
    fi
    # Fall through and try an older step rather than giving up: a step can be
    # half-written when preemption lands mid-save. Not when the caller named the
    # step -- see SINGLE above.
    if [ "$SINGLE" = "1" ]; then
        log "step ${step} was named explicitly; not substituting another step"
        exit 1
    fi
done

log "no loadable checkpoint under ${CKPT_ROOT}"
exit 1
