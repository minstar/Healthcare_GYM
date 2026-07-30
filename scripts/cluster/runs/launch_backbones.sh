#!/bin/bash
# Launch the training arms for the rebuttal.
#
# Two independent axes live here:
#   * backbones — the model-family / size axis, arms grpo + ttopd
#   * methods   — the ablation that asks whether the gain is really TT-OPD or just
#                 the cosine length reward. That is a question about the METHOD,
#                 not about backbones, so grpo_cosine only runs on the anchor
#                 backbone (q9b) by default. Adding it to all four would triple
#                 the queue for no extra evidence.
#
# Each arm gets its own autoretry loop, because pt2_preemptible preempts with
# REQUEUE and GraceTime=0 and a single sbatch will not survive to step 60.
#
#   ./launch_backbones.sh                  # every backbone's default arms
#   ./launch_backbones.sh q4b              # one backbone, its default arms
#   ./launch_backbones.sh q4b:ttopd        # one arm, even if not a default
#   ./launch_backbones.sh q9b:grpo_cosine  # the length-reward ablation on the anchor
#
#   ARMS="grpo grpo_cosine" ./launch_backbones.sh q9b   # override the arm set
#   PLAN=1 ./launch_backbones.sh                        # print, launch nothing
#
# Order matters: 4B first. It is the cheapest and it validates the pipeline
# before any expensive backbone is committed to the queue.
set -uo pipefail

# Overridable so the harness can run against another checkout or a fixture tree;
# the default is the layout every run in the paper used.
RUN_ROOT="${RUN_ROOT:-/data/project/private/minstar/workspace/hcgym_rebuttal}"
PUBLIC=/data/project/public/checkpoints

# Every arm train_hcgym.slurm knows how to run. Explicit <tag>:<arm> selectors
# are validated against this, so a typo fails loudly instead of matching nothing.
ALL_ARMS=(grpo grpo_cosine ttopd)

# tag | backbone path | tool-call format | inject schemas | default arms | data pool
#
# q9b is the anchor backbone and the only one carrying grpo_cosine by default.
#
# The family axis is GLM-4-9B (Glm4ForCausalLM, 9.4B dense, Zhipu) against the
# Qwen3.5-9B anchor: same scale, same density, different family. Gemma-4 was the
# first choice and had to be dropped — sglang 0.5.9 ships no gemma4 model at all
# (gemma3n is the newest), and transformers cannot even load gemma-4-12B's
# gemma4_unified config. Both engines must support a backbone, not just one.
#
# GLM-4 is text-only, so it runs on the text-only pool. The anchor gets a matching
# q9btxt entry on that SAME pool: comparing GLM-on-text-only against
# Qwen-on-full-pool would confound family with training distribution.
FULL=full_4modality_clean
TEXT=full_4modality_clean_textonly

BACKBONES=(
  "q4b|${PUBLIC}/Qwen3.5-4B|qwen3_coder|False|grpo ttopd|${FULL}"
  "q9b|${RUN_ROOT}/models/Qwen3.5-9B|qwen3_coder|False|grpo grpo_cosine ttopd|${FULL}"
  "q27b|${PUBLIC}/Qwen3.5-27B|qwen3_coder|False|grpo ttopd|${FULL}"
  "q9btxt|${RUN_ROOT}/models/Qwen3.5-9B|qwen3_coder|False|grpo ttopd|${TEXT}"
  "glm9b|${RUN_ROOT}/models/GLM-4-9B-0414|hermes|True|grpo ttopd|${TEXT}"
)

backbone_ready() {
    # A backbone is usable only when config.json, a weight index, every shard the
    # index names, and a tokenizer are all present, with no download still in
    # flight. Checking config.json alone would launch a partially downloaded model.
    local dir="$1"
    [ -f "${dir}/config.json" ] || return 1
    find "$dir" -name '*.incomplete' -print -quit 2>/dev/null | grep -q . && return 1
    [ -f "${dir}/tokenizer.json" ] || [ -f "${dir}/tokenizer_config.json" ] || return 1

    local index="${dir}/model.safetensors.index.json"
    if [ -f "$index" ]; then
        python3 - "$dir" "$index" <<'PY_INNER' || return 1
import json, os, sys
d, idx = sys.argv[1], sys.argv[2]
shards = set(json.load(open(idx))["weight_map"].values())
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
sys.exit(1 if missing else 0)
PY_INNER
    else
        # Single-shard checkpoint.
        ls "${dir}"/*.safetensors >/dev/null 2>&1 || return 1
    fi
    return 0
}

WANT="${1:-all}"
PLAN="${PLAN:-0}"

# Split an explicit "tag:arm" selector once, up front.
WANT_TAG="${WANT%%:*}"
WANT_ARM=""
if [ "$WANT" != "${WANT#*:}" ]; then
    WANT_ARM="${WANT#*:}"
    case " ${ALL_ARMS[*]} " in
        *" ${WANT_ARM} "*) ;;
        *) echo "[fatal] unknown arm '${WANT_ARM}' — known arms: ${ALL_ARMS[*]}"; exit 1 ;;
    esac
fi

# Validate the backbone selector too, so a typo fails loudly instead of quietly
# matching no backbone and launching nothing.
KNOWN_TAGS=""
for entry in "${BACKBONES[@]}"; do KNOWN_TAGS="${KNOWN_TAGS}${entry%%|*} "; done
if [ "$WANT" != "all" ]; then
    case " ${KNOWN_TAGS}" in
        *" ${WANT_TAG} "*) ;;
        *) echo "[fatal] unknown backbone '${WANT_TAG}' — known: ${KNOWN_TAGS% }"; exit 1 ;;
    esac
fi

# Same for an ARMS= override, which otherwise would only fail once the job ran.
for a in ${ARMS:-}; do
    case " ${ALL_ARMS[*]} " in
        *" ${a} "*) ;;
        *) echo "[fatal] unknown arm '${a}' in ARMS — known arms: ${ALL_ARMS[*]}"; exit 1 ;;
    esac
done

launched=0
for entry in "${BACKBONES[@]}"; do
    IFS='|' read -r tag path fmt inject default_arms pool <<< "$entry"

    # Which arms does this backbone run? An explicit selector wins over the
    # defaults; ARMS= overrides both.
    if [ -n "${ARMS:-}" ]; then
        arms="$ARMS"
    elif [ -n "$WANT_ARM" ]; then
        arms="$WANT_ARM"
    else
        arms="$default_arms"
    fi

    # Does this backbone participate at all?
    case "$WANT" in
        all) ;;
        "$tag") ;;
        *) [ "$WANT_TAG" = "$tag" ] || continue ;;
    esac

    for arm in $arms; do
        sel="${tag}:${arm}"

        # config.json lands first in an HF download, so its presence says nothing
        # about the weights. Require every shard named by the index to exist, and
        # require no in-flight .incomplete files — otherwise a backbone that is
        # still downloading launches as a half-built model.
        if ! backbone_ready "$path"; then
            echo "[skip] ${sel} — backbone not ready: ${path}"
            continue
        fi
        if [ -f "${RUN_ROOT}/logs/.autoretry_${tag}_${arm}.done" ]; then
            echo "[skip] ${sel} — already complete"
            continue
        fi
        # The trailing space is load-bearing: without it this pattern would also
        # match the q9b_grpo_cosine loop and wrongly skip q9b_grpo.
        if pgrep -f "autoretry.sh ${tag}_${arm} " >/dev/null 2>&1; then
            echo "[skip] ${sel} — autoretry loop already running"
            continue
        fi

        if [ "$PLAN" = "1" ]; then
            echo "[plan] ${sel}  exp=${tag}_${arm}  backbone=${path}  format=${fmt}  inject_schemas=${inject}  pool=${pool}"
            launched=$((launched + 1))
            continue
        fi

        echo "[launch] ${sel}  backbone=${path}  format=${fmt}  inject_schemas=${inject}  pool=${pool}"
        nohup "${RUN_ROOT}/runs/autoretry.sh" \
              "${tag}_${arm}" "$path" "$arm" \
              "TOOL_FORMAT=${fmt},INJECT_SCHEMAS=${inject},DATA_DIR=${RUN_ROOT}/data/verl_parquet/${pool}" \
              > "${RUN_ROOT}/logs/autoretry_${tag}_${arm}.out" 2>&1 &
        launched=$((launched + 1))
        sleep 2
    done
done

if [ "$PLAN" = "1" ]; then
    echo
    echo "plan only — ${launched} arm(s) would launch, nothing submitted"
    exit 0
fi

echo
echo "running autoretry loops:"
pgrep -af "autoretry.sh" 2>/dev/null | sed 's/^/  /' || echo "  none"
