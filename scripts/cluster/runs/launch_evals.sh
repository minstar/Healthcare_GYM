#!/bin/bash
# Submit the benchmark evaluation matrix.
#
# Every condition in the rebuttal table goes through runs/eval_agentic.slurm with
# the SAME turn budget, engine, decoding and benchmark list. That uniformity is
# the point: the submitted paper could not difference its own columns because
# Base+AR ran largely on sglang while the RL arms ran on transformers (worth
# 1.0-3.1 pp) and because TT-OPD was scored at 5 turns while base/GRPO used 10.
# Do not add a per-condition override here without a written reason.
#
#   ./launch_evals.sh                 # every checkpoint that has finished training
#   ./launch_evals.sh q9b_ttopd       # one condition
#   ./launch_evals.sh base            # the untrained Base+AR reference
#   PLAN=1 ./launch_evals.sh          # print, submit nothing
#   BENCH="medqa" ./launch_evals.sh   # narrow the benchmark list (smoke)
#
# Base (text) is NOT here: it is a single-turn log-probability condition and runs
# through scripts/eval_benchmark_logprob.py, not the AgentRunner.
set -uo pipefail

# Overridable so the harness can run against another checkout or a fixture tree;
# the default is the layout every run in the paper used.
RUN_ROOT="${RUN_ROOT:-/data/project/private/minstar/workspace/hcgym_rebuttal}"
SCRIPT="${RUN_ROOT}/runs/eval_agentic.slurm"

# Six benchmarks, two per domain, chosen to cover every claim the rebuttal makes
# at roughly a fifth of the cost of all 18 (4,721 items vs ~24,000). Widen with
# BENCH="..." for the camera-ready.
#
#   medqa 1273        the flagship number, quoted in the paper and every review
#   mmlu 1089         where the agentic-textual transfer gap lives — the paper's
#                     largest reported degradation, so it cannot be dropped
#   kqa_golden 201    smallest LFQA; earliest signal
#   medication_qa 666 the rebuttal's largest LFQA claim (+11.4 pp over base)
#   vqa_rad 451       the paper's primary visual QA, smallest with images
#   slake 1061        a second visual QA, so a VQA result is not a single dataset
#
# Deliberately excluded: pathvqa (6,719 — dominates wall-clock for no new claim),
# pmc_vqa (a substitute release, not the official one), mimic_iii/eicu
# (credentialed PhysioNet, unrecoverable), vqa_med_2021 (ships no images),
# quilt_vqa (gated; one click at huggingface.co/datasets/wisdomik/Quilt_VQA then
# `bash restore_data.sh D_quilt`).
BENCH="${BENCH:-medqa mmlu kqa_golden medication_qa vqa_rad slake}"
MAX_TURNS="${MAX_TURNS:-5}"
PLAN="${PLAN:-0}"

# tag | checkpoint or backbone path
# Trained arms point at the checkpoint dir; autoretry writes them under checkpoints/<exp>.
Q9B="${RUN_ROOT}/models/Qwen3.5-9B"
# tag | path | prompt mode
#
# The three untrained prompting rows separate the method from the scaffolding: a
# reasoning-and-acting prompt over the same tool set, and the plain condition given
# more forceful tool instructions. Neither needs training, so both can run the
# moment a GPU frees. base_react additionally reports
# a react_rate — a low score at low adherence means the model ignored the format,
# which is the opposite conclusion from "ReAct does not help", so read them together.
CONDITIONS=(
  "base|${Q9B}|default"
  "base_strong_tool|${Q9B}|strong_tool"
  "base_react|${Q9B}|react"
  # The prompting upper bound: same model, same tools, allowed to read its own
  # failed attempt and retry. If this reaches the RL arms, the contribution claim
  # has to be restated — so it runs the full four-strategy ladder, including
  # LAST_ATTEMPT, which is what separates "reflection helped" from "a second
  # sample helped". Extra inference cost is reported alongside the score.
  "base_reflexion|${Q9B}|default"
  "q4b_grpo|${RUN_ROOT}/checkpoints/q4b_grpo"
  "q4b_ttopd|${RUN_ROOT}/checkpoints/q4b_ttopd"
  "q9b_grpo|${RUN_ROOT}/checkpoints/q9b_grpo"
  "q9b_grpo_cosine|${RUN_ROOT}/checkpoints/q9b_grpo_cosine"
  "q9b_ttopd|${RUN_ROOT}/checkpoints/q9b_ttopd"
  "q9b_ttopd_hints|${RUN_ROOT}/checkpoints/q9b_ttopd_hints"
  "q9b_opsd|${RUN_ROOT}/checkpoints/q9b_opsd"
  "q27b_grpo|${RUN_ROOT}/checkpoints/q27b_grpo"
  "q27b_ttopd|${RUN_ROOT}/checkpoints/q27b_ttopd"
  "q9btxt_grpo|${RUN_ROOT}/checkpoints/q9btxt_grpo"
  "q9btxt_ttopd|${RUN_ROOT}/checkpoints/q9btxt_ttopd"
  "glm9b_grpo|${RUN_ROOT}/checkpoints/glm9b_grpo"
  "glm9b_ttopd|${RUN_ROOT}/checkpoints/glm9b_ttopd"
)

WANT="${1:-all}"

# Decide whether a run has anything worth evaluating. This only INSPECTS — the
# actual FSDP-shard merge happens inside the job, on a compute node, because
# merging loads the whole model and must not run on the login node.
#
# resolve_ckpt.sh exit codes: 0 = a loadable path exists, 2 = shards present but
# not merged yet (still worth submitting), anything else = nothing usable.
checkpoint_status() {
    "${RUN_ROOT}/runs/resolve_ckpt.sh" "$1" --check >/dev/null 2>&1
    echo $?
}

submitted=0
for entry in "${CONDITIONS[@]}"; do
    IFS='|' read -r tag path mode <<< "$entry"
    mode="${mode:-default}"
    case "$tag" in *reflexion*) reflex=1 ;; *) reflex=0 ;; esac
    [ "$WANT" = "all" ] || [ "$WANT" = "$tag" ] || continue

    # The base condition points straight at a backbone; trained arms need a
    # checkpoint that either is already merged or has shards to merge.
    if [ "${tag#base}" != "$tag" ]; then
        model="$path"
        [ -f "${model}/config.json" ] || { echo "[skip] ${tag} — no backbone at ${path}"; continue; }
    else
        case "$(checkpoint_status "$path")" in
            0|2) model="$path" ;;
            *)   echo "[skip] ${tag} — no checkpoint worth evaluating under ${path}"; continue ;;
        esac
    fi
    if [ -d "${RUN_ROOT}/eval_results/${tag}" ] && \
       [ -n "$(ls -A "${RUN_ROOT}/eval_results/${tag}" 2>/dev/null)" ]; then
        echo "[skip] ${tag} — eval_results/${tag} is already populated"
        continue
    fi
    # Fail CLOSED. A broken squeue and an idle squeue both return an empty string,
    # and this is the only thing standing between a retry and a duplicate 8-hour
    # GPU job writing the same eval_results/${tag} as the one already queued. So
    # check the exit status, and refuse to submit when the queue is unreadable.
    if ! QSTATE="$(squeue -u "${USER:-$(id -un)}" -h -n "hcgym-eval-${tag}" -o %T 2>&1)"; then
        echo "[skip] ${tag} — cannot read the queue (${QSTATE%%$'\n'*}); refusing to submit rather than risk a duplicate"
        continue
    fi
    if printf '%s' "$QSTATE" | grep -q .; then
        echo "[skip] ${tag} — already queued or running"
        continue
    fi

    if [ "$PLAN" = "1" ]; then
        echo "[plan] ${tag}  model=${model}  turns=${MAX_TURNS}  prompt_mode=${mode}"
        submitted=$((submitted + 1))
        continue
    fi

    # 27B needs tensor parallelism; everything else fits on one card. The count
    # goes on the command line AND in EVAL_GPUS because Slurm does not expand
    # variables inside #SBATCH directives.
    case "$tag" in
        q27b_*) gpus=4 ;;
        *)      gpus=1 ;;
    esac

    echo "[submit] ${tag}  model=${model}  gpus=${gpus}  prompt_mode=${mode}"
    sbatch -J "hcgym-eval-${tag}" \
           --gres="gpu:${gpus}" \
           --export="ALL,EVAL_GPUS=${gpus},MODEL=${model},TAG=${tag},BENCH=${BENCH},MAX_TURNS=${MAX_TURNS},PROMPT_MODE=${mode},REFLEXION=${reflex}" \
           "$SCRIPT"
    submitted=$((submitted + 1))
    sleep 2
done

echo
if [ "$PLAN" = "1" ]; then
    echo "plan only — ${submitted} eval(s) would submit, nothing sent"
else
    echo "${submitted} eval(s) submitted"
fi
