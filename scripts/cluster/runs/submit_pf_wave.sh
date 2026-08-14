#!/bin/bash
# Post-fix (sentinel-v2) eval wave submitter.
#
# Every submission gets a FRESH output dir (TAG suffix _pf, repeats _pf_r2/_r3):
# the old eval_results/<arm>/ dirs hold pre-fix artifacts and mixing extraction
# rules inside one dir is exactly the failure the resume guard now refuses.
# Repeats run concurrently because their dirs are disjoint.
#
#   ./submit_pf_wave.sh A            # gradeability arms, text-4, 3 repeats
#   ./submit_pf_wave.sh B            # Table-1 matched cells, text-4, 3 repeats
#   ./submit_pf_wave.sh missing      # cells absent regardless of the fix
#   PLAN=1 ./submit_pf_wave.sh A     # print, submit nothing
#   REPEATS=1 ./submit_pf_wave.sh A  # override repeat count
set -uo pipefail

RUN_ROOT="${RUN_ROOT:-/data/project/private/minstar/workspace/hcgym_rebuttal}"
SCRIPT="${RUN_ROOT}/runs/eval_agentic.slurm"
CK="${RUN_ROOT}/checkpoints"
Q4B_BASE="/data/project/public/checkpoints/Qwen3.5-4B"
Q9B_BASE="${RUN_ROOT}/models/Qwen3.5-9B"
Q27B_BASE="/data/project/public/checkpoints/Qwen3.5-27B"

# slake/vqa_rad are excluded on purpose: 0 artifacts across all 190 stored
# files -- the stored VQA numbers are already clean and re-running them buys
# nothing (PLAN_14DAY_20260814.md).
BENCH="${BENCH:-medqa mmlu kqa_golden medication_qa}"
REPEATS="${REPEATS:-3}"
PLAN="${PLAN:-0}"

WAVE="${1:?usage: submit_pf_wave.sh A|B|missing}"

declare -a JOBS
case "$WAVE" in
  A)  # gradeability claim arms (GRADEABILITY.md rewrite depends on these)
    JOBS=(
      "q4b_grpo_s480|${CK}/q4b_grpo/global_step_480/actor/merged"
      "q4b_grpo_s520|${CK}/q4b_grpo/global_step_520/actor/merged"
      "q4b_grpo_fmtmatch_s480|${CK}/q4b_grpo_fmtmatch/global_step_480/actor/merged"
      "q4b_grpo_fmtmatch_s524|${CK}/q4b_grpo_fmtmatch/global_step_524/actor/merged"
      "q4b_grpo_golddrop_s480|${CK}/q4b_grpo_golddrop/global_step_480/actor/merged"
      "q4b_grpo_golddrop_s524|${CK}/q4b_grpo_golddrop/global_step_524/actor/merged"
      "base4b|${Q4B_BASE}"
    ) ;;
  B)  # Table-1 matched cells (base/GRPO/TT-OPD per backbone)
    JOBS=(
      "base|${Q9B_BASE}"
      "q9b_grpo_s480|${CK}/q9b_grpo/global_step_480/actor/merged"
      "q9b_grpo_s520|${CK}/q9b_grpo/global_step_520/actor/merged"
      "q9b_grpo_s670|${CK}/q9b_grpo/global_step_670/actor/merged"
      "q9b_ttopd_s730|${CK}/q9b_ttopd/global_step_730/actor/merged"
      "q4b_grpo_s1370|${CK}/q4b_grpo/global_step_1370/actor/merged"
      "q4b_ttopd_s90|${CK}/q4b_ttopd/global_step_90/actor/merged"
      "q4b_ttopd_s980|${CK}/q4b_ttopd/global_step_980/actor/merged"
      "q9btxt_grpo_s340|${CK}/q9btxt_grpo/global_step_340/actor/merged"
      "q9btxt_ttopd_s660|${CK}/q9btxt_ttopd/global_step_660/actor/merged"
    ) ;;
  missing)  # cells absent regardless of the extraction fix (single run).
            # base4b's LFQA gap is covered by wave A's base4b_pf (text-4).
    REPEATS=1
    JOBS=(
      "base27b|${Q27B_BASE}"
    )
    BENCH="medqa kqa_golden medication_qa" ;;
  *) echo "unknown wave '$WAVE'"; exit 2 ;;
esac

n=0
for spec in "${JOBS[@]}"; do
  arm="${spec%%|*}"; model="${spec#*|}"
  if [ ! -e "${model}/config.json" ]; then
    echo "[skip] ${arm}: no config.json under ${model}"; continue
  fi
  for r in $(seq 1 "$REPEATS"); do
    tag="${arm}_pf"; [ "$r" -gt 1 ] && tag="${arm}_pf_r${r}"
    if [ "$PLAN" = "1" ]; then
      echo "[plan] TAG=${tag} BENCH='${BENCH}' MODEL=${model}"
    else
      sbatch -J "hcgym-eval-${tag}" \
        --export=ALL,MODEL="${model}",TAG="${tag}",BENCH="${BENCH}" \
        "$SCRIPT"
    fi
    n=$((n + 1))
  done
done
echo "[done] wave ${WAVE}: ${n} submission(s) (repeats=${REPEATS}, bench='${BENCH}')"
