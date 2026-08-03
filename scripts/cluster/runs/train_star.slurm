#!/bin/bash
#SBATCH --job-name=hcgym-star
#SBATCH --partition=pt2_preemptible
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --output=/data/project/private/minstar/workspace/hcgym_rebuttal/logs/slurm_%x_%j.log
# pt2_preemptible preempts with PreemptMode=REQUEUE and GraceTime=0, so a preempted
# job restarts under the SAME job id and reopens this same file. Slurm's default open
# mode is truncate, which erases the job's own earlier output every time it is
# requeued: q4b_grpo's checkpoints run contiguously from step 10 to 630 with an
# identical config either side, but its validation curve survives only for steps
# 10-190 and 390-600 -- the middle third was overwritten. Append instead.
#SBATCH --open-mode=append
#
# STaR baseline arm (Zelikman et al., NeurIPS 2022, arXiv:2203.14465).
#
# WHAT THIS ARM TESTS. The paper names the RL methods of Section 2.3 without ever
# comparing against them, and it has no evidence separating a real capability gain
# from the policy learning to satisfy the reward's surface form. STaR is the
# null hypothesis for BOTH: is reinforcement learning needed at all, beyond
# supervised imitation of the rollouts that happened to succeed? Sample from the
# current policy, keep the CORRECT ones, fine-tune, repeat. If this matches the
# RL arms, the paper's contribution claim has to be restated.
#
# THIS IS A REIMPLEMENTATION, NOT THE AUTHORS' CODE. github.com/ezelikman/STaR is
# JAX 0.2.12 / mesh-transformer-jax / GPT-J-6B on TPU + Ray and cannot run on this
# cluster. It was not run, and nothing here may be described as its output. The
# full deviation list is in baselines/STAR_PORT.md and must be quoted in the
# rebuttal.
#
#   sbatch --export=ALL,BACKBONE=<path>,EXP=q9b_star train_star.slurm
#
#   SCRIPT=$RUN_ROOT/runs/train_star.slurm ./autoretry.sh q9b_star <BACKBONE> star
#
# HELD FIXED WITH THE OTHER ARMS, so an arm-to-arm difference is the method:
#
#   pool          full_4modality_clean, train split, the SAME 3390 tasks the RL
#                 arms' parquet is built from (data/domains/<POOL>/{tasks,split_tasks}.json).
#   MAX_TURNS     5, identical to train_hcgym.slurm and eval_agentic.slurm.
#   student GPUs  7 train the student, exactly as in every RL arm. The 8th is
#                 idle during SFT (it hosts the EMA teacher in ttopd); all 8
#                 serve the policy during the generation half, which no RL arm
#                 has an analogue of.
#   reward guard  the degenerate filter, on with the same settings as every arm.
#   rollout budget ITERATIONS x |pool| x SAMPLES_PER_TASK
#                 = 3 x 3390 x 3 = 30,510, which is exactly
#                 total_epochs x |pool| x rollout.n = 3 x 3390 x 3 for the RL
#                 arms. star_generate.py prints and persists this arithmetic
#                 every run; do not change SAMPLES_PER_TASK or ITERATIONS
#                 without changing the claim.
#
# DELIBERATELY NOT SET: COSINE_* (the cosine length reward is TT-OPD's addition
# and belongs to no baseline) and every BT_OPD_/HINT_OPD_/ADAPTIVE_DISTILL_
# variable. autoretry.sh submits with --export=ALL, so a stale COSINE_REWARD=1
# in the submitting shell would silently shape this arm's diagnostics; they are
# unset explicitly below rather than assumed absent, the same discipline
# train_hcgym.slurm uses.
#
# ACCEPTANCE SIGNAL. ACCEPT_ON=qa_accuracy — correctness alone, from the GYM's
# own grader. Accepting on the 6D composite would select for format/coherence,
# i.e. for exactly the reward-format adaptation this arm exists to rule out;
# measured on 200 real training tasks the composite ranks a wrong-but-fluent
# rollout above a correct-but-terse one 52.5% of the time
# (tests/test_star_baseline.py::test_composite_reward_ranks_...). Every signal is
# recorded for every rollout, so the composite-accepted variant can be produced
# afterwards with star_generate.py --refilter-from and NO new rollouts.
#
# PREEMPTION. pt2_preemptible requeues with GraceTime=0. star_iterate.py writes
# iter_NN/state.json after each completed iteration and resumes past them, so a
# requeue re-runs at most one iteration. Drive it with autoretry.sh.
#
# DRY_RUN=1 resolves the arm, prints DRYRUN-prefixed lines and the exact
# commands it would run, and exits before anything is served or trained.
set -euo pipefail

source /data/project/private/minstar/workspace/hcgym_rebuttal/runs/common_env.sh

BACKBONE="${BACKBONE:?set BACKBONE to a model directory}"
EXP="${EXP:?set EXP to an experiment name}"
# ARM is exported by autoretry.sh for every template; this one has a single arm.
ARM="${ARM:-star}"

POOL="${POOL:-full_4modality_clean}"
SPLIT="${SPLIT:-train}"
MAX_TURNS="${MAX_TURNS:-5}"
ITERATIONS="${ITERATIONS:-3}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-3}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
ACCEPT_ON="${ACCEPT_ON:-qa_accuracy}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-}"
MAX_ACCEPTED_PER_TASK="${MAX_ACCEPTED_PER_TASK:-1}"
INIT_FROM="${INIT_FROM:-base}"
DATASET_SCOPE="${DATASET_SCOPE:-current}"
SFT_SOURCE="${SFT_SOURCE:-sft_path}"
# 7 student GPUs for the SFT half — the same student topology as every RL arm.
STUDENT_GPUS="${STUDENT_GPUS:-7}"
# All 8 serve the policy during generation. Generation has no counterpart in the
# RL arms' GPU budget (their rollouts are colocated with training), so using the
# idle 8th here changes no arm-to-arm comparison; it only shortens wall clock.
SERVE_GPUS="${SERVE_GPUS:-8}"
SERVE_PORT="${SERVE_PORT:-31500}"
WORKERS="${WORKERS:-16}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_BATCH="${SFT_BATCH:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-8192}"
HELDOUT_SIZE="${HELDOUT_SIZE:-200}"
HELDOUT_SPLIT="${HELDOUT_SPLIT:-test}"
ALLOW_NO_TRL="${ALLOW_NO_TRL:-0}"
DRY_RUN="${DRY_RUN:-0}"

OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}/baselines/STaR/${EXP}}"
[ "$DRY_RUN" = "1" ] || mkdir -p "$OUT_ROOT"

STAR_ITERATE="${HCGYM_ROOT}/scripts/rebuttal/star_iterate.py"
STAR_GENERATE="${HCGYM_ROOT}/scripts/rebuttal/star_generate.py"
POOL_DIR="${HCGYM_ROOT}/data/domains/${POOL}"

MISSING=0
for f in "${POOL_DIR}/tasks.json" "${POOL_DIR}/split_tasks.json" \
         "${STAR_ITERATE}" "${STAR_GENERATE}" \
         "${HCGYM_ROOT}/bioagents/training/sft_trainer.py" \
         "${HCGYM_ROOT}/scripts/verl/reward_fn.py" "${BACKBONE}/config.json"; do
    if [ ! -e "$f" ]; then
        echo "[warn] missing required input: $f"
        MISSING=$((MISSING + 1))
    fi
done

# sft_trainer.py imports trl inside train(). Without it the job would spend the
# whole generation half (hours) and then die at the first optimizer step, so it
# is checked up front rather than discovered late.
if ! python -c "import trl" 2>/dev/null; then
    echo "[warn] trl is not importable in ${VENV}"
    echo "[warn]   bioagents/training/sft_trainer.py needs it for the SFT half."
    echo "[warn]   remedy:  ${VENV}/bin/pip install 'trl>=0.8'"
    echo "[warn]   override: ALLOW_NO_TRL=1 (generation half only)"
    if [ "$ALLOW_NO_TRL" != "1" ]; then
        MISSING=$((MISSING + 1))
    fi
fi

if [ "$MISSING" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "[fatal] ${MISSING} required input(s) missing"; exit 1
fi

echo "[run] exp=${EXP} arm=${ARM} backbone=${BACKBONE}"
echo "[run] pool=${POOL}/${SPLIT} max_turns=${MAX_TURNS} iterations=${ITERATIONS} k=${SAMPLES_PER_TASK}"
echo "[run] accept_on=${ACCEPT_ON} init_from=${INIT_FROM} dataset_scope=${DATASET_SCOPE}"
echo "[run] student_gpus=${STUDENT_GPUS} serve_gpus=${SERVE_GPUS} out=${OUT_ROOT}"

# ── Reward shaping ────────────────────────────────────────────────────────────
# Identical to every other arm: the degenerate filter guards all of them and is
# orthogonal to the method. It affects this arm only through the recorded
# verl_score diagnostic — the acceptance decision runs on qa_accuracy, which no
# DEGENERATE_* setting touches.
export REWARD_DEBUG_LOG=0
export DEGENERATE_FILTER=1
export DEGENERATE_EXCLUDE=1
export DEGENERATE_SENTINEL=-999.0
export DEGENERATE_NGRAM_THRESHOLD=0.3
export DEGENERATE_MIN_LENGTH=200
export DEGENERATE_GIBBERISH=1

# TT-OPD machinery, off and explicitly so (see the --export=ALL note above).
unset COSINE_REWARD COSINE_L_MAX COSINE_CHARS_PER_TOKEN \
      COSINE_R0_CORRECT COSINE_RL_CORRECT COSINE_R0_WRONG COSINE_RL_WRONG \
      COSINE_R_EXCEED 2>/dev/null || true
unset BT_OPD_MAX_TURN BT_OPD_BIDIRECTIONAL BT_OPD_MODEL_PATH BT_OPD_TOPK_RATIO \
      HINT_OPD_ENABLED HINT_OPD_CORRECT HINT_OPD_INCORRECT \
      CONDITIONAL_TEACHER_UPDATE ADAPTIVE_DISTILL_COEF \
      ADAPTIVE_DISTILL_KL_DECAY_START ADAPTIVE_DISTILL_KL_OFF \
      ADAPTIVE_DISTILL_COEF_FLOOR 2>/dev/null || true
unset OPSD_GOLD_CONDITIONING OPSD_GOLD_MAX_CHARS OPSD_TOKEN_CLIP 2>/dev/null || true

# The budget accounting star_generate.py prints reads these to describe the RL
# reference, so they must match train_hcgym.slurm's defaults.
export ROLLOUT_N="${ROLLOUT_N:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

STAR_ARGS=(
    --base-model "${BACKBONE}"
    --out-root "${OUT_ROOT}"
    --iterations "${ITERATIONS}"
    --pool "${POOL}"
    --split "${SPLIT}"
    --samples-per-task "${SAMPLES_PER_TASK}"
    --temperature "${TEMPERATURE}"
    --max-turns "${MAX_TURNS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --workers "${WORKERS}"
    --accept-on "${ACCEPT_ON}"
    --max-accepted-per-task "${MAX_ACCEPTED_PER_TASK}"
    --init-from "${INIT_FROM}"
    --dataset-scope "${DATASET_SCOPE}"
    --sft-source "${SFT_SOURCE}"
    --sft-gpus "${STUDENT_GPUS}"
    --sft-epochs "${SFT_EPOCHS}"
    --sft-lr "${SFT_LR}"
    --sft-batch "${SFT_BATCH}"
    --sft-grad-accum "${SFT_GRAD_ACCUM}"
    --sft-max-length "${SFT_MAX_LENGTH}"
    --serve-port "${SERVE_PORT}"
    --serve-gpus "${SERVE_GPUS}"
    --heldout-split "${HELDOUT_SPLIT}"
    --heldout-size "${HELDOUT_SIZE}"
)
if [ -n "${ACCEPT_THRESHOLD}" ]; then
    STAR_ARGS+=(--accept-threshold "${ACCEPT_THRESHOLD}")
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "DRYRUN arm=${ARM}"
    echo "DRYRUN backbone=${BACKBONE}"
    echo "DRYRUN pool=${POOL} split=${SPLIT}"
    echo "DRYRUN student_gpus=${STUDENT_GPUS}"
    echo "DRYRUN serve_gpus=${SERVE_GPUS}"
    echo "DRYRUN max_turns=${MAX_TURNS}"
    echo "DRYRUN iterations=${ITERATIONS}"
    echo "DRYRUN samples_per_task=${SAMPLES_PER_TASK}"
    echo "DRYRUN accept_on=${ACCEPT_ON}"
    echo "DRYRUN init_from=${INIT_FROM}"
    echo "DRYRUN dataset_scope=${DATASET_SCOPE}"
    echo "DRYRUN sft_source=${SFT_SOURCE}"

    # Enumerate the LIVE environment rather than echoing a hand-maintained list,
    # so a shaping variable that leaked in through --export=ALL cannot hide.
    # compgen -e is a bash builtin: on this cluster `env` resolves to a uv
    # PATH-shim that prints nothing and $HOME is not mounted on compute nodes.
    n_cos=0
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        echo "DRYRUN LEAKED cosine ${name}=${!name}"
        n_cos=$((n_cos + 1))
    done < <(compgen -e | grep -E '^COSINE_' | sort || true)
    echo "DRYRUN cosine_count=${n_cos}"

    n_dist=0
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        echo "DRYRUN LEAKED distill ${name}"
        n_dist=$((n_dist + 1))
    done < <(compgen -e | grep -E '^(BT_OPD_|HINT_OPD_|CONDITIONAL_TEACHER_UPDATE|ADAPTIVE_DISTILL_|OPSD_)' | sort || true)
    echo "DRYRUN distill_env_count=${n_dist}"

    n_deg=0
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        echo "DRYRUN degenerate ${name}=${!name}"
        n_deg=$((n_deg + 1))
    done < <(compgen -e | grep -E '^DEGENERATE_' | sort || true)
    echo "DRYRUN degenerate_count=${n_deg}"

    if [ "$MISSING" -gt 0 ]; then
        echo "DRYRUN preflight=MISSING:${MISSING}"
    else
        echo "DRYRUN preflight=OK"
    fi

    cd "$HCGYM_ROOT"
    python "${STAR_ITERATE}" "${STAR_ARGS[@]}" --dry-run
    echo "DRYRUN ok"
    exit 0
fi

cd "$HCGYM_ROOT"

python "${STAR_ITERATE}" "${STAR_ARGS[@]}"

echo "[done] ${EXP} $(date '+%F %T')"
