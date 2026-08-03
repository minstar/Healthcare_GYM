#!/bin/bash
#SBATCH --job-name=hcgym-opsd
#SBATCH --partition=pt2_preemptible
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --output=/data/project/private/minstar/workspace/hcgym_rebuttal/logs/slurm_%x_%j.log
# pt2_preemptible preempts with PreemptMode=REQUEUE and GraceTime=0, so a preempted
# job restarts under the SAME job id and reopens this same file. Slurm's default open
# mode is truncate, which erases the job's own earlier output every time it is
# requeued: q4b_grpo's checkpoints run contiguously from step 10 to 630 with an
# identical config either side, but its validation curve survives only for steps
# 10-190 and 390-600 -- the middle third was overwritten. Append instead.
#SBATCH --open-mode=append
#
# OPSD baseline arm (Zhao et al., ICML 2026, arXiv:2601.18734) for the rebuttal.
#
# Ports on-policy self-distillation into the same verl fork as TT-OPD so this arm
# isolates whether TT-OPD's gains come from its specific machinery or from
# on-policy self-distillation in general. The training setup (backbone, data,
# GPU topology of 7 student + 1 teacher, batch/rollout budget, degenerate-filter
# reward guard) is IDENTICAL to runs/train_hcgym.slurm; only the method differs:
#
#   ON  : frozen teacher (= OPSD fixed_teacher; the base weights never update),
#         per-sample gold-answer conditioning of the teacher (OPSD privileged
#         context), token-level KL on the student's on-policy trajectory
#         (loss_mode=opsd), OPSD per-token KL clipping, pure distillation
#         objective (no policy-gradient task reward, exactly as OPSD trains).
#   OFF : every TT-OPD addition — EMA/self-distillation teacher updates,
#         outcome-conditioned correctness hints, turn-level truncation,
#         bidirectional reward-sign flip, top-k position filtering, cosine
#         length-controlled reward, adaptive distill coef.
#
# This is a faithful PORT, not the official OPSD script; the full deviation list
# is in baselines/OPSD_PORT.md and must be quoted in the rebuttal.
#
#   sbatch --export=ALL,BACKBONE=<path>,EXP=q9b_opsd train_opsd.slurm
#
# Preemption on this partition is REQUEUE with GraceTime=0; drive with
# autoretry.sh, resume_mode=auto picks up the latest checkpoint.
set -euo pipefail

source /data/project/private/minstar/workspace/hcgym_rebuttal/runs/common_env.sh

BACKBONE="${BACKBONE:?set BACKBONE to a model directory}"
EXP="${EXP:?set EXP to an experiment name}"
TOOL_FORMAT="${TOOL_FORMAT:-qwen3_coder}"
INJECT_SCHEMAS="${INJECT_SCHEMAS:-False}"
DATA_DIR="${DATA_DIR:-${RUN_ROOT}/data/verl_parquet/full_4modality_clean}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MAX_TURNS="${MAX_TURNS:-5}"
ROLLOUT_N="${ROLLOUT_N:-3}"
# 7 GPUs train the student; the 8th hosts the frozen teacher — the same student
# topology as both other arms.
STUDENT_GPUS="${STUDENT_GPUS:-7}"
TRAIN_BATCH="${TRAIN_BATCH:-7}"

CKPT_DIR="${RUN_ROOT}/checkpoints/${EXP}"
ROLLOUT_DIR="${RUN_ROOT}/rollout_dumps/${EXP}"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

for f in "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" \
         "${HCGYM_ROOT}/scripts/verl/tool_config_full.yaml" \
         "${HCGYM_ROOT}/scripts/verl/reward_fn.py" "${BACKBONE}/config.json"; do
    [ -e "$f" ] || { echo "[fatal] missing required input: $f"; exit 1; }
done

echo "[run] exp=${EXP} arm=opsd backbone=${BACKBONE}"
echo "[run] data=${DATA_DIR} tool_format=${TOOL_FORMAT} student_gpus=${STUDENT_GPUS}"

# ── Reward shaping ────────────────────────────────────────────────────────────
# The degenerate filter guards every arm; it is orthogonal to the method.
# COSINE_REWARD is deliberately NOT set: the cosine length-controlled reward is
# TT-OPD's addition and must be OFF in this arm. (Rewards are still computed for
# logging/metrics, but with use_task_rewards=False below they carry no gradient.)
export REWARD_DEBUG_LOG=1
export DEGENERATE_FILTER=1
export DEGENERATE_EXCLUDE=1
export DEGENERATE_SENTINEL=-999.0
export DEGENERATE_NGRAM_THRESHOLD=0.3
export DEGENERATE_MIN_LENGTH=200
export DEGENERATE_GIBBERISH=1

# ── OPSD arm ──────────────────────────────────────────────────────────────────
# NOT set (all TT-OPD machinery off): HINT_OPD_ENABLED, BT_OPD_MAX_TURN,
# BT_OPD_BIDIRECTIONAL, BT_OPD_TOPK_RATIO, CONDITIONAL_TEACHER_UPDATE,
# ADAPTIVE_DISTILL_COEF, COSINE_REWARD.
# (The opsd loss never reads BT_OPD_* anyway; leaving them unset keeps the log
# output honest.)

# Privileged conditioning: inject the sample's ground-truth answer into the
# teacher context (reward_model.ground_truth, falling back to
# extra_info.correct_answer for open-ended samples whose graded key is empty).
export OPSD_GOLD_CONDITIONING=1
export OPSD_GOLD_MAX_CHARS="${OPSD_GOLD_MAX_CHARS:-2000}"

# Per-token KL clipping. OPSD added this because style tokens ('wait', 'think')
# show 6-15x higher KL than content tokens and dominate the gradient. 0.05 is
# their published thinking-model value (run_opsd_1b/4b: 0.05, 8b: 0.06); note it
# was tuned for their full-vocab forward-KL values, not our sampled-token k3
# estimator — see OPSD_PORT.md. Override with OPSD_TOKEN_CLIP=0 to disable.
export OPSD_TOKEN_CLIP="${OPSD_TOKEN_CLIP:-0.05}"

# Frozen teacher = OPSD's fixed_teacher: distillation.self_distillation is NOT
# set, so the teacher-update block in ray_trainer never runs and the teacher
# stays at the base weights for the whole run. No EMA.
DISTILL_ARGS=(
    distillation.enabled=True
    distillation.teacher_model.model_path="${BACKBONE}"
    distillation.teacher_model.inference.name=sglang
    distillation.teacher_model.enable_resource_pool=True
    distillation.teacher_model.n_gpus_per_node=1
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_model.inference.gpu_memory_utilization=0.5
    distillation.teacher_model.inference.enforce_eager=True
    distillation.teacher_model.inference.free_cache_engine=True
    +distillation.teacher_model.inference.engine_kwargs.sglang.attention_backend=triton
    distillation.distillation_loss.loss_mode=opsd
    # OPSD backpropagates the divergence directly as a supervised loss (their
    # generalized_jsd_loss path), not through a policy gradient, and its
    # objective is pure distillation — no task-reward term.
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.use_task_rewards=False
)

cd "$VERL_ROOT"

# DRY_RUN=1 resolves the config, prints it on DRYRUN- prefixed lines, and exits
# before the trainer starts.
#
# This script was the only one of the four training templates with no DRY_RUN
# handling at all, while its siblings all documented and honoured it. Running
# `DRY_RUN=1 bash runs/train_opsd.slurm` therefore did exactly what the operator
# was trying to avoid: it started Ray, brought up seven FSDP workers and began
# loading a 9.4B model — on whatever host the command was typed on, which for a
# sanity check is the login node. Verified once, by doing it.
DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRYRUN exp=${EXP}"
    echo "DRYRUN backbone=${BACKBONE}"
    echo "DRYRUN data_dir=${DATA_DIR}"
    echo "DRYRUN student_gpus=${STUDENT_GPUS}"
    echo "DRYRUN train_batch=${TRAIN_BATCH}"
    echo "DRYRUN max_turns=${MAX_TURNS}"
    echo "DRYRUN teacher=${TEACHER:-<same as backbone>}"
    n=0
    for arg in "${DISTILL_ARGS[@]}"; do
        echo "DRYRUN distill ${arg}"
        n=$((n + 1))
    done
    echo "DRYRUN distill_arg_count=${n}"
    echo "DRYRUN ok"
    exit 0
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH}" \
    data.max_prompt_length=8192 \
    data.max_response_length=12288 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    data.return_raw_chat=True \
    data.tool_config_path=null \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${BACKBONE}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers="${TRAIN_BATCH}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${HCGYM_ROOT}/scripts/verl/tool_config_full.yaml" \
    actor_rollout_ref.rollout.multi_turn.inject_tool_schemas="${INJECT_SCHEMAS}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${HCGYM_ROOT}/scripts/verl/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    "${DISTILL_ARGS[@]}" \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=hcgym-rebuttal \
    trainer.experiment_name="${EXP}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node="${STUDENT_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.total_epochs="${TOTAL_EPOCHS}"

echo "[done] ${EXP} $(date '+%F %T')"
