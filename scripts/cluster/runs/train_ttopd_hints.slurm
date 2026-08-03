#!/bin/bash
#SBATCH --job-name=hcgym-ttopd-hints
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
# Healthcare AI GYM — the ttopd_hints arm.
#
# WHAT THIS ARM IS FOR
# --------------------
# TT-OPD's outcome-conditioned privileged hints never executed in any run that
# produced a number in the paper. Three gates blocked them (see HINT_FIX.md):
# the streaming teacher path never forwarded hint tokens, no construction site
# passed a tokenizer to the hint builder, and the batch path that did build them
# was unreachable whenever the teacher had its own resource pool — which every
# run script from v26 on configures. So "TT-OPD" as executed was
#
#     GRPO + cosine length reward + EMA teacher + turn-level truncation
#
# with the third named mechanism inert. This arm is the first run in which hints
# actually fire, and it is the ONLY thing that changes: the config below is
# byte-for-byte the ttopd arm of train_hcgym.slurm, checked automatically (see
# PARITY below). ttopd_hints minus ttopd therefore isolates exactly one thing —
# the privileged outcome signal reaching the teacher — and nothing else.
#
#   sbatch --export=ALL,BACKBONE=<path>,EXP=q9b_ttopd_hints train_ttopd_hints.slurm
#
# READ THIS BEFORE RUNNING THE ttopd ARM AGAIN
# --------------------------------------------
# train_hcgym.slurm's ttopd arm exports HINT_OPD_ENABLED=1. Under the OLD verl
# that flag was inert, which is why the paper's TT-OPD numbers contain no hints.
# Under the FIXED verl the same flag now fires. To keep reproducing the paper's
# TT-OPD, that arm must export HINT_OPD_ENABLED=0; otherwise "ttopd" silently
# becomes this arm and the comparison collapses. That one-line change belongs to
# whoever owns train_hcgym.slurm; this script only refuses to be confused with it.
#
# PREFLIGHT
# ---------
# The job will not start training until verify_hint_injection.py confirms, against
# the exact verl tree this job imports, that hints can fire. An arm whose entire
# purpose is one mechanism must not be allowed to run with that mechanism inert a
# second time. SKIP_HINT_CHECK=1 overrides (do not).
#
# PARITY
# ------
# DRY_RUN=1 on train_hcgym.slurm prints its fully resolved arm config on DRYRUN-
# prefixed lines. This script prints the same lines and diffs them against
# `ARM=ttopd`. Any difference is fatal, because a difference means this arm is no
# longer a one-variable delta from the arm it is meant to be compared against.
# PARITY=0 disables the check; use it only when the divergence is understood.
#
# Preemption on this partition is REQUEUE with GraceTime=0; drive with autoretry.sh.
set -euo pipefail

source /data/project/private/minstar/workspace/hcgym_rebuttal/runs/common_env.sh

BACKBONE="${BACKBONE:?set BACKBONE to a model directory}"
EXP="${EXP:?set EXP to an experiment name}"
ARM=ttopd_hints
TOOL_FORMAT="${TOOL_FORMAT:-qwen3_coder}"
INJECT_SCHEMAS="${INJECT_SCHEMAS:-False}"
DATA_DIR="${DATA_DIR:-${RUN_ROOT}/data/verl_parquet/full_4modality_clean}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MAX_TURNS="${MAX_TURNS:-5}"
ROLLOUT_N="${ROLLOUT_N:-3}"
# 7 GPUs train the student, the 8th hosts the frozen EMA teacher — the same split
# every arm uses, so the effective batch geometry is unchanged.
STUDENT_GPUS="${STUDENT_GPUS:-7}"
TRAIN_BATCH="${TRAIN_BATCH:-7}"
DRY_RUN="${DRY_RUN:-0}"
PARITY="${PARITY:-1}"
REFERENCE_ARM="${RUN_ROOT}/runs/train_hcgym.slurm"

CKPT_DIR="${RUN_ROOT}/checkpoints/${EXP}"
ROLLOUT_DIR="${RUN_ROOT}/rollout_dumps/${EXP}"
[ "$DRY_RUN" = "1" ] || mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

MISSING=0
for f in "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" \
         "${HCGYM_ROOT}/scripts/verl/tool_config_full.yaml" \
         "${HCGYM_ROOT}/scripts/verl/reward_fn.py" "${BACKBONE}/config.json"; do
    if [ ! -e "$f" ]; then
        echo "[warn] missing required input: $f"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "[fatal] ${MISSING} required input(s) missing"; exit 1
fi

echo "[run] exp=${EXP} arm=${ARM} backbone=${BACKBONE}"
echo "[run] data=${DATA_DIR} tool_format=${TOOL_FORMAT} student_gpus=${STUDENT_GPUS}"

# ── Reward shaping ────────────────────────────────────────────────────────────
# Identical to every other arm; orthogonal to the method under test.
export REWARD_DEBUG_LOG=1
export DEGENERATE_FILTER=1
export DEGENERATE_EXCLUDE=1
export DEGENERATE_SENTINEL=-999.0
export DEGENERATE_NGRAM_THRESHOLD=0.3
export DEGENERATE_MIN_LENGTH=200
export DEGENERATE_GIBBERISH=1

# ── Cosine length-controlled reward (Yeo et al., arXiv:2502.03373) ────────────
# Values copied verbatim from train_hcgym.slurm's cosine_reward_on. The PARITY
# check below fails the job if they ever stop matching, so this copy cannot rot
# silently. Note the correctness boundary these values imply: a correct
# trajectory scores >= 0.7 and a wrong one lands in [-0.5, 0.0], which is what
# makes teacher_manager's `score > 0` hint threshold the right discriminator.
export COSINE_REWARD=1
export COSINE_L_MAX=12288
export COSINE_CHARS_PER_TOKEN=5.0
export COSINE_R0_CORRECT=1.1
export COSINE_RL_CORRECT=0.7
export COSINE_R0_WRONG=0.0
export COSINE_RL_WRONG=-0.3
export COSINE_R_EXCEED=-0.5

# ── EMA teacher + outcome-conditioned hints + turn-level truncation ───────────
export BT_OPD_MAX_TURN=0
export BT_OPD_BIDIRECTIONAL=1
export BT_OPD_MODEL_PATH="${BACKBONE}"
export BT_OPD_TOPK_RATIO=0.5

# THE mechanism this arm exists to test. Same flag, same hint texts as the ttopd
# arm — the difference is that the verl in VERL_ROOT now acts on them.
export HINT_OPD_ENABLED=1
export HINT_OPD_DYNAMIC=1
export HINT_OPD_CORRECT="The diagnostic reasoning appears sound, but could there be alternative diagnoses worth considering? Double-check whether the key findings truly support this conclusion."
export HINT_OPD_INCORRECT="Some aspects of this reasoning may need re-examination. Have you fully considered all the clinical findings? It might be worth revisiting the differential diagnosis and verifying each step of the reasoning process."
# Gold conditioning is the OPSD baseline's privileged signal and takes precedence
# over hints wherever both are on. Pinned off here so a stale export in the
# submitting shell (autoretry.sh submits with --export=ALL) cannot silently turn
# this arm into the OPSD arm.
export OPSD_GOLD_CONDITIONING=0
# Histogram of injected/skipped reasons every N samples in the training log, so
# "did hints fire" is answerable from the log rather than by inference.
export HINT_OPD_LOG_EVERY="${HINT_OPD_LOG_EVERY:-64}"

export CONDITIONAL_TEACHER_UPDATE=1
export ADAPTIVE_DISTILL_COEF=1
export ADAPTIVE_DISTILL_KL_DECAY_START=0.3
export ADAPTIVE_DISTILL_KL_OFF=0.7
export ADAPTIVE_DISTILL_COEF_FLOOR=0.1

# The teacher serves the same checkpoint as the student, and every backbone in
# this study is a VLM (Qwen3_5ForConditionalGeneration, Gemma4Unified...), so the
# two B200 settings below apply to the teacher engine as well as the student's.
DISTILL_ARGS=(
    distillation.enabled=True
    # Deliberately absent, exactly as in train_hcgym.slurm ARM=ttopd:
    #   +distillation.self_distillation=True
    # It gates a teacher-refresh block that calls update_teacher_ema() and
    # update_teacher_from_checkpoint(), neither of which exists on
    # TeacherModelManager, so turning it on AttributeErrors at step 5
    # (ema_update_interval=5). This arm carried it until the parity check caught
    # it, which meant the arm would have died at step 5 AND, had it survived,
    # ttopd_hints minus ttopd would have measured hints PLUS a live EMA teacher
    # rather than hints alone. The two arms must differ only in the hint
    # mechanism; the remaining EMA settings below are inert without this flag and
    # are kept so both arms pass hydra the same keys.
    +distillation.teacher_update_interval=30
    +distillation.use_ema=True
    +distillation.ema_decay=0.999
    +distillation.ema_update_interval=5
    distillation.teacher_model.model_path="${BACKBONE}"
    distillation.teacher_model.inference.name=sglang
    distillation.teacher_model.enable_resource_pool=True
    distillation.teacher_model.n_gpus_per_node=1
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_model.inference.gpu_memory_utilization=0.5
    distillation.teacher_model.inference.enforce_eager=True
    distillation.teacher_model.inference.free_cache_engine=False
    +distillation.teacher_model.inference.engine_kwargs.sglang.attention_backend=triton
    +distillation.teacher_model.inference.engine_kwargs.sglang.mm_attention_backend=triton_attn
    distillation.distillation_loss.loss_mode=bt_opd_kl
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.distillation_loss_coef=2.0
)

# ── Resolved-config report (same format as train_hcgym.slurm's DRY_RUN) ───────
emit_dryrun() {
    echo "DRYRUN arm=ttopd"     # compared against the ttopd arm, so report as ttopd
    echo "DRYRUN backbone=${BACKBONE}"
    echo "DRYRUN student_gpus=${STUDENT_GPUS}"
    echo "DRYRUN train_batch=${TRAIN_BATCH}"
    echo "DRYRUN rollout_n=${ROLLOUT_N}"
    echo "DRYRUN max_turns=${MAX_TURNS}"
    echo "DRYRUN total_epochs=${TOTAL_EPOCHS}"

    local n_cos=0 name
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        echo "DRYRUN cosine ${name}=${!name}"
        n_cos=$((n_cos + 1))
    done < <(compgen -e | grep -E '^COSINE_' | sort || true)
    echo "DRYRUN cosine_count=${n_cos}"

    local n_dist=0
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        echo "DRYRUN distill_env ${name}"
        n_dist=$((n_dist + 1))
    done < <(compgen -e | grep -E '^(BT_OPD_|HINT_OPD_|CONDITIONAL_TEACHER_UPDATE|ADAPTIVE_DISTILL_)' | sort || true)
    echo "DRYRUN distill_env_count=${n_dist}"

    local a
    for a in ${DISTILL_ARGS[@]+"${DISTILL_ARGS[@]}"}; do
        echo "DRYRUN distill_arg ${a}"
    done
    echo "DRYRUN distill_args_count=${#DISTILL_ARGS[@]}"
}

# ── PARITY: prove this arm is a one-variable delta from ttopd ─────────────────
# HINT_OPD_LOG_EVERY and OPSD_GOLD_CONDITIONING are this file's own additions —
# one is logging, the other pins off a signal the ttopd arm never sets — so they
# are the only lines allowed to be absent from the reference.

# Keep the config-bearing DRYRUN lines, drop status lines (ok/preflight) and the
# derived counts that move with the two exclusions above.
parity_filter() {
    grep -E '^DRYRUN (arm=|backbone=|student_gpus=|train_batch=|rollout_n=|max_turns=|total_epochs=|cosine |cosine_count=|distill_env |distill_arg |distill_args_count=)' \
        | grep -vE '^DRYRUN distill_env (HINT_OPD_LOG_EVERY|OPSD_GOLD_CONDITIONING)$' \
        | sort
}

check_parity() {
    if [ ! -f "$REFERENCE_ARM" ]; then
        echo "[fatal] reference arm not found: ${REFERENCE_ARM}"; return 1
    fi
    local ref mine diff_out
    # The reference runs in a subshell with this file's own two exports unset, so
    # --export=ALL leakage cannot make the reference look like it defines them.
    if ! ref=$(
        unset HINT_OPD_LOG_EVERY OPSD_GOLD_CONDITIONING
        DRY_RUN=1 ARM=ttopd BACKBONE="$BACKBONE" EXP="$EXP" TOOL_FORMAT="$TOOL_FORMAT" \
        INJECT_SCHEMAS="$INJECT_SCHEMAS" DATA_DIR="$DATA_DIR" TOTAL_EPOCHS="$TOTAL_EPOCHS" \
        MAX_TURNS="$MAX_TURNS" ROLLOUT_N="$ROLLOUT_N" STUDENT_GPUS="$STUDENT_GPUS" \
        TRAIN_BATCH="$TRAIN_BATCH" bash "$REFERENCE_ARM" 2>&1 | parity_filter
    ); then
        echo "[fatal] could not dry-run the reference arm ${REFERENCE_ARM}"; return 1
    fi
    mine=$(emit_dryrun | parity_filter)

    if diff_out=$(diff <(echo "$ref") <(echo "$mine")); then
        echo "[parity] ttopd_hints config is identical to train_hcgym.slurm ARM=ttopd"
        return 0
    fi
    echo "[fatal] ttopd_hints has DRIFTED from ARM=ttopd; the arms are no longer comparable."
    echo "[fatal] '<' = train_hcgym.slurm ARM=ttopd, '>' = this file. Re-sync, or set PARITY=0."
    echo "$diff_out"
    return 1
}

if [ "$PARITY" = "1" ]; then
    check_parity || exit 1
else
    echo "[warn] PARITY=0: not checking this arm against train_hcgym.slurm ARM=ttopd"
fi

# ── PREFLIGHT: refuse to train an inert mechanism twice ───────────────────────
if [ "${SKIP_HINT_CHECK:-0}" != "1" ]; then
    echo "[preflight] verifying that outcome-conditioned hints can fire in ${VERL_ROOT}"
    if ! python3 "${HCGYM_ROOT}/scripts/rebuttal/verify_hint_injection.py" \
            --verl "${VERL_ROOT}" --data "${DATA_DIR}/train.parquet"; then
        echo "[fatal] hint injection is still blocked in ${VERL_ROOT}."
        echo "[fatal] This arm exists only to test hints; refusing to burn 8 GPUs on an inert mechanism."
        exit 1
    fi
else
    echo "[warn] SKIP_HINT_CHECK=1: not verifying that hints can fire"
fi

if [ "$DRY_RUN" = "1" ]; then
    emit_dryrun
    if [ "$MISSING" -gt 0 ]; then
        echo "DRYRUN preflight=MISSING:${MISSING}"
    else
        echo "DRYRUN preflight=OK"
    fi
    echo "DRYRUN hints=live"
    echo "DRYRUN ok"
    exit 0
fi

cd "$VERL_ROOT"

# Two settings this B200 fleet requires, both found the hard way. They are stated
# here and not inline: a '#' inside a backslash-continued command terminates the
# command at that line, silently truncating every argument that follows it.
#
#   free_cache_engine=False   The release/resume path goes through
#     TorchMemorySaver, whose resume does an inplace write to a tensor built under
#     inference_mode and dies with "Inplace update to inference tensor outside
#     InferenceMode is not allowed". Both resume() and release() are gated on this
#     flag, so False sidesteps both. Applied to the teacher engine as well.
#
#   mm_attention_backend=triton_attn   verl hardcodes mm_attention_backend="fa3"
#     in workers/rollout/sglang_rollout/async_sglang_server.py, which is fatal on
#     Blackwell for VLM backbones. **engine_kwargs is splatted AFTER that default,
#     so the config value wins. Every backbone here is a VLM, teacher included.
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
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn \
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
    ${DISTILL_ARGS[@]+"${DISTILL_ARGS[@]}"} \
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
