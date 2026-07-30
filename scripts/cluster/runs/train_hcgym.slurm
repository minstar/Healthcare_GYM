#!/bin/bash
#SBATCH --job-name=hcgym-train
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
# Healthcare AI GYM — cross-backbone GRPO / GRPO+cosine / TT-OPD training.
#
# One template for every arm of the rebuttal. All arms hold the rollout budget,
# batch size and GPU count fixed and differ only in the method under test, so an
# arm-to-arm difference cannot be attributed to the training setup.
#
#   sbatch --export=ALL,BACKBONE=<path>,ARM=grpo,EXP=q9b_grpo               train_hcgym.slurm
#   sbatch --export=ALL,BACKBONE=<path>,ARM=grpo_cosine,EXP=q9b_grpo_cosine train_hcgym.slurm
#   sbatch --export=ALL,BACKBONE=<path>,ARM=ttopd,EXP=q9b_ttopd             train_hcgym.slurm
#
# The three arms form an additive ladder, so each component can be priced separately:
#
#   grpo         GRPO only
#   grpo_cosine  GRPO + cosine length-controlled reward          <- length control alone
#   ttopd        grpo_cosine + turn-level KL against a FROZEN teacher
#                + bidirectional sign flip + top-K filtering     <- TT-OPD as measured
#
# grpo_cosine minus grpo isolates the length reward; ttopd minus grpo_cosine is
# what TT-OPD contributes beyond it. That subtraction is only valid if the two
# cosine-bearing arms are shaped identically, so the cosine block below has
# exactly ONE definition site (cosine_reward_on) that both arms call.
#
# NOTE ON WHAT "ttopd" IS NOT. THREE of the mechanisms the paper names were inert
# in every published run, so they are absent here too — this arm reproduces what
# was measured, not what was described:
#
#   outcome-conditioned hints  never executed (v15-v32). See
#       Healthcare_GYM/scripts/rebuttal/verify_hint_injection.py, which fails at
#       rev e14d6a58 and passes on the fixed fork. HINT_OPD_ENABLED=0 below keeps
#       this arm faithful; runs/train_ttopd_hints.slurm is the arm that enables it.
#
#   truncation                 never executed. The "Truncated" in TT-OPD's name is
#       implemented only by _build_turn_mask (losses.py:442), gated on
#       `max_turn > 0` (losses.py:531). Every run script sets BT_OPD_MAX_TURN=0,
#       and their own comments say so: "0 = apply OPD to ALL turns (no truncation)".
#       There is no other truncation path — no L_max-based turn discarding exists
#       in the loss. Note "Turn-level" IS real: max_turn=0 is precisely what makes
#       the KL span the whole trajectory rather than the final response.
#
#   EMA teacher                never executed. ray_trainer.py:1849,1855 call
#       update_teacher_ema() and update_teacher_from_checkpoint(); neither method
#       exists on TeacherModelManager (hasattr is False for both). The block was
#       unreachable because self_distillation was undeclared and defaulted False,
#       so every published run trained against a teacher frozen at initialization.
#       Declaring the field made the flag live, so this arm now omits the override
#       rather than AttributeError at step 5.
#
# Keeping BT_OPD_MAX_TURN=0 and leaving self_distillation off is therefore
# deliberate. Turning either on would be a new experiment, not a bug fix, and must
# not be folded into the reproduction arm.
#
# Preemption on this partition is REQUEUE with GraceTime=0, so the run must be
# restartable from disk. verl's resume_mode=auto picks up the latest checkpoint;
# drive this script with autoretry.sh rather than relying on a single allocation.
#
# DRY_RUN=1 resolves the arm, prints the config it would train with on DRYRUN-
# prefixed lines, and exits before the trainer starts. tests/test_arms.sh uses it.
set -euo pipefail

source /data/project/private/minstar/workspace/hcgym_rebuttal/runs/common_env.sh

BACKBONE="${BACKBONE:?set BACKBONE to a model directory}"
ARM="${ARM:?set ARM to grpo, grpo_cosine or ttopd}"
EXP="${EXP:?set EXP to an experiment name}"
TOOL_FORMAT="${TOOL_FORMAT:-qwen3_coder}"
INJECT_SCHEMAS="${INJECT_SCHEMAS:-False}"
DATA_DIR="${DATA_DIR:-${RUN_ROOT}/data/verl_parquet/full_4modality_clean}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MAX_TURNS="${MAX_TURNS:-5}"
ROLLOUT_N="${ROLLOUT_N:-3}"
# 7 GPUs train the student in EVERY arm. In ttopd the 8th hosts the frozen EMA
# teacher; in grpo and grpo_cosine it is left idle on purpose, so the student
# topology (and therefore the effective batch geometry) is identical across arms
# and the arms differ only by method.
STUDENT_GPUS="${STUDENT_GPUS:-7}"
TRAIN_BATCH="${TRAIN_BATCH:-7}"
DRY_RUN="${DRY_RUN:-0}"
# Chat-template thinking mode. Default False = what every published run used, so
# the three reproduction arms are untouched by this knob.
#
# With False the template prefills a CLOSED empty block into each assistant turn
# (`<|im_start|>assistant\n<think>\n\n</think>\n\n`); with True it opens
# `<think>\n` and lets the model close it.
#
# A 5-step probe (job 60783) reported True at 0.0% degenerate against False at
# 23.9-38.1%, which read as "the response-length explosion is a template artifact".
# THE FULL RUNS SAY THE OPPOSITE, and the probe is the outlier. Matched steps of
# q4b_grpo (False) against q4b_grpo_think (True), degenerate/mean@1:
#
#     step      10     20     30     40     50     60     70     80
#     False  86.0%  29.1%  24.5%   7.8%   1.9%   0.2%   0.5%   0.5%
#     True   67.6%  67.4%  66.1%  67.2%  68.9%  57.1%  66.7%  70.6%
#
# False starts far worse and LEARNS OUT of it — 0.5% at step 80 and still 0.5% at
# step 660. True is pinned near 66% and is trending up where it stopped. Validation
# accuracy is indistinguishable over the same steps (0.2535 vs 0.2574), so this is
# not a quality trade. Why the probe disagreed at 5 steps is NOT explained; do not
# quote it. ENABLE_THINKING=1 runs the arm, but on this evidence it is worse.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
case "$ENABLE_THINKING" in
    1|true|True) THINK_FLAG=True ;;
    0|false|False) THINK_FLAG=False ;;
    *) echo "[fatal] ENABLE_THINKING must be 0 or 1, got '${ENABLE_THINKING}'"; exit 1 ;;
esac

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
echo "[run] data=${DATA_DIR} tool_format=${TOOL_FORMAT} student_gpus=${STUDENT_GPUS} enable_thinking=${THINK_FLAG}"

# ── Reward shaping ────────────────────────────────────────────────────────────
# The degenerate filter guards every arm; it is orthogonal to the method.
export REWARD_DEBUG_LOG=1
export DEGENERATE_FILTER=1
export DEGENERATE_EXCLUDE=1
export DEGENERATE_SENTINEL=-999.0
export DEGENERATE_NGRAM_THRESHOLD=0.3
export DEGENERATE_MIN_LENGTH=200
export DEGENERATE_GIBBERISH=1

# ── Method components, one definition each ────────────────────────────────────
# Every component has an explicit _on and _off. The _off variants matter because
# autoretry.sh submits with --export=ALL: a stale COSINE_REWARD=1 left in the
# submitting shell would otherwise silently give the plain grpo baseline length
# control and destroy the ablation. Arms never rely on a variable being absent.

# Cosine length-controlled reward (Yeo et al., arXiv:2502.03373).
# SOLE definition site. Called by both ttopd and grpo_cosine — do not copy these
# values into an arm branch, or the two arms stop being comparable.
COSINE_VAR_NAMES=(
    COSINE_REWARD COSINE_L_MAX COSINE_CHARS_PER_TOKEN
    COSINE_R0_CORRECT COSINE_RL_CORRECT
    COSINE_R0_WRONG COSINE_RL_WRONG COSINE_R_EXCEED
)

# One variable per line: tests/test_arms.sh greps this block to prove each value
# has exactly one definition site, and that check is only readable if the
# assignments are not doubled up.
cosine_reward_on() {
    export COSINE_REWARD=1
    export COSINE_L_MAX=12288
    export COSINE_CHARS_PER_TOKEN=5.0
    export COSINE_R0_CORRECT=1.1
    export COSINE_RL_CORRECT=0.7
    export COSINE_R0_WRONG=0.0
    export COSINE_RL_WRONG=-0.3
    export COSINE_R_EXCEED=-0.5
}

cosine_reward_off() {
    unset "${COSINE_VAR_NAMES[@]}"
}

# EMA teacher + outcome-conditioned privileged hints + turn-level truncation.
DISTILL_VAR_NAMES=(
    BT_OPD_MAX_TURN BT_OPD_BIDIRECTIONAL BT_OPD_MODEL_PATH BT_OPD_TOPK_RATIO
    HINT_OPD_ENABLED HINT_OPD_CORRECT HINT_OPD_INCORRECT
    CONDITIONAL_TEACHER_UPDATE ADAPTIVE_DISTILL_COEF
    ADAPTIVE_DISTILL_KL_DECAY_START ADAPTIVE_DISTILL_KL_OFF
    ADAPTIVE_DISTILL_COEF_FLOOR
)

DISTILL_ARGS=()

distillation_on() {
    export BT_OPD_MAX_TURN=0
    export BT_OPD_BIDIRECTIONAL=1
    export BT_OPD_MODEL_PATH="${BACKBONE}"
    export BT_OPD_TOPK_RATIO=0.5

    # Deliberately 0, and this is the whole point of the arm.
    #
    # Outcome-conditioned hints never executed in ANY published run (v15-v32):
    # on the batch path no tokenizer ever reached the hint builder, and on the
    # streaming path the hints were never forwarded. See
    # Healthcare_GYM/scripts/rebuttal/verify_hint_injection.py, which fails at
    # rev e14d6a58 and passes on the current fork.
    #
    # The fork now injects hints correctly. That makes this flag live, so leaving
    # it at 1 would quietly turn this arm into a DIFFERENT method than the one the
    # paper reports. This arm exists to reproduce the paper, so hints stay off and
    # TT-OPD here means GRPO + cosine + EMA teacher + turn truncation — exactly
    # what was measured. runs/train_ttopd_hints.slurm is the arm that turns them on.
    export HINT_OPD_ENABLED=0
    export HINT_OPD_CORRECT="The diagnostic reasoning appears sound, but could there be alternative diagnoses worth considering? Double-check whether the key findings truly support this conclusion."
    export HINT_OPD_INCORRECT="Some aspects of this reasoning may need re-examination. Have you fully considered all the clinical findings? It might be worth revisiting the differential diagnosis and verifying each step of the reasoning process."

    export CONDITIONAL_TEACHER_UPDATE=1
    export ADAPTIVE_DISTILL_COEF=1
    export ADAPTIVE_DISTILL_KL_DECAY_START=0.3
    export ADAPTIVE_DISTILL_KL_OFF=0.7
    export ADAPTIVE_DISTILL_COEF_FLOOR=0.1

    DISTILL_ARGS=(
        distillation.enabled=True
        # Deliberately absent: +distillation.self_distillation=True
        #
        # The teacher-refresh block it gates (ray_trainer.py:1849,1855) calls
        # update_teacher_ema() and update_teacher_from_checkpoint() — neither
        # method exists on TeacherModelManager. Verified: hasattr() is False for
        # both. So the block cannot ever have run; every published run had
        # self_distillation defaulting to False and a teacher frozen at
        # initialization, and turning it on now would AttributeError at step 5
        # (ema_update_interval=5).
        #
        # This arm reproduces what was measured, so the teacher stays frozen.
        # Implementing the EMA hot-swap is a NEW experiment and belongs in its own
        # arm, not folded in here — the same rule this file applies to
        # BT_OPD_MAX_TURN and HINT_OPD_ENABLED.
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
}

distillation_off() {
    unset "${DISTILL_VAR_NAMES[@]}"
    DISTILL_ARGS=()
}

# ── Arm selection ─────────────────────────────────────────────────────────────
case "$ARM" in
    grpo)
        cosine_reward_off
        distillation_off
        ;;
    grpo_cosine)
        cosine_reward_on
        distillation_off
        ;;
    ttopd)
        cosine_reward_on
        distillation_on
        ;;
    *)
        echo "[fatal] ARM must be grpo, grpo_cosine or ttopd, got '${ARM}'"; exit 1
        ;;
esac

if [ "$DRY_RUN" = "1" ]; then
    # Report the resolved state by enumerating the live environment, not by
    # echoing a hand-maintained list. A variable added to one arm and forgotten
    # in the other therefore cannot hide from tests/test_arms.sh.
    #
    # compgen -e (a bash builtin listing exported names) rather than `env`: on
    # this cluster `env` resolves to a uv PATH-shim in ~/.local/bin that prints
    # nothing, and $HOME is not even mounted on pt2 compute nodes, so `env`
    # would behave differently here than in the job. A builtin cannot be
    # shadowed by PATH.
    echo "DRYRUN arm=${ARM}"
    echo "DRYRUN backbone=${BACKBONE}"
    echo "DRYRUN student_gpus=${STUDENT_GPUS}"
    echo "DRYRUN train_batch=${TRAIN_BATCH}"
    echo "DRYRUN rollout_n=${ROLLOUT_N}"
    echo "DRYRUN max_turns=${MAX_TURNS}"
    echo "DRYRUN total_epochs=${TOTAL_EPOCHS}"

    n_cos=0
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        echo "DRYRUN cosine ${name}=${!name}"
        n_cos=$((n_cos + 1))
    done < <(compgen -e | grep -E '^COSINE_' | sort || true)
    echo "DRYRUN cosine_count=${n_cos}"

    n_dist=0
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        echo "DRYRUN distill_env ${name}"
        n_dist=$((n_dist + 1))
    done < <(compgen -e | grep -E '^(BT_OPD_|HINT_OPD_|CONDITIONAL_TEACHER_UPDATE|ADAPTIVE_DISTILL_)' | sort || true)
    echo "DRYRUN distill_env_count=${n_dist}"

    for a in ${DISTILL_ARGS[@]+"${DISTILL_ARGS[@]}"}; do
        echo "DRYRUN distill_arg ${a}"
    done
    echo "DRYRUN distill_args_count=${#DISTILL_ARGS[@]}"

    if [ "$MISSING" -gt 0 ]; then
        echo "DRYRUN preflight=MISSING:${MISSING}"
    else
        echo "DRYRUN preflight=OK"
    fi
    echo "DRYRUN ok"
    exit 0
fi

cd "$VERL_ROOT"

# Three settings this B200 fleet requires, all found the hard way:
#  free_cache_engine=False   sglang keeps its KV cache resident. The release/
#    resume path goes through TorchMemorySaver, whose resume does an inplace
#    write to a tensor built under inference_mode and dies. At
#    gpu_memory_utilization=0.3 on a 192GB card there is nothing to reclaim.
#  mm_attention_backend      verl hardcodes "fa3" in async_sglang_server.py,
#    which raises on Blackwell for any VLM backbone. engine_kwargs is splatted
#    after that default, so config wins.
#  attention_backend=triton  a SECOND, distinct override for the TEXT attention
#    path, which the same hardcoded "fa3" default also selects. Both are needed:
#    mm_attention_backend covers the vision tower, this one covers everything
#    else. It looks like a redundant duplicate of the line above and is not —
#    dropping it re-breaks text attention on sm100 while the multimodal path
#    keeps working, which is a confusing way to fail. Set on the rollout engine
#    and on the distillation teacher's engine, and on the eval server as
#    --attention-backend triton.

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
    +data.apply_chat_template_kwargs.enable_thinking="${THINK_FLAG}" \
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
