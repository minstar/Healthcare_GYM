#!/bin/bash
#SBATCH --job-name=hcgym-think
#SBATCH --partition=pt2_preemptible
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --output=/data/project/private/minstar/workspace/hcgym_rebuttal/logs/slurm_%x_%j.log
# pt2_preemptible preempts with PreemptMode=REQUEUE and GraceTime=0, so a preempted
# job restarts under the SAME job id and reopens this same file. Slurm's default open
# mode is truncate, which erases the job's own earlier output every time it is
# requeued: q4b_grpo's checkpoints run contiguously from step 10 to 630 with an
# identical config either side, but its validation curve survives only for steps
# 10-190 and 390-600 -- the middle third was overwritten. Append instead.
#SBATCH --open-mode=append
#
#
# PROBE: does enable_thinking=True remove the degenerate phase?
#
# Measured baseline (q4b_grpo, thinking OFF, same everything else):
#   step 1: 18/21 degenerate   step 5: 19/21   step 15: 15/21   step 20: 2/21   step 30: 0/21
#
# Cause identified from the rollout dumps. With enable_thinking=False the chat
# template PREFILLS a closed empty block into each assistant turn:
#     <|im_start|>assistant\n<think>\n\n</think>\n\n        <- thinking OFF
#     <|im_start|>assistant\n<think>\n                        <- thinking ON
# The model then parrots that prefill. In one 47,646-char rollout it answered
# correctly, never emitted a stop token, invented its own `user` turn in Chinese,
# and role-played both sides until it hit the 12,288-token cap: 52 `<think>`
# blocks, 64 `user` and 64 `assistant` markers, 0 tool calls.
#
# NOT a context-length problem. Qwen3.5-4B has max_position_embeddings=262144;
# the 12,288 cap is a configured budget, not a model limit. Raising it would only
# buy the model more room to babble.
#
# This is a PROBE, not a replacement arm. runs/train_hcgym.slurm keeps
# enable_thinking=False because that is what the paper ran, and the degenerate
# phase is part of what it measured — quite possibly what it called "response
# length explosion". Changing the reproduction arm would hide that, not fix it.
#
# Prove the rebuilt stack end-to-end on a real B200 node before committing to
# long runs: sglang rollout, the bioagents tool bridge, the reward function, and
# the checkpoint path. Three GRPO steps on the smallest backbone is enough to
# exercise every one of them.
set -euo pipefail

source /data/project/private/minstar/workspace/hcgym_rebuttal/runs/common_env.sh

BACKBONE="${BACKBONE:-/data/project/public/checkpoints/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-${RUN_ROOT}/data/verl_parquet/full_4modality_clean}"
EXP="${EXP:-think_$(date +%H%M%S)}"

echo "[smoke] === stage 1: imports and tool bridge ==="
python - <<'PY'
import os, sys
import torch, sglang, transformers, verl
print(f"  torch {torch.__version__} | sglang {sglang.__version__} | transformers {transformers.__version__}")
print(f"  visible GPUs: {torch.cuda.device_count()} | {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"  compute capability: sm{cap[0]}{cap[1]}")

sys.path.insert(0, os.environ["HCGYM_ROOT"])
sys.path.insert(0, os.path.join(os.environ["HCGYM_ROOT"], "scripts", "verl"))
import bioagents_tool
print(f"  bioagents_tool OK -> {bioagents_tool.__file__}")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "reward_fn", os.path.join(os.environ["HCGYM_ROOT"], "scripts", "verl", "reward_fn.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
print(f"  reward_fn.compute_score OK -> {rf.compute_score.__name__}")
PY

echo "[smoke] === stage 2: three GRPO steps ==="
cd "$VERL_ROOT"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=7 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    data.return_raw_chat=True \
    data.tool_config_path=null \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path="${BACKBONE}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=7 \
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
    actor_rollout_ref.rollout.n=2 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=7 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${HCGYM_ROOT}/scripts/verl/tool_config_full.yaml" \
    actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=False \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${HCGYM_ROOT}/scripts/verl/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=hcgym-rebuttal \
    trainer.experiment_name="${EXP}" \
    trainer.default_local_dir="${RUN_ROOT}/checkpoints/${EXP}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=5 \
    trainer.rollout_data_dir="${RUN_ROOT}/rollout_dumps/${EXP}" \
    trainer.total_epochs=1

echo "[done] smoke ${EXP} $(date '+%F %T')"
