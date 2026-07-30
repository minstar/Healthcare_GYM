#!/bin/bash
# Shared environment for Healthcare AI GYM rebuttal runs on the pt2 B200 fleet.
# Sourced by every sbatch script here. Never run directly.

# These three absolute paths are this cluster's layout, not anything portable.
# They are overridable from the environment so the harness can be pointed at
# another checkout or at a fixture tree, but the defaults are what every run in
# the paper used. The resolved values are echoed below, because a stale RUN_ROOT
# inherited through `sbatch --export=ALL` would otherwise silently redirect a run.
RUN_ROOT="${RUN_ROOT:-/data/project/private/minstar/workspace/hcgym_rebuttal}"
HCGYM_ROOT="${HCGYM_ROOT:-/data/project/private/minstar/workspace/minstar/Healthcare_GYM}"
VERL_ROOT="${VERL_ROOT:-/data/project/private/minstar/workspace/verl_ttopd}"
VENV="${RUN_ROOT}/.venv"

export RUN_ROOT HCGYM_ROOT VERL_ROOT
export HCGYM_RUN_ROOT="${RUN_ROOT}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${HCGYM_ROOT}:${HCGYM_ROOT}/scripts/verl:${PYTHONPATH:-}"
# $HOME is not mounted on pt2 compute nodes; ~/.local must never satisfy an import.
export PYTHONNOUSERSITE=1

# Every home-rooted cache has to be redirected or the job dies on the worker.
export HF_HOME="${RUN_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${RUN_ROOT}/.cache/xdg"
export FLASHINFER_WORKSPACE_BASE="${RUN_ROOT}/.cache/flashinfer"
export TORCH_EXTENSIONS_DIR="${RUN_ROOT}/.cache/torch_ext"
export TOKENIZERS_PARALLELISM=false
# Triton's JIT cache is per-job in tmpfs so concurrent jobs cannot corrupt each other.
# Outside Slurm the scratch is named `local_<pid>`, not a bare pid: prune_stale_scratch
# below only considers all-digit names, so an interactive run can never be mistaken
# for a job id that has since left the queue and swept out from under itself.
export TRITON_HOME="/tmp/hcgym_${SLURM_JOB_ID:-local_$$}"
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
export TMPDIR="${TRITON_HOME}/tmp"

# /tmp here is a 256GB tmpfs, so it is node RAM, and a preempted job is killed
# before it can clean up after itself. Leftovers accumulate until ray cannot spill
# and the job dies reporting something that looks nothing like a full disk (job
# 60769 ran with 9.3GB left of 256GB). Drop the scratch of jobs that are no longer
# queued; anything still in squeue is left alone, including this job's own
# directory, which has to survive a requeue.
#
# Covered by tests/test_scratch_prune.sh — this deletes directories, so it is not
# allowed to be obvious-looking and wrong.
prune_stale_scratch() {
    local root="$1" live dir id
    live=$(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '_' '\n' | grep -x '[0-9]\+' | sort -u)
    # No list means squeue failed, NOT that nothing is running: this runs inside a
    # job, so the current job is always in its own output. Prune only on real info.
    [ -n "$live" ] || return 0
    live="${live}
${SLURM_JOB_ID:-$$}"
    for dir in "$root"/hcgym_*; do
        [ -d "$dir" ] || continue
        id=${dir##*/hcgym_}
        case "$id" in *[!0-9]*) continue ;; esac
        printf '%s\n' "$live" | grep -qx "$id" || rm -rf -- "$dir"
    done
}
prune_stale_scratch /tmp

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" \
         "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" "${RUN_ROOT}/logs"

# Slurm on this cluster exports ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES
# with an identical mask (verified on a compute node: both "0,1,2,3,4,5,6,7").
# verl's worker refuses to start when both are set, so drop the ROCm one — the
# CUDA variable carries the same mask, so nothing about GPU assignment changes.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# B200 / sm100 specifics. Each of these was arrived at by a failure, so the reason
# is recorded next to it — none of them is a default worth carrying by habit.
#
# NCCL picks an interface by heuristic and on these nodes can choose one that does
# not carry inter-GPU traffic; the bond is the only one that does, and getting it
# wrong hangs collective init rather than erroring.
export NCCL_SOCKET_IFNAME=bond0
# One connection per device keeps NCCL kernels serialized against compute on
# sm100, which is what upstream verl's own Blackwell recipe sets.
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Deliberately NOT expandable_segments:True. verl runs actor and rollout colocated
# on the same GPUs, so sglang frees its KV cache through TorchMemorySaver, which
# refuses to run under expandable segments ("TorchMemorySaver is disabled for the
# current process because expandable_segments is not supported yet") and takes the
# scheduler down with it. expandable_segments only helps the async/disjoint layout.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
# No graph compilation anywhere. The multimodal backbones recompile per image
# shape, so dynamo spends longer tracing than the step saves, and a compiled
# graph is one more thing that has to survive the sleep/wake cycle.
export TORCHDYNAMO_DISABLE=1
# torch 2.9.1 + cuDNN < 9.15 trips a guard on the unused Conv3d vision path.
export SGLANG_DISABLE_CUDNN_CHECK=1
# The colocated layout hands the KV cache back to the allocator between rollouts,
# so sglang's idle-time memory audit sees a pool it did not release itself and
# aborts. TorchMemorySaver owns that memory here; the check does not know it.
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
# vLLM is not the rollout engine, but some verl import paths still construct its
# config; V1 is the only engine present in this pin.
export VLLM_USE_V1=1
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
# The CUDA runtime lives in the pip nvidia-* wheels, not on the system path. The
# sglang server is spawned as a bare subprocess and will not start without these
# ("libcudart.so.12: cannot open shared object file").
_SITE="${VENV}/lib/python3.12/site-packages"
_NVLIBS=$(find "${_SITE}/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_NVLIBS}${_SITE}/torch/lib:${VENV}/lib:${LD_LIBRARY_PATH:-}"

# Optional credential bundle, exported wholesale so any tool that wants an API key
# finds one. Nothing in the training or eval path needs it — local weights, a local
# sglang server and a local FTS5 index — so an absent file is not an error. Point
# HCGYM_ENV_FILE at your own bundle; the default lives in the run root, which is
# gitignored, so no credential path is ever published with this script.
HCGYM_ENV_FILE="${HCGYM_ENV_FILE:-${RUN_ROOT}/.env}"
if [ -f "$HCGYM_ENV_FILE" ]; then
    set -a; source "$HCGYM_ENV_FILE"; set +a
fi

echo "[env] node=$(hostname) job=${SLURM_JOB_ID:-none} venv=${VENV}"
echo "[env] run_root=${RUN_ROOT} hcgym_root=${HCGYM_ROOT} verl_root=${VERL_ROOT}"
echo "[env] $(python -c 'import torch,sglang;print(f"torch {torch.__version__} sglang {sglang.__version__} gpus {torch.cuda.device_count()}")' 2>/dev/null)"
