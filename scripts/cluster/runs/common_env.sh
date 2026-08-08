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
    local root="$1" raw rc live dir id grace="${SCRATCH_GRACE_MIN:-10}"

    # squeue's EXIT STATUS, captured on its own. Reading it off the pipeline would
    # report `sort`'s status instead, which is 0 whatever squeue did, so a squeue
    # that emitted a PARTIAL list before failing would look authoritative -- and
    # every running job missing from that partial list would have its scratch
    # deleted underneath it. ray spills into TMPDIR, so that corrupts live runs and
    # looks like a random cluster fault. launch_evals.sh already fails closed on
    # this exact question; the pruner has more to lose and must too.
    # Both guards below are written `cmd || fallback` rather than as a bare
    # assignment. The callers run under `set -euo pipefail`, and a bare
    # `x=$(failing-pipeline)` aborts the whole script at that line -- so the two
    # checks that exist to handle exactly that failure would never be reached.
    # That is not hypothetical: it silently killed every autoretry preflight the
    # moment `squeue` answered with an empty list, which made the harness refuse
    # to resubmit any arm at all, with no error printed anywhere.
    raw=$(squeue -u "$USER" -h -o '%i' 2>/dev/null) && rc=0 || rc=$?
    [ "$rc" -eq 0 ] || return 0

    # `grep` exits 1 when nothing matches, and under `pipefail` that is the
    # pipeline's status, so this one needs the `|| true` for the empty-list case.
    live=$(printf '%s\n' "$raw" | tr '_' '\n' | grep -x '[0-9]\+' | sort -u) || true
    # An empty list means squeue answered but listed nothing, which cannot be true
    # from inside a running job -- this job is always in its own output. Treat it
    # as no information rather than as "nothing is running".
    [ -n "$live" ] || return 0
    live="${live}
${SLURM_JOB_ID:-$$}"

    for dir in "$root"/hcgym_*; do
        [ -d "$dir" ] || continue
        id=${dir##*/hcgym_}
        # All-digit names only, so an interactive `local_<pid>` scratch is never a
        # candidate. An EMPTY id would also pass this test, and `hcgym_` is a
        # directory this harness never creates, so require a real id.
        [ -n "$id" ] || continue
        case "$id" in *[!0-9]*) continue ;; esac
        printf '%s\n' "$live" | grep -qx "$id" && continue
        # Second, independent condition: untouched for $grace minutes. A job that
        # started between the squeue snapshot above and this loop is absent from
        # the list but has a fresh directory, and one bad answer from squeue should
        # not be enough on its own to delete anything.
        [ -n "$(find "$dir" -maxdepth 0 -mmin +"$grace" 2>/dev/null)" ] || continue
        rm -rf -- "$dir"
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

# triton JIT-compiles a small cuda_utils.c the first time its NVIDIA driver is
# touched, and links it with `-lcuda`. It locates the directory by searching for
# libcuda.so.1 -- but gcc's -lcuda needs the DEVELOPMENT symlink libcuda.so, which
# the driver package does not always install. On a node without it every sglang
# server dies at startup with
#
#   /usr/bin/ld: cannot find -lcuda
#   subprocess.CalledProcessError: Command '['/usr/bin/gcc', ... cuda_utils.c ...
#
# surfacing as an opaque ray ActorDiedError. It is node-dependent, which is why
# some arms trained for 1,400 steps while others failed 45 times in a row without
# ever reaching step 1: whether a job runs came down to which node it landed on.
#
# Put a directory holding a real libcuda.so on the LINKER path. NVIDIA ships a
# stub for exactly this purpose -- it resolves the symbols at link time and the
# real driver library is what loads at run time via libcuda.so.1. Falls back to a
# job-local symlink when no stub is installed either.
_libcuda_linkdir() {
    local d
    for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/cuda/lib64/stubs; do
        [ -e "$d/libcuda.so" ] && { echo "$d"; return; }
    done
    for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
        if [ -e "$d/libcuda.so.1" ]; then
            mkdir -p "${TRITON_HOME}/libcuda" || return
            ln -sf "$d/libcuda.so.1" "${TRITON_HOME}/libcuda/libcuda.so" || return
            echo "${TRITON_HOME}/libcuda"
            return
        fi
    done
}
_LIBCUDA_DIR=$(_libcuda_linkdir)
if [ -n "$_LIBCUDA_DIR" ]; then
    export LIBRARY_PATH="${_LIBCUDA_DIR}:${LIBRARY_PATH:-}"
    export TRITON_LIBCUDA_PATH="${_LIBCUDA_DIR}"
else
    echo "[env] WARNING: no libcuda.so found; triton's driver JIT will fail on this node" >&2
fi

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
