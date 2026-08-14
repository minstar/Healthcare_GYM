#!/bin/bash
#SBATCH --job-name=hcgym-eval
#SBATCH --partition=pt2_preemptible
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/data/project/private/minstar/workspace/hcgym_rebuttal/logs/slurm_%x_%j.log
# pt2_preemptible preempts with PreemptMode=REQUEUE and GraceTime=0, so a preempted
# job restarts under the SAME job id and reopens this same file. Slurm's default open
# mode is truncate, which erases the job's own earlier output every time it is
# requeued: q4b_grpo's checkpoints run contiguously from step 10 to 630 with an
# identical config either side, but its validation curve survives only for steps
# 10-190 and 390-600 -- the middle third was overwritten. Append instead.
#SBATCH --open-mode=append
#
# 20h, not the 12h this used to request. Measured on job 60628: the four text
# sets take 5:57:57 (medqa 2:43:57, mmlu 1:56:10, kqa_golden 0:15:10,
# medication_qa 1:02:40) and the two visual sets 2:25:39 + 3:51:30, i.e. 12:15:06
# for the suite -- over the old limit before a single FSDP shard is merged, and
# the merge has never once been run in this tree.
#
# Those timings were all taken with 3-4 clients sharing one server (see the PORT
# comment below), so a job that now gets its own server should be materially
# faster; 20h is headroom for the case where it is not. The cost of guessing low
# is a TIMEOUT that restarts the benchmark from row 0, because RESUME_FROM is
# deliberately not auto-detected.
#
#
# Sizing: one GPU. A 9B checkpoint serves comfortably in 192GB, and on a saturated
# partition a 1-GPU request packs into an existing gap while an --exclusive
# whole-node request waits for a node to drain.
#
# Slurm parses #SBATCH directives literally — it does NOT expand shell variables —
# so a bigger backbone needs the count in BOTH places, on the command line (which
# overrides the directive) and in EVAL_GPUS (which sets sglang's --tp-size):
#   sbatch --gres=gpu:4 --export=ALL,EVAL_GPUS=4,MODEL=...,TAG=... eval_agentic.slurm
# Multi-turn agentic benchmark evaluation: serve the checkpoint with sglang, then
# drive it through the GYM AgentRunner.
#
# Every condition in the rebuttal table runs through THIS script with the same
# MAX_TURNS and the same engine. The submitted paper could not difference its own
# columns because base/GRPO ran at 10 turns while TT-OPD ran at 5, and because
# Base+AR was largely sglang while the RL arms were transformers. Holding both
# fixed here is the point, so do not vary MAX_TURNS between conditions.
#
#   sbatch --export=ALL,MODEL=<path>,TAG=<name>,BENCH="medqa mmlu medmcqa" eval_agentic.slurm
#
# The multimodal tool-forwarding comparison runs the SAME binary twice, with one
# flag flipped and a distinct TAG per arm:
#   sbatch --export=ALL,MODEL=$Q9B,TAG=base_vqa_tools,BENCH="vqa_rad slake" eval_agentic.slurm
#   sbatch --export=ALL,MODEL=$Q9B,TAG=base_vqa_notools,BENCH="vqa_rad slake",MULTIMODAL_TOOLS=0 eval_agentic.slurm
# Each results JSON records which arm produced it under "multimodal_tool_forwarding".
set -euo pipefail

source /data/project/private/minstar/workspace/hcgym_rebuttal/runs/common_env.sh

MODEL="${MODEL:?set MODEL to a checkpoint directory}"
TAG="${TAG:?set TAG to a condition name}"
BENCH="${BENCH:-medqa mmlu medmcqa}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TEMPERATURE="${TEMPERATURE:-0.1}"
# Per-task raw turn dumps (<bench>_transcripts.jsonl). Default ON: rows keep
# only the extracted `submitted` string, so without the dump the no-answer
# class of a run can never be re-scored post-hoc -- the 2026-08 campaign had
# to re-run every affected arm for exactly that reason. ~15MB per arm x bench.
DUMP_TRANSCRIPTS="${DUMP_TRANSCRIPTS:-1}"
# Eval-only prompting baselines: default | strong_tool | react.
# "default" is the Base+AR condition and its prompt is byte-identical to HEAD, so
# adding modes here cannot move the reference row.
PROMPT_MODE="${PROMPT_MODE:-default}"
# Multimodal tool forwarding. Until 2026-07-29 the image path accepted a tool
# list and never sent it, so every VQA row under prompt_mode default/strong_tool
# measured a TOOLLESS agent under a heading that says tool-using. The fix is in;
# this flag keeps the broken behaviour reachable from the SAME binary so the two
# arms of that comparison differ in one flag and nothing else.
#   1 (default) — forward the catalog. Correct, and what you get if you forget.
#   0           — withhold it, reproducing the pre-fix measurement on purpose.
# Affects the image benchmarks only: text-only benchmarks never reach that call,
# and prompt_mode=react carries its catalog in the system prompt.
MULTIMODAL_TOOLS="${MULTIMODAL_TOOLS:-1}"
case "$MULTIMODAL_TOOLS" in
    1|0) ;;
    *) echo "[fatal] MULTIMODAL_TOOLS must be 1 (forward, default) or 0 (withhold), got '${MULTIMODAL_TOOLS}'"; exit 1 ;;
esac
export HCGYM_MULTIMODAL_TOOLS="$MULTIMODAL_TOOLS"
[ "$MULTIMODAL_TOOLS" = 1 ] && MM_ARM=forward || MM_ARM=withheld
# REFLEXION=1 swaps in the self-critique ladder (Shinn et al. 2023), which needs a
# different driver because it re-runs each task. Everything else — server, turn
# budget, decoding, benchmark list — stays identical, which is the whole point.
REFLEXION="${REFLEXION:-0}"
REFLEXION_STRATEGIES="${REFLEXION_STRATEGIES:-all}"
REFLEXION_MAX_ATTEMPTS="${REFLEXION_MAX_ATTEMPTS:-3}"
# Resume offset for a single benchmark. The harness writes <bench>_partial.json as
# it goes, so a timed-out job can be continued instead of rescored from zero. Four
# eval jobs died at the old 8h wall with slake unstarted and vqa_rad 250-420 of 451
# done; that work is recoverable with this.
#
# Deliberately NOT auto-detected from the partial file: silently skipping tasks
# because a stale partial happens to exist is how a run ends up scoring a subset
# and reporting it as the whole benchmark. Pass it explicitly when you mean it.
#
# The offset is applied to every benchmark in BENCH, so it only means anything when
# BENCH names one. Checked here as well as in the harness so a wrong invocation
# costs nothing instead of a GPU allocation and a model load.
RESUME_FROM="${RESUME_FROM:-0}"
# Scoring rule for the closed-vocabulary visual-QA sets (vqa_rad, slake, pathvqa).
# cf_em is the default; `substring` reproduces the pre-2026-07-29 published numbers
# and is what a partial written before that date has to be finished under, since a
# single accuracy may not be computed under two rules. Everything else is unaffected.
VQA_SCORER="${VQA_SCORER:-cf_em}"
case "$VQA_SCORER" in
    cf_em|substring) ;;
    *) echo "[fatal] VQA_SCORER must be cf_em or substring, got '${VQA_SCORER}'" >&2; exit 2 ;;
esac
if [ "$RESUME_FROM" -gt 0 ] && [ "$(echo $BENCH | wc -w)" -gt 1 ]; then
    echo "[fatal] RESUME_FROM=$RESUME_FROM with BENCH='$BENCH'." >&2
    echo "        The offset applies to every benchmark listed; resume one per job." >&2
    exit 2
fi
# A PORT AND A SERVER NAME THAT BELONG TO THIS JOB.
#
# Both used to be constants -- port 31000, served-model-name `hcgym` -- and the
# readiness gate below was satisfied by ANY server answering on that port with
# that name. #SBATCH --gres=gpu:1 is not exclusive, so several eval jobs land on
# one node routinely, and 12 of the 20 sglang logs in logs/ record
# `[Errno 98] ... bind on address ('127.0.0.1', 31000): address already in use`.
# Those jobs did not fail. They evaluated against a NEIGHBOUR's server: 61162,
# 61163 and 61164 all ran on c36f-i08-3-gpu189 through 60755's server and lost
# their backend within 88 ms of its SIGTERM.
#
# It was harmless only by luck -- every co-scheduled job so far served the same
# Qwen3.5-9B path, so the weights were right by accident. Serving distinct
# CHECKPOINTS under one constant name turns that into a results file that is
# complete, error-free, plausible, and scored against another arm's weights.
# There is no way to detect it afterwards.
#
# Three independent defences, because any one of them alone closes the hole and
# the cost of a miss is a number that cannot be retracted:
#   1. a port derived from the job id, so a collision is unlikely to begin with;
#   2. a served-model-name carrying the tag, so a neighbour's server cannot
#      satisfy this job's readiness check even on a shared port;
#   3. an assertion that the socket we are about to use is ours (below).
PORT="${PORT:-$((31000 + ${SLURM_JOB_ID:-$$} % 2000))}"
SERVED_NAME="hcgym-${TAG}"
OUT="${RUN_ROOT}/eval_results/${TAG}"
mkdir -p "$OUT"

# One output directory holds one arm. The results files are timestamped, so two
# arms landing in the same TAG would be indistinguishable by filename and only
# tellable apart by opening each JSON — exactly the misattribution this whole
# change exists to prevent. The path is NOT rewritten per arm, because
# launch_evals.sh's duplicate-submission guard keys on eval_results/<TAG>; give
# the second arm its own TAG instead.
MARKER="${OUT}/.multimodal_tools_arm"
if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" != "$MM_ARM" ]; then
    echo "[fatal] ${OUT} already holds '$(cat "$MARKER")' results; this job is '${MM_ARM}'."
    echo "        Use a different TAG so the two arms cannot be confused."
    exit 1
fi
echo "$MM_ARM" > "$MARKER"

# MODEL may be a raw verl checkpoint dir. Resolve it to something loadable,
# merging FSDP shards here on the compute node rather than at submit time — the
# merge loads the whole model. A weightless huggingface/ subdir is rejected by the
# resolver, which is the failure this guard exists to prevent.
if [ ! -f "${MODEL}/config.json" ] || ! ls "${MODEL}"/*.safetensors >/dev/null 2>&1; then
    echo "[eval] resolving checkpoint under ${MODEL}"
    RESOLVED=$("${RUN_ROOT}/runs/resolve_ckpt.sh" "$MODEL") || {
        echo "[fatal] no loadable checkpoint under ${MODEL}"; exit 1; }
    echo "[eval] using ${RESOLVED}"
    MODEL="$RESOLVED"
fi

echo "[eval] tag=${TAG} model=${MODEL}"
echo "[eval] prompt_mode=${PROMPT_MODE}"
echo "[eval] multimodal_tools=${MM_ARM} (HCGYM_MULTIMODAL_TOOLS=${MULTIMODAL_TOOLS}) — image benchmarks only"
echo "[eval] benchmarks='${BENCH}' max_turns=${MAX_TURNS} temperature=${TEMPERATURE}"

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --tp-size "${EVAL_GPUS:-1}" \
    --attention-backend triton \
    > "${RUN_ROOT}/logs/sglang_${TAG}_${SLURM_JOB_ID:-$$}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# 300 x 10s. The liveness check matters more than the budget: without it a dead
# server looks identical to a slow one and the job burns its whole allocation.
HEALTHY=0
for i in $(seq 1 300); do
    # Match the FULL served name, not the `hcgym` prefix. This is defence 2: a
    # neighbour's server answering on this port advertises its own tag, so it
    # cannot satisfy us, and we keep waiting for (or correctly fail on) our own.
    if curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${SERVED_NAME}\""; then
        HEALTHY=1; echo "[eval] server healthy after ~$((i * 10))s"; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[fatal] sglang died during startup"
        tail -60 "${RUN_ROOT}/logs/sglang_${TAG}_${SLURM_JOB_ID:-$$}.log"
        exit 1
    fi
    sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "[fatal] server never became healthy"; exit 1; }

# Defence 3: the socket we are about to send 4,721 questions to must be ours.
#
# The name check above is already decisive against a neighbour, but only while
# names stay distinct; this one holds regardless of what anyone else is called.
# sglang serves from a CHILD of $SERVER_PID, so accept any descendant.
#
# ss is not on every image. A missing tool must not block an eval -- defences 1
# and 2 already stand on their own -- so a failure to CHECK is a warning, while
# a check that RUNS and finds a stranger is fatal.
if command -v ss >/dev/null 2>&1; then
    LISTEN_PIDS=$(ss -ltnpH "sport = :${PORT}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
    if [ -z "$LISTEN_PIDS" ]; then
        echo "[warn] could not read the owner of port ${PORT}; relying on the served-name check"
    else
        OWNED=0
        for lp in $LISTEN_PIDS; do
            p="$lp"
            # Walk to init; stop if we meet our own server on the way up.
            while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ]; do
                [ "$p" = "$SERVER_PID" ] && { OWNED=1; break; }
                p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null)
            done
            [ "$OWNED" = 1 ] && break
        done
        if [ "$OWNED" != 1 ]; then
            echo "[fatal] port ${PORT} is held by pid(s) '${LISTEN_PIDS}', none descended from this job's server (${SERVER_PID})."
            echo "        Refusing to evaluate ${TAG} through another job's weights."
            exit 1
        fi
        echo "[eval] port ${PORT} confirmed owned by this job (server pid ${SERVER_PID})"
    fi
else
    echo "[warn] ss unavailable; port ownership not asserted (served-name check still applies)"
fi

# A template/parser mismatch produces a silently meaningless agentic eval, so
# confirm the model actually emits a parseable tool call before scoring anything.
python - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({
        "model": "hcgym",
        "messages": [{"role": "user", "content": "Look up the recommended first-line treatment for community-acquired pneumonia."}],
        "tools": [{"type": "function", "function": {
            "name": "search_evidence",
            "description": "Search the medical knowledge base.",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"}},
                           "required": ["query"]}}}],
        "max_tokens": 256, "temperature": 0.0,
    }).encode(),
    headers={"Content-Type": "application/json"})
msg = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]
calls = msg.get("tool_calls")
if calls:
    print(f"[probe] native tool call OK -> {calls[0]['function']['name']}")
else:
    # Not fatal: the GYM AgentRunner parses six textual tool-call formats itself.
    print(f"[probe] no native tool_calls field; AgentRunner will parse text. content={msg.get('content','')[:200]!r}")
PY

cd "$HCGYM_ROOT"
if [ "$REFLEXION" = "1" ]; then
    python scripts/rebuttal/run_reflexion_eval.py \
        --model_path "$MODEL" \
        --backend sglang \
        --server-url "http://127.0.0.1:${PORT}" \
        --benchmarks $BENCH \
        --strategies $REFLEXION_STRATEGIES \
        --max-attempts "$REFLEXION_MAX_ATTEMPTS" \
        --max-turns "$MAX_TURNS" \
        --max-samples "$MAX_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --output-dir "$OUT"
    echo "[done] eval ${TAG} $(date '+%F %T')"
    exit 0
fi

python scripts/eval_benchmark_multiturn.py \
    --model_path "$MODEL" \
    --backend sglang \
    --server-url "http://127.0.0.1:${PORT}" \
    --benchmarks $BENCH \
    --max-turns "$MAX_TURNS" \
    --max-samples "$MAX_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --no-think \
    --prompt-mode "$PROMPT_MODE" \
    --resume-from "$RESUME_FROM" \
    --vqa-scorer "$VQA_SCORER" \
    $([ "$DUMP_TRANSCRIPTS" = "1" ] && echo --dump-transcripts) \
    --output-dir "$OUT"

echo "[done] eval ${TAG} $(date '+%F %T')"
