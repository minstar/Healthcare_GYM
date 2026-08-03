#!/bin/bash
# Decision tests for runs/watchdog.sh.
#
#   ./tests/test_watchdog.sh
#
# NOTHING IS SUBMITTED AND NOTHING IS CANCELLED. squeue, sbatch, sacct and pgrep
# are replaced by stubs on PATH that read fixture files, so every cluster state
# the watchdog cares about can be forced and the resulting decision asserted.
#
# What is real and what is stubbed, deliberately:
#   REAL     runs/watchdog.sh, and the discovery half of its delegation — each
#            scenario really executes launch_backbones.sh / launch_evals.sh in
#            PLAN=1 mode, which is read-only, so the candidate lists under test
#            are the launchers' actual answers and cannot drift from them.
#   STUBBED  squeue/sacct/pgrep (cluster state) and sbatch (which is never
#            reached anyway). The ACT half is asserted separately in the
#            "exec path" scenarios, which point WD_LAUNCH_* at a recorder.
#   FIXTURE  WD_LOGDIR / WD_STATE_DIR are a scratch dir per scenario, so slurm
#            logs, .done markers, quarantine files and strike counters are all
#            forced inputs and the real run root is never written to. WD_PROC is
#            a fixture tree too, so the argv guard on the only kill path can be
#            tested without a live process.
#
# Two couplings to the live tree remain. launch_backbones.sh hardcodes
# its own RUN_ROOT, so its readiness and .done checks read the real directory. If
# a backbone download breaks, or an arm genuinely completes, the candidate-list
# assertions below change — and they should, because the matrix changed. And
# launch_evals.sh skips a tag whose eval_results/<tag> exists, which would make
# the eval assertions fail as soon as the matrix produced real results; the
# scenarios that care call shadow_run_root() to empty just that directory.
set -uo pipefail
export LC_ALL=C

RUN_ROOT=/data/project/private/minstar/workspace/hcgym_rebuttal
REAL_RUN_ROOT="$RUN_ROOT"
WD="${RUN_ROOT}/runs/watchdog.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0
SCEN=""

scenario() { SCEN="$1"; echo; echo "── $1"; }
ok()  { PASS=$((PASS+1)); echo "  ok   — $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL — $1"; }

# has <desc> <regex>   — assert the last run's output matches
has()   { if grep -qE "$2" "$OUT"; then ok "$1"; else bad "$1 (no line matching /$2/)"; fi; }
hasnt() { if grep -qE "$2" "$OUT"; then bad "$1 (unexpected: $(grep -m1 -E "$2" "$OUT"))"; else ok "$1"; fi; }
count() { # count <desc> <regex> <n>
    local n; n="$(grep -cE "$2" "$OUT")"
    if [ "$n" = "$3" ]; then ok "$1 (${n})"; else bad "$1: expected $3, got $n"; fi
}

# ── stubs ─────────────────────────────────────────────────────────────────────
BIN="$WORK/bin"; mkdir -p "$BIN"

cat > "$BIN/squeue" <<'EOF'
#!/bin/bash
# Fixture-backed squeue. Supports the two invocations the tree uses:
#   squeue -u U -h -o '%i|%j|%T'          (watchdog)
#   squeue -u U -h -n NAME -o %T          (launch_evals.sh, autoretry.sh)
# STUB_SQUEUE_FAIL=1 simulates a controller that cannot be reached.
if [ "${STUB_SQUEUE_FAIL:-0}" = "1" ]; then
  echo "slurm_load_jobs error: Socket timed out on send/recv operation" >&2
  exit 1
fi
NAME=""; FMT="%i|%j|%T"
while [ $# -gt 0 ]; do
  case "$1" in
    -n) NAME="$2"; shift 2 ;;
    -o) FMT="$2"; shift 2 ;;
    -u) shift 2 ;;
    *)  shift ;;
  esac
done
[ -f "${STUB_SQUEUE:-/nonexistent}" ] || exit 0
while IFS='|' read -r id name state; do
  [ -n "${id:-}" ] || continue
  [ -n "$NAME" ] && [ "$name" != "$NAME" ] && continue
  case "$FMT" in
    "%T") echo "$state" ;;
    *)    echo "${id}|${name}|${state}" ;;
  esac
done < "$STUB_SQUEUE"
EOF

cat > "$BIN/sacct" <<'EOF'
#!/bin/bash
# sacct -j <id> -X -P --format=State --noheader
# STUB_SACCT_FAIL=1 simulates a slurmdbd outage.
if [ "${STUB_SACCT_FAIL:-0}" = "1" ]; then
  echo "sacct: error: slurm_persist_conn_open_without_init: failed to open persistent connection" >&2
  exit 1
fi
ID=""
while [ $# -gt 0 ]; do case "$1" in -j) ID="$2"; shift 2 ;; *) shift ;; esac; done
[ -f "${STUB_SACCT:-/nonexistent}" ] || exit 0
grep "^${ID}|" "$STUB_SACCT" 2>/dev/null | head -1 | cut -d'|' -f2
EOF

cat > "$BIN/pgrep" <<'EOF'
#!/bin/bash
# pgrep [-a] -f <pattern>, backed by a "pid<space>cmdline" fixture.
LIST=0; PAT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -af|-fa) LIST=1; shift ;;
    -a) LIST=1; shift ;;
    -f) shift ;;
    *)  PAT="$1"; shift ;;
  esac
done
[ -f "${STUB_PGREP:-/nonexistent}" ] || exit 1
OUT="$(grep -E "$PAT" "$STUB_PGREP" 2>/dev/null)"
[ -z "$OUT" ] && exit 1
if [ "$LIST" = 1 ]; then printf '%s\n' "$OUT"; else printf '%s\n' "$OUT" | awk '{print $1}'; fi
exit 0
EOF

cat > "$BIN/sbatch" <<'EOF'
#!/bin/bash
# Must never be reached by these tests. Records the attempt and fails loudly.
echo "STUB-SBATCH-CALLED $*" >> "${STUB_SBATCH:-/dev/null}"
echo "Submitted batch job 999999"
EOF

chmod +x "$BIN"/*
export PATH="$BIN:$PATH"

# A recorder standing in for a launcher, for the exec-path scenarios.
cat > "$WORK/recorder.sh" <<'EOF'
#!/bin/bash
if [ "${PLAN:-0}" = "1" ]; then
    printf '[plan] %s  fake=1\n' $FAKE_PLAN
    exit 0
fi
echo "RECORDED $1" >> "$REC_FILE"
echo "[submit] $1"
EOF
chmod +x "$WORK/recorder.sh"

# A launcher that spawns a long-lived background child, the way
# launch_backbones.sh nohups an autoretry loop. Used to prove the flock fd is
# not inherited.
cat > "$WORK/spawner.sh" <<'EOF'
#!/bin/bash
if [ "${PLAN:-0}" = "1" ]; then
    printf '[plan] %s  fake=1\n' $FAKE_PLAN
    exit 0
fi
nohup sleep 30 > /dev/null 2>&1 &
echo "$!" > "$SPAWN_PID_FILE"
echo "[submit] $1"
EOF
chmod +x "$WORK/spawner.sh"

# Write a /proc-style cmdline fixture: mkargv <procdir> <pid> <argv...>
mkargv() {
    local d="$1" pid="$2"; shift 2
    mkdir -p "${d}/${pid}"
    local a first=1
    for a in "$@"; do
        [ "$first" = 1 ] || printf '\0' >> "${d}/${pid}/cmdline"
        printf '%s' "$a" >> "${d}/${pid}/cmdline"
        first=0
    done
    printf '\0' >> "${d}/${pid}/cmdline"
}

# ── per-scenario fixture reset ────────────────────────────────────────────────
N=0
fresh() {
    N=$((N+1))
    ST="$WORK/s${N}"; mkdir -p "$ST"
    export WD_LOGDIR="$ST" WD_STATE_DIR="$ST"
    export STUB_SQUEUE="$ST/squeue.txt" STUB_SACCT="$ST/sacct.txt" \
           STUB_PGREP="$ST/pgrep.txt"  STUB_SBATCH="$ST/sbatch.calls"
    : > "$STUB_SQUEUE"; : > "$STUB_SACCT"; : > "$STUB_PGREP"; : > "$STUB_SBATCH"
    OUT="$ST/out.txt"
    unset WD_LAUNCH_BACKBONES WD_LAUNCH_EVALS REC_FILE FAKE_PLAN SPAWN_PID_FILE
    unset STUB_SQUEUE_FAIL STUB_SACCT_FAIL WD_USER WD_EVAL_REQUIRE_DONE
    # An EMPTY /proc by default, so the pgrep fixture is the only source of truth
    # about loops and a scenario cannot accidentally depend on what is running on
    # the machine. (It can: the first draft of S2 used 857833 as an invented pid,
    # and 857833 is a real live autoretry loop for a DIFFERENT arm on this host,
    # so the argv check rejected it — correctly — and the test failed.) Scenarios
    # that exercise the argv guard point WD_PROC at their own fixture tree.
    export WD_PROC="$ST/emptyproc"; mkdir -p "$WD_PROC"
    export WD_ONCE=1 WD_DRY_RUN=1 WD_MAX_TRAIN=2 WD_MAX_EVAL=3
    unset WD_LOG WD_LOCK
    # Undo any shadow a previous scenario installed, so the coupling to the live
    # tree is opt-in per scenario rather than leaking forward.
    export RUN_ROOT="$REAL_RUN_ROOT"
}

# A RUN_ROOT whose eval_results is empty and everything else is the real tree.
#
# launch_evals.sh skips any tag whose eval_results/<tag> already exists — correct
# behaviour, and the guard that stops a finished arm being rescored. But it means
# the eval assertions below go red the moment the matrix produces its first real
# results, testing the launcher's dedupe against live state instead of the
# watchdog's GPU cap. Everything except eval_results is symlinked, so the models,
# checkpoints and launcher scripts a scenario reads are still the real ones.
shadow_run_root() {
    local shadow="$ST/runroot" entry base
    mkdir -p "$shadow/eval_results"
    for entry in "$REAL_RUN_ROOT"/*; do
        base=$(basename "$entry")
        [ "$base" = eval_results ] && continue
        ln -sfn "$entry" "$shadow/$base"
    done
    export RUN_ROOT="$shadow"
}
run() { "$WD" > "$OUT" 2>&1; }

# A log for <exp>, job <id>, containing <body>.
mklog() { printf '%s\n' "$3" > "${ST}/slurm_hcgym-${1}_${2}.log"; }

# The real ImportError that killed job 60585, and the real argparse rejection
# from eval job 60592. Copied verbatim from logs/ in the run root.
FATAL_IMPORT="ImportError: cannot import name 'AutoModelForCausalLMWithValueHead' from 'trl' (/data/project/private/minstar/workspace/hcgym_rebuttal/.venv/lib/python3.12/site-packages/trl/__init__.py)"
FATAL_BENCH="eval_benchmark_multiturn.py: error: argument --benchmarks: invalid choice: 'mmlu_college_bio' (choose from 'medqa', 'medmcqa')"
# The line verl prints once an optimizer step has actually been applied.
GATE_LINE="step:1 - actor/pg_loss:0.031 - actor/grad_norm:0.42 - training/global_step:1 - training/epoch:0"

echo "watchdog decision tests — no sbatch, no scancel, no GPU"
echo "target: $WD"

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S1  empty queue, gate closed — launches the q4b validation wave only"
fresh; run
has  "in-flight is zero"                'INFLIGHT +train 0/2 node\(s\) \[\] +eval 0/3 gpu\(s\) \[\]'
has  "gate reported closed"             'GATE +CLOSED .* no optimizer step has completed yet'
has  "launches q4b:grpo"                'LAUNCH +train q4b:grpo '
has  "launches q4b:ttopd"               'LAUNCH +train q4b:ttopd '
hasnt "does not launch any q9b arm"     'LAUNCH +train q9b'
hasnt "does not launch any q27b arm"    'LAUNCH +train q27b'
hasnt "does not launch any glm9b arm"   'LAUNCH +train glm9b'
has  "counts what the gate is holding"  'GATE +9 non-q4b arm\(s\) are otherwise launchable and waiting'
hasnt "sbatch never invoked"            'STUB-SBATCH-CALLED'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S2  at the training cap — holds, asks the launcher for nothing"
fresh
printf '60612|hcgym-q4b_grpo|RUNNING\n' >> "$STUB_SQUEUE"
printf '857833 /bin/bash %s/runs/autoretry.sh q4b_ttopd /path/to/Qwen3.5-4B ttopd EXTRA=1\n' "$RUN_ROOT" >> "$STUB_PGREP"
run
has  "both slots counted (job + loop)"  'INFLIGHT +train 2/2'
has  "holds at cap"                     'HOLD +train :: at cap 2/2'
hasnt "launches nothing"                'LAUNCH +train'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S2b two concurrent autoretry loops are two slots, not one mangled name"
# Regression: live_loops() used printf '%s' per match, so with more than one loop
# running the sed output ran together and produced a phantom experiment
# "q4b_grpoq4b_ttopd" that occupied a third slot. Only reproducible with >1 loop.
fresh; export WD_MAX_TRAIN=99
printf '857833 /bin/bash %s/runs/autoretry.sh q4b_grpo /p/Qwen3.5-4B grpo EXTRA=1\n'  "$RUN_ROOT" >> "$STUB_PGREP"
printf '858050 /bin/bash %s/runs/autoretry.sh q4b_ttopd /p/Qwen3.5-4B ttopd EXTRA=1\n' "$RUN_ROOT" >> "$STUB_PGREP"
run
has  "exactly two slots, named apart"   'INFLIGHT +train 2/99 node\(s\) \[(q4b_grpo q4b_ttopd|q4b_ttopd q4b_grpo)\]'
hasnt "no concatenated phantom name"    'q4b_grpoq4b_ttopd|q4b_ttopdq4b_grpo'
has  "first loop reported live"         'STATUS +train:q4b_grpo = LIVE .*autoretry loop'
has  "second loop reported live"        'STATUS +train:q4b_ttopd = LIVE .*autoretry loop'

scenario "S2c a shell that merely MENTIONS a loop does not occupy that arm's slot"
# pgrep -f matches the joined command line. Counting an operator shell as a live
# loop is not destructive, but it withholds that arm's slot for as long as the
# shell lives — a silent stall. When /proc is readable the argv check settles it.
fresh; export WD_MAX_TRAIN=99
export WD_PROC="$ST/proc"; mkdir -p "$WD_PROC"
mkargv "$WD_PROC" 1610404 /bin/bash "${RUN_ROOT}/runs/autoretry.sh" q4b_grpo /p/m grpo
mkargv "$WD_PROC" 1613556 bash -c "nohup ${RUN_ROOT}/runs/autoretry.sh q4b_ttopd /p/m ttopd &"
printf '1610404 /bin/bash %s/runs/autoretry.sh q4b_grpo /p/m grpo\n' "$RUN_ROOT" >> "$STUB_PGREP"
printf '1613556 bash -c nohup %s/runs/autoretry.sh q4b_ttopd /p/m ttopd &\n' "$RUN_ROOT" >> "$STUB_PGREP"
run
has  "the real loop occupies a slot"    'INFLIGHT +train 1/99 node\(s\) \[q4b_grpo\]'
hasnt "the decoy shell does not"        'q4b_ttopd = LIVE'
# NOTE the limit of this fix. The watchdog's own accounting is now argv-accurate,
# but launch_backbones.sh:154 dedups with the same joined-cmdline `pgrep -f`, and
# it is the launcher — not the watchdog — that decides whether an arm is offered.
# So the decoy still withholds q4b:ttopd, one layer further out. That check is
# delegated by design; see WATCHDOG.md "Known limitations" for why it was left.
has  "the launcher is still fooled, and that is where it is decided" \
                                        'DELEGATE +launch_backbones: q4b:ttopd — autoretry loop already running'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S3  cap is not the limiter — the GATE is what holds the matrix back"
fresh; export WD_MAX_TRAIN=99; run
has  "gate closed"                      'GATE +CLOSED'
count "only the two q4b arms launch"    'LAUNCH +train ' 2
hasnt "no q9b despite 99 free slots"    'LAUNCH +train q9b'
hasnt "no q27b despite 99 free slots"   'LAUNCH +train q27b'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S4  gate passed — the full matrix is released"
fresh; export WD_MAX_TRAIN=99
mklog q4b_grpo 60700 "$GATE_LINE"
printf '60700|hcgym-q4b_grpo|RUNNING\n' >> "$STUB_SQUEUE"
printf '60700|RUNNING\n' >> "$STUB_SACCT"
run
has  "gate open, with its evidence"     'GATE +OPEN .*slurm_hcgym-q4b_grpo_60700\.log printed "training/global_step:"'
has  "releases q9b:grpo"                'LAUNCH +train q9b:grpo '
has  "releases the length-reward arm"   'LAUNCH +train q9b:grpo_cosine '
has  "releases q27b"                    'LAUNCH +train q27b:'
has  "releases the GLM family arm"      'LAUNCH +train glm9b:'
has  "running q4b arm not relaunched"   'SKIP +train q4b:grpo :: already in flight'

# gate is sticky across ticks
export WD_MAX_TRAIN=0; run
has  "gate stays open once proven"      'GATE +OPEN .*already proven'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S5  a config error — quarantined, said loudly, never relaunched"
fresh; export WD_MAX_TRAIN=99
mklog q4b_grpo 60800 "$GATE_LINE"          # open the gate so q9b is a candidate
mklog q9b_grpo 60801 "$FATAL_IMPORT"
printf '60801|FAILED\n' >> "$STUB_SACCT"
run
has  "classified FATAL, not retryable"  'STATUS +train:q9b_grpo = FATAL .*the venv cannot satisfy the code'
has  "says it loudly"                   'QUARANTINE +!!!! train:q9b_grpo WILL NOT BE RELAUNCHED'
has  "quotes the offending line"        "QUARANTINE +>>> ImportError: cannot import name 'AutoModelForCausalLMWithValueHead'"
has  "tells the operator how to clear"  'QUARANTINE +clear it with: rm .*\.wd_quarantine_train_q9b_grpo'
hasnt "q9b:grpo is not launched"        'LAUNCH +train q9b:grpo '
has  "healthy siblings still launch"    'LAUNCH +train q9b:ttopd '
if [ -f "${ST}/.wd_quarantine_train_q9b_grpo" ]; then ok "quarantine marker written"; else bad "no quarantine marker"; fi

# second tick — idempotent, not re-announced, still held
run
has  "second tick still holds it"       'STATUS +train:q9b_grpo = FATAL .*already quarantined, still held'
has  "and skips it with the reason"     'SKIP +train q9b:grpo :: quarantined'
hasnt "never launches it"               'LAUNCH +train q9b:grpo '
count "announced once, not twice"       'WILL NOT BE RELAUNCHED' 0
has  "but the held set is restated"     'HELD +1 quarantined, needing a human: *train:q9b_grpo'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S6  preemption — left alone for autoretry, NOT quarantined"
fresh; export WD_MAX_TRAIN=99
mklog q4b_grpo 60810 "$GATE_LINE"
mklog q9b_ttopd 60811 "slurmstepd: error: *** JOB 60811 ON node-171 CANCELLED AT 2026-07-28T23:40:00 DUE TO PREEMPTION ***"
printf '60811|PREEMPTED\n' >> "$STUB_SACCT"
run
has  "classified RETRY"                 'STATUS +train:q9b_ttopd = RETRY .*state=PREEMPTED — autoretry owns this'
hasnt "not quarantined"                 'QUARANTINE .*q9b_ttopd'
has  "and remains launchable"           'LAUNCH +train q9b:ttopd '

scenario "S6b node failure and wall-clock timeout are also retryable"
fresh; export WD_MAX_TRAIN=0
mklog q9b_grpo   60820 "slurmstepd: error: *** JOB 60820 ON node-186 CANCELLED AT 2026-07-28T01:00:00 DUE TO NODE FAILURE ***"
mklog q9b_ttopd  60821 "slurmstepd: error: *** JOB 60821 ON node-188 CANCELLED AT 2026-07-28T02:00:00 DUE TO TIME LIMIT ***"
run
has  "node failure -> RETRY"            'STATUS +train:q9b_grpo = RETRY .*node failure'
has  "time limit -> RETRY"              'STATUS +train:q9b_ttopd = RETRY .*wall clock'
hasnt "neither is quarantined"          'QUARANTINE'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S7  a completed arm — reported DONE, consumes no slot"
fresh; export WD_MAX_TRAIN=99
touch "${ST}/.autoretry_q9b_grpo.done"
mklog q9b_ttopd 60830 "[done] q9b_ttopd 2026-07-28 23:50:00"
run
has  "done marker recognised"           'STATUS +train:q9b_grpo = DONE .*\.autoretry_q9b_grpo\.done'
has  "completion sentinel recognised"   'STATUS +train:q9b_ttopd = DONE .*completion sentinel'
has  "neither counts as in flight"      'INFLIGHT +train 0/99'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S8  a second watchdog is already running — decides nothing"
fresh
flock -n "${ST}/.watchdog.lock" -c 'sleep 30' &
HOLDER=$!
sleep 0.4
run
kill "$HOLDER" 2>/dev/null; wait "$HOLDER" 2>/dev/null
has  "reports the held lock"            'LOCK +another watchdog already holds .* exiting without deciding'
hasnt "runs no tick"                    'BEGIN'
hasnt "launches nothing"                'LAUNCH'
hasnt "asks the launchers nothing"      'DELEGATE'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S9  other workstreams on the six-node slice are invisible"
fresh
printf '60561|pro4full-asis-lbc|RUNNING\n60589|pivotrl-v31|RUNNING\n60564|saas-baseline|PENDING\n' >> "$STUB_SQUEUE"
run
has  "they consume no hcgym slot"       'INFLIGHT +train 0/2 node\(s\) \[\] +eval 0/3 gpu\(s\) \[\]'
hasnt "pro4full never mentioned"        'pro4full'
hasnt "pivotrl never mentioned"         'pivotrl'
hasnt "saas never mentioned"            'saas'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S10 evals — capped by GPU, and a bad benchmark name is fatal not retryable"
fresh; shadow_run_root; export WD_MAX_EVAL=3
run
count "three 1-GPU evals launch"        'LAUNCH +eval ' 3
has  "the fourth is held at the cap"    'HOLD +eval .*cap reached \(1 more would be 4/3 gpu'

fresh; shadow_run_root; export WD_MAX_EVAL=9
mklog eval-base 60840 "$FATAL_BENCH"
printf '60840|FAILED\n' >> "$STUB_SACCT"
run
has  "argparse rejection is FATAL"      'STATUS +eval:base = FATAL .*rejected by argparse'
has  "eval base quarantined"            'QUARANTINE +!!!! eval:base WILL NOT BE RELAUNCHED'
hasnt "base eval not resubmitted"       'LAUNCH +eval base '
has  "its siblings still launch"        'LAUNCH +eval base_strong_tool '

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S11 an unrecognised failure — retried, then given up on after N strikes"
fresh; export WD_MAX_TRAIN=0 WD_UNKNOWN_STRIKES=3
mklog q9b_grpo 60850 "some novel crash nobody has a pattern for"
printf '60850|FAILED\n' >> "$STUB_SACCT"; run
has  "strike 1, keeps trying"           'STATUS +train:q9b_grpo = UNKNOWN .*strike 1/3, letting autoretry try again'
run
count "same job id is not a new strike" 'strike 2/3' 0
has  "still strike 1 (idempotent)"      'strike 1/3'
rm -f "${ST}"/slurm_hcgym-q9b_grpo_*.log
mklog q9b_grpo 60851 "another novel crash"; printf '60851|FAILED\n' >> "$STUB_SACCT"; run
has  "strike 2 on a new job id"         'strike 2/3'
rm -f "${ST}"/slurm_hcgym-q9b_grpo_*.log
mklog q9b_grpo 60852 "and another"; printf '60852|FAILED\n' >> "$STUB_SACCT"; run
has  "gives up at the third"            'STATUS +train:q9b_grpo = UNKNOWN .*strike 3/3 — giving up'
has  "and quarantines it"               'QUARANTINE +!!!! train:q9b_grpo WILL NOT BE RELAUNCHED'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S12 a fix releases the quarantine without human intervention"
fresh; export WD_MAX_TRAIN=99
mklog q4b_grpo 60860 "$GATE_LINE"
mklog q9b_grpo 60861 "$FATAL_IMPORT"; printf '60861|FAILED\n' >> "$STUB_SACCT"; run
has  "quarantined first"                'QUARANTINE +!!!! train:q9b_grpo'
rm -f "${ST}"/slurm_hcgym-q9b_grpo_*.log
mklog q9b_grpo 60862 "$GATE_LINE"; printf '60862|RUNNING\n' >> "$STUB_SACCT"; run
has  "newer clean job releases it"      'UNQUARANTINE +train:q9b_grpo released .*job 60862 is newer'
if [ -f "${ST}/.wd_quarantine_train_q9b_grpo" ]; then bad "marker still present"; else ok "quarantine marker removed"; fi

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S13 the exec path really invokes the launcher, once, with a narrow selector"
fresh
export WD_DRY_RUN=0 WD_MAX_TRAIN=1 WD_MAX_EVAL=0
export WD_LAUNCH_BACKBONES="$WORK/recorder.sh" WD_LAUNCH_EVALS="$WORK/recorder.sh"
export REC_FILE="$ST/recorded.txt" FAKE_PLAN="q9b:grpo q27b:ttopd"
: > "$REC_FILE"
run
count "exactly one launcher call"       'LAUNCH +  \| \[submit\]' 1
if [ "$(cat "$REC_FILE")" = "RECORDED q9b:grpo" ]; then
    ok "called with the single arm selector 'q9b:grpo', not 'all'"
else
    bad "recorder saw: $(tr '\n' ' ' < "$REC_FILE")"
fi
has  "the second candidate is held"     'HOLD +train q27b:ttopd :: cap reached'

# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSIONS FOUND IN REVIEW — each of these failed against the first version.
# ═══════════════════════════════════════════════════════════════════════════════
scenario "S15 BLOCKER: a dead log superseded by a queued retry must NOT be triaged"
# The first version derived state from the newest log FILE only. Against the real
# queue that quarantined q4b_grpo off job 60585's ImportError while job 60612 for
# the same arm was already PENDING with the venv fixed — and SIGTERMed the live
# autoretry loop. Since q4b is the gate backbone, the whole matrix stalled.
fresh; export WD_MAX_TRAIN=99
mklog q4b_grpo 60585 "$FATAL_IMPORT"
printf '60585|FAILED\n' >> "$STUB_SACCT"
printf '60612|hcgym-q4b_grpo|PENDING\n' >> "$STUB_SQUEUE"
printf '857833 /bin/bash %s/runs/autoretry.sh q4b_grpo /p/Qwen3.5-4B grpo EXTRA=1\n' "$RUN_ROOT" >> "$STUB_PGREP"
run
has  "the queued retry wins over the log" 'STATUS +train:q4b_grpo = LIVE .*job 60612 PENDING'
has  "and says the log is superseded"     'log 60585 is an earlier attempt and is NOT evidence'
hasnt "NOT quarantined"                   'QUARANTINE .*q4b_grpo'
hasnt "the live loop is NOT signalled"    'SIGTERM'
hasnt "no marker on disk"                 'clear it with'
if [ -f "${ST}/.wd_quarantine_train_q4b_grpo" ]; then bad "quarantine marker written anyway"; else ok "no quarantine marker on disk"; fi
# It is not double-submitted either — but note WHERE that is decided: the arm is
# never offered as a candidate at all, because launch_backbones.sh dedups on the
# live loop itself. The watchdog's own in-flight skip is defence in depth behind
# it, so the line to assert here is the delegated one.
has  "the launcher itself dedups it"      'DELEGATE +launch_backbones: q4b:grpo — autoretry loop already running'
hasnt "so it is never launched"           'LAUNCH +train q4b:grpo'

scenario "S15b a live LOOP alone does NOT suppress triage — that is the whole feature"
# The mirror image: if a live autoretry loop deferred triage, the config-error
# quarantine could never fire, because the loop is alive by construction.
fresh; export WD_MAX_TRAIN=99 WD_ON_FATAL=warn
mklog q4b_grpo 60585 "$FATAL_IMPORT"
printf '60585|FAILED\n' >> "$STUB_SACCT"
printf '857833 /bin/bash %s/runs/autoretry.sh q4b_grpo /p/Qwen3.5-4B grpo EXTRA=1\n' "$RUN_ROOT" >> "$STUB_PGREP"
run
has  "no job, loop only -> still triaged" 'STATUS +train:q4b_grpo = FATAL .*the venv cannot satisfy the code'
has  "and quarantined"                    'QUARANTINE +!!!! train:q4b_grpo WILL NOT BE RELAUNCHED'
has  "WD_ON_FATAL=warn spares the loop"   'WD_ON_FATAL=warn: leaving the autoretry loop alone'
unset WD_ON_FATAL

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S16 BLOCKER: an unreadable queue means DECIDE NOTHING, not 'nothing is running'"
# squeue failing and squeue being empty are the same empty string. The first
# version read a socket timeout as an idle cluster and resubmitted 3 evals per
# tick that were already PENDING.
fresh; export WD_DRY_RUN=0 WD_MAX_TRAIN=99 WD_MAX_EVAL=9
export WD_LAUNCH_BACKBONES="$WORK/recorder.sh" WD_LAUNCH_EVALS="$WORK/recorder.sh"
export REC_FILE="$ST/recorded.txt" FAKE_PLAN="q9b:grpo"
: > "$REC_FILE"
export STUB_SQUEUE_FAIL=1
run
has  "says the queue is unreadable"     'DEGRADED +cannot read the queue :: squeue exited 1: slurm_load_jobs error'
has  "explains why it is refusing"      'DEGRADED +.*duplicate jobs that are already queued'
has  "counts consecutive bad ticks"     'DEGRADED +consecutive degraded ticks: 1'
hasnt "launches nothing at all"         'LAUNCH'
hasnt "triages nothing at all"          'STATUS'
hasnt "asks no launcher"                'DELEGATE'
if [ ! -s "$REC_FILE" ]; then ok "no launcher was executed"; else bad "recorder saw: $(cat "$REC_FILE")"; fi
hasnt "sbatch never invoked"            'STUB-SBATCH-CALLED'
# and it recovers on its own once the controller answers again
unset STUB_SQUEUE_FAIL
export WD_MAX_TICKS=0
run
has  "recovers without intervention"    'LAUNCH +train q9b:grpo'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S17 MAJOR: USER unset (cron/systemd/container) must not blind the queue view"
# `squeue -u "$USER"` under `set -u` died inside a command substitution: the error
# went to stderr, which the documented `nohup ... >/dev/null 2>&1` discards, and
# the tick carried on against an apparently empty queue.
fresh
printf '60628|hcgym-eval-base|PENDING\n60612|hcgym-q4b_grpo|RUNNING\n' >> "$STUB_SQUEUE"
( unset USER; "$WD" > "$OUT" 2>&1 )
hasnt "no unbound-variable death"       'unbound variable'
has  "the queue is still read"          'INFLIGHT +train 1/2 node\(s\) \[q4b_grpo\] +eval 1/3 gpu\(s\) \[base\]'
has  "and the user it resolved is logged" 'START .*user=[a-z]'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S18 MAJOR: a slurmdbd outage defers — it must not charge a strike"
# With sacct down, a healthy RUNNING arm was classified from its partial log and
# collected three strikes in three ticks, then got quarantined and its loop killed.
fresh; export WD_MAX_TRAIN=0 WD_UNKNOWN_STRIKES=3
mklog q9b_grpo 70001 "(TaskRunner pid=123) step 4 of 60, rollout ok, reward mean 0.31"
export STUB_SACCT_FAIL=1
run; run; run
has  "classified DEFER"                 'STATUS +train:q9b_grpo = DEFER .*sacct exited 1'
count "never charges a strike"          'strike [0-9]+/3' 0
hasnt "never quarantines"               'QUARANTINE'
if [ -f "${ST}/.wd_strikes_train_q9b_grpo" ]; then bad "a strike file was written"; else ok "no strike file after 3 ticks"; fi

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S19 MAJOR: the kill path checks ARGV, not a substring of the command line"
# `[[ $cmd != */autoretry.sh\ $exp\ * ]]` matched anywhere in the joined cmdline,
# so a plain shell whose command merely MENTIONED the arm was a kill target. It
# hit the author's own session twice during development.
fresh; export WD_MAX_TRAIN=99 WD_DRY_RUN=1
export WD_PROC="$ST/proc"; mkdir -p "$WD_PROC"
mklog q9b_grpo 60870 "$FATAL_IMPORT"
printf '60870|FAILED\n' >> "$STUB_SACCT"
mklog q4b_grpo 60869 "$GATE_LINE"
# 1610404 is a real loop; 1613556 is an operator shell that merely mentions it.
mkargv "$WD_PROC" 1610404 /bin/bash "${RUN_ROOT}/runs/autoretry.sh" q9b_grpo /p/m grpo "EXTRA=1"
mkargv "$WD_PROC" 1613556 bash "$ST/decoy.sh" "nohup ${RUN_ROOT}/runs/autoretry.sh q9b_grpo /p/m grpo EXTRA=1 &"
printf '1610404 /bin/bash %s/runs/autoretry.sh q9b_grpo /p/m grpo EXTRA=1\n' "$RUN_ROOT" >> "$STUB_PGREP"
printf '1613556 bash %s/decoy.sh nohup %s/runs/autoretry.sh q9b_grpo /p/m grpo EXTRA=1 &\n' "$ST" "$RUN_ROOT" >> "$STUB_PGREP"
run
has  "the genuine loop is a target"     'would SIGTERM autoretry loop pid=1610404'
has  "the decoy shell is refused"       'pid 1613556: argv is not an autoretry.sh invocation for q9b_grpo, refusing to signal'
count "exactly one kill target"         'would SIGTERM' 1

scenario "S19b a pid with no readable /proc entry is refused, not signalled"
fresh; export WD_MAX_TRAIN=99 WD_DRY_RUN=1
export WD_PROC="$ST/proc"; mkdir -p "$WD_PROC"
mklog q9b_grpo 60871 "$FATAL_IMPORT"; printf '60871|FAILED\n' >> "$STUB_SACCT"
printf '999001 /bin/bash %s/runs/autoretry.sh q9b_grpo /p/m grpo\n' "$RUN_ROOT" >> "$STUB_PGREP"
run
has  "unreadable cmdline is refused"    'pid 999001: cannot read .*/999001/cmdline, refusing to signal'
hasnt "and nothing is signalled"        'would SIGTERM'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S20 BLOCKER for the results: an arm's eval waits until that arm is DONE"
# resolve_ckpt.sh --check answers 2 from global_step_10 (save_freq=10) while the
# arm still has fifty steps to run. Scoring then populates eval_results/<tag>,
# which launch_evals.sh skips forever after — so the rebuttal table would carry
# step-10 numbers for arms that trained to completion, silently.
fresh; export WD_DRY_RUN=0 WD_MAX_TRAIN=0 WD_MAX_EVAL=9
export WD_LAUNCH_EVALS="$WORK/recorder.sh" WD_LAUNCH_BACKBONES="$WORK/recorder.sh"
export REC_FILE="$ST/recorded.txt" FAKE_PLAN="q9b_grpo"
: > "$REC_FILE"
mklog q9b_grpo 60880 "(TaskRunner pid=1) step 12 of 60"
printf '60880|hcgym-q9b_grpo|RUNNING\n' >> "$STUB_SQUEUE"
printf '60880|RUNNING\n' >> "$STUB_SACCT"
run
has  "held while the arm is LIVE"       'HOLD +eval q9b_grpo :: arm q9b_grpo is LIVE, not DONE'
has  "and says exactly why"             'save_freq=10 .*make launch_evals skip the finished model forever'
if [ ! -s "$REC_FILE" ]; then ok "no eval submitted mid-training"; else bad "recorder saw: $(cat "$REC_FILE")"; fi
# now the arm finishes
touch "${ST}/.autoretry_q9b_grpo.done"
: > "$STUB_SQUEUE"
run
has  "released once the arm is DONE"    'LAUNCH +eval q9b_grpo ::'
if grep -q "RECORDED q9b_grpo" "$REC_FILE"; then ok "eval submitted only after DONE"; else bad "recorder saw: $(cat "$REC_FILE")"; fi

scenario "S20b untrained base* rows are never held — they need no checkpoint"
fresh; shadow_run_root; export WD_MAX_TRAIN=0 WD_MAX_EVAL=9; run
has  "base launches immediately"        'LAUNCH +eval base ::'
hasnt "and is never held for training"  'HOLD +eval base .*not DONE'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S21 MAJOR: the eval cap counts GPUs, so a 4-GPU q27b eval costs 4"
# The cap was documented as GPUs but counted one slot per job, so three q27b
# evals could take 12 GPUs — 1.5 whole nodes — under a cap that read "3".
fresh; export WD_DRY_RUN=1 WD_MAX_TRAIN=0 WD_MAX_EVAL=4
export WD_LAUNCH_EVALS="$WORK/recorder.sh" WD_LAUNCH_BACKBONES="$WORK/recorder.sh"
export FAKE_PLAN="q27b_grpo q27b_ttopd"
touch "${ST}/.autoretry_q27b_grpo.done" "${ST}/.autoretry_q27b_ttopd.done"
run
has  "the first q27b eval fits exactly" 'LAUNCH +eval q27b_grpo :: room \(0\+4/4 gpu'
has  "the second cannot"                'HOLD +eval q27b_ttopd :: cap reached \(4 more would be 8/4 gpu'

scenario "S21b a cap that can never admit a condition says so instead of idling"
fresh; export WD_DRY_RUN=1 WD_MAX_TRAIN=0 WD_MAX_EVAL=3
export WD_LAUNCH_EVALS="$WORK/recorder.sh" WD_LAUNCH_BACKBONES="$WORK/recorder.sh"
export FAKE_PLAN="q27b_grpo"
touch "${ST}/.autoretry_q27b_grpo.done"
run
has  "names the impossible condition"   'HOLD +eval q27b_grpo :: needs 4 gpu\(s\) but WD_MAX_EVAL=3 .* can NEVER be admitted'

scenario "S21c a live q27b eval is counted as 4 GPUs, not 1"
fresh; export WD_MAX_TRAIN=0 WD_MAX_EVAL=4
printf '60890|hcgym-eval-q27b_grpo|RUNNING\n' >> "$STUB_SQUEUE"
run
has  "in-flight shows 4 GPUs for one job" 'INFLIGHT +.*eval 4/4 gpu\(s\) \[q27b_grpo\]'
has  "and the eval cap is reached"        'HOLD +eval :: at cap 4/4 gpu'

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S22 MINOR: strikes are cleared when an arm looks healthy again"
# Strikes only ever incremented, so they accumulated over a whole night and a
# single late hiccup could quarantine an arm that had long since recovered.
fresh; export WD_MAX_TRAIN=0 WD_UNKNOWN_STRIKES=3
mklog q9b_grpo 60900 "some novel crash"; printf '60900|FAILED\n' >> "$STUB_SACCT"; run
has  "strike 1 recorded"                'strike 1/3'
if [ -f "${ST}/.wd_strikes_train_q9b_grpo" ]; then ok "strike file written"; else bad "no strike file"; fi
rm -f "${ST}"/slurm_hcgym-q9b_grpo_*.log
mklog q9b_grpo 60901 "(TaskRunner pid=1) step 3 of 60"; printf '60901|RUNNING\n' >> "$STUB_SACCT"; run
has  "recovery clears the counter"      'STRIKE +train:q9b_grpo strike counter cleared'
if [ -f "${ST}/.wd_strikes_train_q9b_grpo" ]; then bad "strike file survived recovery"; else ok "strike file removed"; fi

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S23 MAJOR: the flock fd is not inherited by the loops the launcher spawns"
# `exec 9>>lock` is not close-on-exec, so the 72h autoretry loops inherited it.
# A watchdog that died at 01:00 could not be restarted until they exited: the
# lock was held by its own grandchildren.
fresh
export WD_DRY_RUN=0 WD_MAX_TRAIN=1 WD_MAX_EVAL=0
export WD_LAUNCH_BACKBONES="$WORK/spawner.sh" WD_LAUNCH_EVALS="$WORK/spawner.sh"
export FAKE_PLAN="q9b:grpo" SPAWN_PID_FILE="$ST/spawn.pid"
run
CHILD="$(cat "$ST/spawn.pid" 2>/dev/null || echo 0)"
if [ "$CHILD" != 0 ] && kill -0 "$CHILD" 2>/dev/null; then ok "launcher spawned a live background child (pid ${CHILD})"
else bad "no live child spawned, the scenario proves nothing"; fi
if flock -n "${ST}/.watchdog.lock" -c 'true'; then ok "a restarted watchdog can retake the lock while the child still runs"
else bad "lock still held by the inherited child fd"; fi
if [ -e "/proc/${CHILD}/fd/9" ]; then bad "child still holds fd 9 on the lock"; else ok "child has no fd 9"; fi
kill "$CHILD" 2>/dev/null

# ═══════════════════════════════════════════════════════════════════════════════
scenario "S24 static guarantees"
if [ "$(grep -c 'scancel' "$WD")" = 0 ]; then ok "the word scancel does not appear in watchdog.sh"
else bad "watchdog.sh mentions scancel"; fi
if grep -q 'case "\$name" in hcgym-\*)' "$WD"; then ok "squeue view is filtered to hcgym-* at the source"
else bad "no hcgym-* filter found"; fi
if bash -n "$WD"; then ok "bash -n clean"; else bad "syntax error"; fi
if bash -n "${RUN_ROOT}/runs/launch_evals.sh"; then ok "launch_evals.sh bash -n clean"; else bad "launch_evals.sh syntax error"; fi
if grep -q 'refusing to submit rather than risk a duplicate' "${RUN_ROOT}/runs/launch_evals.sh"; then
    ok "launch_evals.sh fails closed on an unreadable queue"
else bad "launch_evals.sh still fails open on squeue error"; fi
if [ "$(grep -c '9>&-' "$WD")" -ge 3 ]; then ok "every child-spawning site closes the lock fd"
else bad "a child-spawning site still inherits fd 9"; fi

echo
echo "──────────────────────────────────────────"
echo "passed ${PASS}   failed ${FAIL}"
[ "$FAIL" = 0 ] || exit 1
