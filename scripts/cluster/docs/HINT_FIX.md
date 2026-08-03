# TT-OPD outcome-conditioned privileged hints: what was broken, what changed, what may be claimed

Date: 2026-07-28
verl fork: `/data/project/private/minstar/workspace/verl_ttopd`, branch `feat/model-family-portability`
Fix commit: `f2849ca3` (defective shipped code: `e14d6a58`). Committed, not pushed.
Checker: `Healthcare_GYM/scripts/rebuttal/verify_hint_injection.py` (untracked; that repo
has other agents' uncommitted work on `main`, so nothing was committed there)

---

## 1. What was broken

**TT-OPD's outcome-conditioned privileged hints never executed. Not once, in any run
that produced a number in the paper.** The paper names them as one of TT-OPD's three
mechanisms. The code that builds them was unreachable under every configuration that
was ever launched.

### Since when

Hints were introduced in `run_qwen35_9b_self_distill_v15_ema_hint.sh`. Every run
script from **v15 through v32** exports `HINT_OPD_ENABLED=1`, and hints fired in none
of them. Two different blockers, split cleanly by run version:

| Runs | teacher path | blocked by |
|---|---|---|
| v15–v25 (+ `run_ttopd_v16b.sh`, `run_3seed_ttopd_v16.sh`) | `enable_resource_pool=False` → batch path | **no tokenizer** |
| v26–v32 | `enable_resource_pool=True` → streaming path | **hints never forwarded** |

Verified mechanically:

```
$ grep -l 'HINT_OPD_ENABLED=1' scripts/verl/run_qwen35_9b_self_distill_v*.sh   # v15..v32
$ grep -o 'enable_resource_pool=[A-Za-z]*' <each>                              # False v15-v25, True v26-v32
```

### The three gates, precisely

1. **Streaming path never forwarded hints.** `AgentLoopWorker.__init__` sets
   `stream_teacher_with_rollout = distillation_config.teacher_model.enable_resource_pool`.
   When true, `_compute_teacher_logprobs` calls `compute_teacher_logprobs_single(...)`
   — and that call site passed only `sequence_ids` and `multi_modal_data`. The batch
   path, which is the only place that ever built hint tokens, is gated on
   `if self.distillation_enabled and not self.stream_teacher_with_rollout`, so with a
   teacher resource pool it is dead code. v26–v32 all set
   `enable_resource_pool=True`.

2. **No tokenizer ever reached the hint builder.** `AsyncTeacherLLMServerManager.__init__`
   took `tokenizer=None` as a default, and *neither* of the two construction sites
   (`agent_loop.py`, `teacher_loop/teacher_model.py`) passed one. Hint construction is
   guarded on `tokenizer is not None`, so the guard was False for every sample on the
   batch path too. The documented fallback `self.config.tokenizer` does not resolve:
   `hasattr()` on a `DictConfig` for a missing key returns False. This is what blocked
   v15–v25.

3. **Multimodal samples are skipped** (inserting tokens shifts image placeholder
   positions). See §5 — in these runs this suppressed exactly zero samples, so it is a
   scope restriction, not a cause.

### Net

> **TT-OPD as actually executed = GRPO + cosine length reward + EMA teacher +
> turn-level truncation.** The privileged-hint mechanism contributed nothing, because
> it never ran. Every teacher logprob in every reported TT-OPD number was computed on
> the bare student sequence, with no privileged conditioning of any kind.

Reproduce the historical defect (exits non-zero):

```
python scripts/rebuttal/verify_hint_injection.py --rev e14d6a58
```

---

## 2. What changed

### `verl/experimental/teacher_loop/teacher_manager.py`

* **New `resolve_privileged_injection(manager, tokenizer, *, has_multimodal, gold_text,
  reward_score) -> (token_ids | None, reason)`** — one decision point, called by *both*
  teacher paths, so they cannot drift apart again. Returns a stable reason string
  (`hint:correct`, `hint:incorrect`, `gold`, `skip:score-missing`, `skip:multimodal`,
  `skip:gold-missing`, `skip:no-tokenizer`, `skip:render-failed`, `skip:disabled`) that
  the tests and the checker assert on.
* **Precedence is explicit and announced.** Gold conditioning (the OPSD baseline's
  privileged signal) and outcome-conditioned hints are mutually exclusive. Gold wins.
  Enabling both prints a one-time warning naming the loser. When gold is enabled but a
  sample has no ground truth there is deliberately **no** fallback to hints — mixing two
  different privileged signals across the samples of one run would make the arm
  uninterpretable; those samples are scored unconditioned.
* **New `hint_correctness(score) -> bool | None`.** Returns `None` — never a guess — for
  a missing, non-numeric or NaN score, and callers suppress injection on `None`. Telling
  the teacher that a wrong trajectory was correct is a worse signal than no signal.
  Threshold is `score > 0.0`, now named `HINT_OPD_CORRECT_THRESHOLD` instead of a bare
  literal. Under the hcgym cosine reward a correct trajectory scores ≥ 0.7 and a wrong
  one lands in `[-0.5, 0.0]`, so 0.0 separates them; the degenerate sentinel `-999.0`
  reads as incorrect, which it is.
* **`tokenizer` is now a REQUIRED parameter** of `AsyncTeacherLLMServerManager.__init__`
  — no default at all. Passing `None` explicitly while conditioning is enabled **raises**
  rather than degrading silently; with conditioning off it prints a warning. A required
  parameter cannot be forgotten, which is exactly how this broke.
* **Batch path refactored** onto the shared resolver, and its per-batch log line replaced
  with a greppable reason histogram.

### `verl/experimental/agent_loop/agent_loop.py`

* `_compute_teacher_logprobs` now calls the shared resolver and forwards
  `hint_token_ids` + `original_length` on the streaming path, for hints as well as gold.
* `is_correct` comes from **`output.reward_score`**, deposited by `await self._compute_score(...)`
  on the line immediately before the teacher call in `_agent_loop_postprocess`. That
  score is populated whenever the async reward loop is active, which is whenever there
  is no separate reward model (`enable_agent_reward_loop = not self.use_rm`) — i.e.
  always, in this study, since the reward is a custom function. When it is `None`,
  injection is suppressed.
* New `_log_injection`: announces the first occurrence of each reason, then a histogram
  every `HINT_OPD_LOG_EVERY` (default 64) samples. `grep 'privileged injection'` on a
  training log now answers "did hints fire, and on how many samples".
* Both construction sites already pass a tokenizer (from `0aa9d647`); the comments now
  say why, and the required parameter enforces it.

### `Healthcare_GYM/scripts/rebuttal/verify_hint_injection.py`

Rewritten. Gate 2's construction-site scan was a **regex false negative**: `[^)]*`
stopped at the first `)` — which appeared inside a comment and inside a nested
`list(zip(...))` — so it reported "no site passes tokenizer" even after `0aa9d647` had
fixed both sites. Now parsed with `ast`. Added GATE 3 (the outcome signal is available
where the hint is chosen: call ordering + `reward_score` consumption + explicit
suppress-on-missing) and GATE 4, which is **live execution** rather than grep: it
imports the working tree, enables hints, and asserts correct/incorrect produce
different token spans, a missing score suppresses, multimodal skips, and gold beats
hint.

### `hcgym_rebuttal/runs/train_ttopd_hints.slurm` (new)

See §3.

### Tests

`tests/experimental/teacher_loop/test_hint_opd_injection_on_cpu.py`, 25 tests, CPU only.

---

## 3. What the new arm isolates

`runs/train_ttopd_hints.slurm` is **config-identical** to `train_hcgym.slurm ARM=ttopd`
— same backbone, batch (7), rollout budget (n=3, 5 turns, 3 epochs), GPU topology
(7 student + 1 teacher), cosine reward values, EMA settings (`ema_decay=0.999`,
`ema_update_interval=5`, `teacher_update_interval=30`), turn truncation
(`BT_OPD_MAX_TURN=0`), loss (`bt_opd_kl`, coef 2.0). The delta is *entirely in the verl
code*: the same `HINT_OPD_ENABLED=1` flag that was inert now fires.

Two properties are checked rather than asserted:

* **Parity.** The script dry-runs `train_hcgym.slurm ARM=ttopd` and diffs the resolved
  config against its own. Any difference is fatal (`PARITY=0` overrides). Verified in
  both directions: identical → `[parity] ... is identical`; a single changed cosine or
  loss-coef value → fatal with a diff.
* **Preflight.** The job refuses to start training unless `verify_hint_injection.py`
  returns 0 against the exact verl tree it will import. An arm whose entire purpose is
  one mechanism must not run with that mechanism inert a second time.
  (`SKIP_HINT_CHECK=1` overrides. Don't.)

It also carries the two B200 settings for both student and teacher engines
(`free_cache_engine=False`, `mm_attention_backend=triton_attn`); every backbone in this
study is a VLM (`Qwen3_5ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`),
so the teacher serves a VLM too.

`ttopd_hints − ttopd` therefore isolates exactly one thing: the privileged outcome
signal reaching the teacher.

---

## 4. What the rebuttal MAY and MAY NOT claim

### MUST retract

* **Any claim that outcome-conditioned privileged hints contributed to the reported
  TT-OPD results.** They did not run. This is not "a weak effect" or "an
  implementation detail" — the code path was unreachable. A mechanism named in the
  abstract and in the method section produced zero gradient signal in every reported
  number.
* **Any ablation, discussion, or intuition in the paper that attributes behaviour to
  the hints.** If the paper says hints steer the teacher toward reinforcing correct
  reasoning and re-examining incorrect reasoning, that sentence describes code that
  never executed. If a hint-related ablation is reported, it is comparing two runs
  that were byte-identical in the hint dimension, and its delta is noise.

### MAY claim, with the correction stated

* The reported TT-OPD numbers are **real numbers for a real method** — they are just
  numbers for a *different, smaller* method than the paper describes:
  **GRPO + cosine length reward + EMA self-distillation teacher + turn-level
  truncation.** Nothing about the training runs was invalid; only the description was.
  Re-describe the method to match what ran, and the empirical results stand as-is.
* The three-way ladder in `train_hcgym.slurm` (`grpo` → `grpo_cosine` → `ttopd`) remains
  a valid ablation of what *did* run, and prices the cosine reward on its own.

### MAY NOT claim without new runs

* That hints help, hurt, or are neutral. There is **no evidence in either direction**.
  The `ttopd_hints` arm produces that evidence; until it has, the honest statement is
  "untested".
* That the fix is a no-op because the mechanism is minor. That is a prediction, not a
  result.

### Suggested framing

> We discovered that the privileged-hint mechanism described in §X was inert in all
> reported runs due to an implementation defect (the hint tokens were never forwarded
> to the teacher on the code path our configuration selected). The reported results are
> therefore results for GRPO + cosine length reward + EMA self-distillation teacher +
> turn-level truncation, and we correct the method description accordingly. We have
> fixed the defect and report the hint-enabled variant separately as `ttopd_hints`; its
> effect was not measured in the submitted version and we make no claim about it there.

Do not bury this in an appendix. Anyone who runs the code will find it in an hour;
the checker script is in the repo precisely so the finding is reproducible by them too.

---

## 5. Findings that do not change the verdict but should be on the record

* **The multimodal restriction cost nothing here.** Both teacher paths skip injection
  for image/video samples. Measured against the configured training data
  (`data/verl_parquet/full_4modality_clean/train.parquet`, 3390 rows): **0 rows are
  multimodal (0.0%)**, and the same holds for `full_4modality` (4543/916) and the test
  splits. Despite the directory name, these splits carry no `images` column and no
  image parts in any prompt. So gate 3 suppressed exactly 0 samples and is reported as
  scope, not as a cause. If image-bearing data is added later, hints will silently skip
  those samples — the log histogram will show `skip:multimodal`.
* **`HINT_OPD_DYNAMIC` is a dead knob.** `train_hcgym.slurm` and every v15–v32 run
  script export it; no Python in either repo reads it. It has never done anything.
* The correctness threshold `score > 0` was already the batch path's behaviour; it is
  preserved exactly, so this fix does not smuggle in a semantic change alongside the
  wiring change.

---

## 6. Two things this fix breaks that someone else must fix

Neither file is mine to edit (`train_hcgym.slurm` and `launch_backbones.sh` are owned by
another agent right now).

1. **`train_hcgym.slurm ARM=ttopd` is no longer the paper's TT-OPD.** It exports
   `HINT_OPD_ENABLED=1`, which was inert under the old code and fires under the fixed
   code. Unless that arm sets **`HINT_OPD_ENABLED=0`**, "ttopd" and "ttopd_hints" become
   the same experiment and the isolation is lost. Job **60581 (`hcgym-q4b_ttopd`) is
   queued right now** and will pick up the fixed verl.

2. **`train_hcgym.slurm`'s `python3 -m verl.trainer.main_ppo` invocation is currently
   broken.** A comment block was inserted between backslash-continued lines (after
   `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \`). Bash joins the
   continuation, then `#` starts a comment that runs to end of line — and that line has
   no trailing backslash, so **the command terminates there** and every argument from
   `actor_rollout_ref.rollout.name=sglang` onward is parsed as a new command
   (`C: command not found`). Demonstrated:

   ```
   $ printf 'echo A \\\n  B \\\n  # comment\n  C \\\n  D\n' > t.sh && bash t.sh
   A B
   t.sh: line 5: C: command not found
   ```

   Comments must go *above* the invocation, not inside it. `train_ttopd_hints.slurm`
   does that, and says why in a comment at the site.

---

## 7. Reproduce everything

```bash
VENV=/data/project/private/minstar/workspace/hcgym_rebuttal/.venv/bin/python
cd /data/project/private/minstar/workspace/minstar/Healthcare_GYM

# historical defect — exits 1
$VENV scripts/rebuttal/verify_hint_injection.py --rev e14d6a58

# fixed working tree — exits 0
$VENV scripts/rebuttal/verify_hint_injection.py

# unit tests — 25 passed
cd /data/project/private/minstar/workspace/verl_ttopd
$VENV -m pytest tests/experimental/teacher_loop/test_hint_opd_injection_on_cpu.py -v

# the new arm, without submitting anything
cd /data/project/private/minstar/workspace/hcgym_rebuttal
DRY_RUN=1 BACKBONE=$PWD/models/Qwen3.5-9B EXP=q9b_ttopd_hints bash runs/train_ttopd_hints.slurm
```
