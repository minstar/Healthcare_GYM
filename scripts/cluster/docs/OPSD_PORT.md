# OPSD Baseline Port — What Was Ported, What Was Changed, and Every Deviation

**Reference method**: OPSD — On-Policy Self-Distillation (Zhao et al., ICML 2026,
arXiv:2601.18734). Reference code: `baselines/OPSD/` (clone of `siyan-zhao/OPSD`),
chiefly `opsd_trainer.py` and `data_collator.py`.

**Port target**: the TT-OPD verl fork at
`/data/project/private/minstar/workspace/verl_ttopd`
(branch `feat/model-family-portability`). New loss mode `opsd` in
`verl/trainer/distillation/losses.py` (`compute_opsd_loss`); gold-answer teacher
conditioning in `verl/experimental/teacher_loop/teacher_manager.py`
(`OPSD_GOLD_CONDITIONING=1`); runner `runs/train_opsd.slurm`.

**Purpose**: baseline arm isolating whether TT-OPD's gains come from TT-OPD's
specific machinery or from on-policy self-distillation in general. The arm keeps
the TT-OPD arm's backbone, data, GPU topology (7 student + 1 teacher), rollout
budget (`rollout.n=3`), batch size (7), epochs, and degenerate-response reward
guard, and differs only in the method under test.

> **The official OPSD training script was NOT run.** This is a re-implementation
> of the OPSD objective inside the same training harness as TT-OPD, which is what
> makes the comparison controlled — and which necessarily introduces the
> deviations listed below.

## What was ported (faithful to the published method)

1. **Single model as student and teacher.** Teacher and student are the same
   backbone; the teacher replica stays frozen at the initial weights for the whole
   run — the exact semantics of OPSD's `fixed_teacher` option, which is what all
   of their released run scripts use (`run_opsd_*.sh`). No EMA teacher
   (`use_ema_teacher` off, matching their released runs), no teacher updates.
2. **Privileged conditioning on the ground-truth solution.** The teacher's
   context is the student's prompt plus the sample's ground-truth answer, rendered
   with wording taken from OPSD's `data_collator.py` teacher prompt ("Here is a
   reference solution… do not copy or paraphrase it. Now, using your own words and
   independent reasoning, derive the same final answer…"). The student sees only
   the problem.
3. **Token-level distribution matching on the student's own on-policy
   trajectory** (their `lmbda=1` fully-on-policy setting): every training sequence
   is sampled from the current student; the teacher only scores it.
4. **Per-token point-wise KL clipping** (their `jsd_token_clip`, added Mar 2026
   after finding that style tokens such as 'wait'/'think' show 6–15× higher KL
   than content tokens and dominate the gradient): each token's divergence is
   clamped to a maximum before aggregation, implemented as the same
   `clamp(max=clip)` on per-token values. Off by default in the loss
   (`OPSD_TOKEN_CLIP` unset); the runner sets 0.05, their published
   thinking-model value (1B/4B: 0.05, 8B: 0.06).
5. **Direct supervised backpropagation of the divergence** (their
   `generalized_jsd_loss` path): `use_policy_gradient=False`. The objective is
   pure distillation with no task-reward term, exactly as OPSD trains
   (`use_task_rewards=False`; task rewards are still computed for logging).
6. **None of TT-OPD's additions**: no EMA teacher, no outcome-conditioned
   correctness hints, no turn-level truncation, no bidirectional reward-sign
   flip, no top-k position filtering, no cosine length-controlled reward, no
   adaptive distillation coefficient.

## Deviations from the published OPSD method (quotable list)

The rebuttal must present these verbatim or in equivalent detail. None of them
is cosmetic; (1) is forced by the serving architecture and the rest follow from
it or from the harness.

1. **Divergence estimator: sampled-token reverse-KL (k3) instead of
   full-vocabulary forward KL.** Published OPSD computes a full-vocabulary
   generalized JSD from two forward passes of the same network; all released runs
   use `beta=0`, i.e. forward KL(teacher ‖ student) summed over the entire
   vocabulary at each position, at softmax temperature 1.1. In our harness the
   teacher is a separate frozen sglang server that returns only the log-probability
   of the token the student actually sampled (shape `[batch, seq_len]`); no
   full-vocabulary teacher logits exist on this path. We therefore use the closest
   quantity computable from per-token sampled logprobs: the non-negative
   low-variance reverse-KL estimator k3 = exp(t−s) − (t−s) − 1 (Schulman 2020)
   along the student's on-policy trajectory, backpropagated directly. This changes
   (a) the direction of the divergence (reverse instead of forward KL — mode-seeking
   rather than mass-covering), (b) full-vocabulary expectation → single-sample
   estimate, and (c) removes their temperature scaling (our teacher scores at
   temperature 1.0, argmax-safe greedy scoring mode).
2. **`token_clip` value is not transferable exactly.** Their 0.05/0.06 clip was
   tuned on full-vocab forward-KL magnitudes; our per-token k3 values live on a
   different scale. We adopt 0.05 as the default (overridable via
   `OPSD_TOKEN_CLIP`); the clip *mechanism* is identical (per-token
   `clamp(max=·)` before masked aggregation), the *threshold* is not calibrated
   to theirs.
3. **`top_k_loss` is not implementable.** Their optional top-k restriction
   renormalizes both distributions over the teacher's top-k tokens; without
   vocabulary-level teacher outputs this cannot be computed. Not ported (their
   released runs do not use it either — `top_k_loss` defaults to 0/off).
4. **Privileged context is injected as a separate user turn, not fused into the
   problem statement.** OPSD's data collator builds one user message containing
   problem + solution + transition prompt. Our harness splices the rendered
   gold-answer turn (via the model's own `apply_chat_template`) between the
   student's prompt and the response inside the teacher's scoring request, then
   stitches the injected span back out of the returned logprobs so positions
   align with the student sequence — the same mechanism the fork's hint path
   uses. The teacher conditions on the same information, but the chat structure
   differs (an extra user turn instead of a longer single message).
5. **Ground-truth granularity differs from their "solution".** OPSD's datasets
   carry full reference reasoning chains ("solution" column). The hcgym data
   carries a graded answer key (`reward_model.ground_truth`, e.g. an option
   letter — fully informative in context because the options are in the prompt)
   and, for open-ended samples whose graded key is empty (1,556/3,390 rows), a
   reference answer text (`extra_info.correct_answer`, used as fallback).
   328/3,390 rows have neither and are scored by the teacher without privileged
   context (counted and logged per batch as `[OPSD-Gold] … missing ground
   truth`). Gold text is capped at `OPSD_GOLD_MAX_CHARS=2000` characters.
6. **`reason_first` mode is not ported.** Their optional mode where the teacher
   first generates an analysis of the solution before scoring is out of scope
   (also off in their released runs).
7. **Training stack and regime differ.** Theirs: TRL SFTTrainer + LoRA (r=64) +
   vLLM colocate, single-turn math prompts, lr 5e-6, ~1024-token completions,
   student = LoRA adapters over the frozen base that doubles as teacher. Ours:
   the rebuttal harness — full-parameter FSDP training, multi-turn tool-use
   rollouts (max 5 assistant turns), lr 1e-6, 12,288-token responses, separate
   frozen teacher replica on a dedicated GPU. These are held identical to the
   TT-OPD arm on purpose: the harness is the controlled variable, the method is
   the manipulated one.
8. **Rollout sampling params follow the harness, not their generation config**
   (their temperature 1.1 / top-p 0.95 / top-k 20 student rollouts vs. the
   harness's shared rollout settings used by every arm).

## Correctness notes (what makes this port trustworthy)

- The loss reuses the fork's NaN/inf sanitization verbatim: teacher logprobs are
  NaN at first-token positions and −inf at zero-probability tokens; NaN maps to
  −20.0 (≈ prob 2e-9), **not** 0.0, which would mean prob 1.0 and bias the KL.
  This handling is load-bearing.
- The injected gold tokens are excluded from the returned teacher logprobs by
  the same stitching logic as the hint path (`_stitch_out_injected_span`), so
  response-token positions stay aligned with the student's sequence while their
  logprobs remain conditioned on prompt + gold turn.
- Gold conditioning is wired into BOTH teacher paths. With
  `enable_resource_pool=True` (the topology every arm uses), teacher logprobs are
  computed by the agent loop's streaming path
  (`agent_loop.py::_compute_teacher_logprobs`), not by
  `compute_teacher_logprobs_batch` — injection only in the batch path would have
  silently produced an unconditioned teacher. The streaming path now extracts the
  per-sample ground truth from the sample's non-tensor fields and injects it; the
  batch (colocate) path does the same for completeness.
- Threading fix required for this arm (and latent in the hint path): neither
  `TeacherModelManager` nor the agent loop passed a tokenizer into
  `AsyncTeacherLLMServerManager`, so injected-turn construction silently no-oped.
  Both construction sites now pass the tokenizer.
- Known issue in the TT-OPD arm (pre-existing, deliberately left untouched):
  under `enable_resource_pool=True` the outcome-conditioned HINT_OPD hints were
  only implemented in the bypassed batch path, i.e. the TT-OPD runs' teacher
  was effectively unhinted on the streaming path. Not changed here to keep the
  already-trained TT-OPD arm's behavior stable; flagged for a separate decision.
- Unit tests (all passing, 20 total):
  `tests/trainer/distillation/test_opsd_loss_on_cpu.py` (registration, masking,
  NaN/inf handling, clip bounding + zero-gradient at clipped tokens, zero loss on
  zero divergence, no sign flip / no truncation, KL-distribution metrics) and
  `tests/experimental/teacher_loop/test_opsd_gold_conditioning_on_cpu.py`
  (ground-truth extraction incl. fallback, chat-template rendering with BOS
  stripping and brace-safe substitution, stitch exactness / short-output
  fallback).

## Run

```bash
sbatch --export=ALL,BACKBONE=<model_dir>,EXP=<name>_opsd \
    /data/project/private/minstar/workspace/hcgym_rebuttal/runs/train_opsd.slurm
```

Metrics to plot the KL distribution: `opsd/kl_mean`, `opsd/kl_p50`,
`opsd/kl_p90`, `opsd/kl_p99`, `opsd/kl_max`, `opsd/k1_signed_mean`,
`opsd/clip_frac`, plus the framework's `distillation/loss_min|max`.
