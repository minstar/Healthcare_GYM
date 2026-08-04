#!/usr/bin/env python3
"""Prove the teacher-logprob transport survives a ragged batch, and scores correctly.

Regression test for the failure that kept every distillation arm at zero
checkpoints on every backbone:

    AssertionError: only the last (ragged) dim may vary across samples.
    Got torch.Size([12771, 1]) vs torch.Size([12662, 1])
      _compute_old_log_prob -> chunk_tensordict -> as_nested_tensor_ragged_last

The teacher server returns (S, C) -- sequence first, candidates last -- but verl
nests per-sample tensors with the RAGGED dim last. With C == 1 the candidate axis
carries nothing, so `_to_ragged_last` drops it at creation and
`compute_forward_kl_topk` restores it before gathering.

Three things have to hold, and each is checked against the real functions rather
than a re-implementation:

  1. the OLD layout really does break a ragged batch (otherwise this test would
     pass even with the fix reverted, and prove nothing);
  2. the NEW layout survives the same batch;
  3. the KL the loss computes on the new layout equals the KL computed by hand
     from the same numbers -- a shape fix that silently changed the objective
     would be worse than the crash.

Run:  python tests/test_teacher_tensor_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

VERL = Path("/data/project/private/minstar/workspace/verl_ttopd")
if str(VERL) not in sys.path:
    sys.path.insert(0, str(VERL))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from verl.experimental.teacher_loop.teacher_manager import (  # noqa: E402
    _pad_teacher_outputs,
    _to_ragged_last,
)
from verl.utils.tensordict_utils import as_nested_tensor_ragged_last  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# Two samples of DIFFERENT valid length -- the condition that triggers the bug.
# Widths are what the agent loop pads to; lengths are what the teacher actually
# returned for each sample.
PROMPT_WIDTH, RESPONSE_WIDTH = 8, 12
# (prompt_length, response_length) per sample. The totals must DIFFER (12 vs 7):
# equal-length samples nest fine even under the broken layout, so a test built on
# them would pass with the fix reverted.
CASES = [(5, 7), (3, 4)]
VOCAB = 16
PAD_ID = 0


def make_raw(seq_len: int, candidates: int):
    """What the teacher server hands back: (S, C) ids + logprobs."""
    torch.manual_seed(seq_len * 100 + candidates)
    ids = torch.randint(0, VOCAB, (seq_len, candidates), dtype=torch.int32)
    logprobs = torch.log_softmax(torch.randn(seq_len, candidates), dim=-1)
    return ids, logprobs


def transport(pairs):
    """Pad each sample, stack, then nest exactly as the training path does."""
    padded = [
        _pad_teacher_outputs(
            ids, lp,
            prompt_width=PROMPT_WIDTH, response_width=RESPONSE_WIDTH,
            prompt_length=pl, response_length=rl, pad_token_id=PAD_ID,
        )
        for (ids, lp), (pl, rl) in zip(pairs, CASES, strict=True)
    ]
    return torch.cat([p[0] for p in padded], dim=0), torch.cat([p[1] for p in padded], dim=0)


def nest_per_sample(batched, lengths):
    """Unpad to the per-sample valid span and nest, as chunk_tensordict does."""
    per_sample = [batched[i][: lengths[i]] for i in range(len(lengths))]
    return as_nested_tensor_ragged_last(per_sample)


valid_lengths = [pl + rl for pl, rl in CASES]

# ── 1. the OLD layout must break, or this test proves nothing ────────────────
raw_old = [make_raw(n, 1) for n in valid_lengths]  # kept as (S, 1)
ids_b, lp_b = transport(raw_old)
try:
    nest_per_sample(lp_b, valid_lengths)
    check("old (S,1) layout breaks a ragged batch", False, "it did NOT raise — the bug is not reproduced")
except AssertionError as e:
    check("old (S,1) layout breaks a ragged batch", "last (ragged) dim" in str(e), str(e)[:120])

# ── 2. the NEW layout must survive the identical batch ───────────────────────
raw_new = [_to_ragged_last(*make_raw(n, 1)) for n in valid_lengths]
check("_to_ragged_last drops the singleton candidate axis", all(lp.dim() == 1 for _, lp in raw_new),
      f"dims {[lp.dim() for _, lp in raw_new]}")

ids_b, lp_b = transport(raw_new)
check("batched teacher tensors are 2-D (bsz, seq)", lp_b.dim() == 2 and ids_b.dim() == 2,
      f"{tuple(lp_b.shape)} / {tuple(ids_b.shape)}")

try:
    lp_nested = nest_per_sample(lp_b, valid_lengths)
    ids_nested = nest_per_sample(ids_b, valid_lengths)
    check("new layout nests a ragged batch", True)
except AssertionError as e:
    lp_nested = ids_nested = None
    check("new layout nests a ragged batch", False, str(e)[:160])

# C > 1 must be refused where the shape is still explicable, not 11 minutes in.
try:
    _to_ragged_last(*make_raw(4, 3))
    check("C>1 is refused at creation", False, "it was accepted")
except NotImplementedError as e:
    check("C>1 is refused at creation", "ragged" in str(e), str(e)[:120])

# ── 3. the loss must compute the SAME KL it would have on the old layout ─────
if lp_nested is not None:
    total_nnz = sum(valid_lengths)
    torch.manual_seed(0)
    student_logits = torch.randn(1, total_nnz, VOCAB)

    from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk

    class _LossCfg:
        log_prob_min_clamp = None

    class _Cfg:
        distillation_loss = _LossCfg()

    try:
        out = compute_forward_kl_topk(
            student_logits=student_logits,
            teacher_topk_log_probs=lp_nested,
            teacher_topk_ids=ids_nested.long(),
            config=_Cfg(),
            data_format="thd",
        )
        got = out["distillation_losses"]
        check("loss returns (bsz, seqlen)", got.shape == (1, total_nnz), f"{tuple(got.shape)}")

        # Hand-computed reference over the same numbers: forward KL of a
        # one-candidate teacher against the student's log-prob at that token.
        t_lp = lp_nested.values().reshape(1, total_nnz, 1)
        t_id = ids_nested.long().values().reshape(1, total_nnz, 1)
        s_lp = torch.gather(F.log_softmax(student_logits, dim=-1), -1, t_id)
        ref = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)
        check("loss matches a hand-computed forward KL", torch.allclose(got, ref, atol=1e-6),
              f"max|Δ| = {(got - ref).abs().max().item():.3e}")
    except Exception as e:  # noqa: BLE001
        check("loss runs on the new layout", False, f"{type(e).__name__}: {str(e)[:200]}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("teacher tensor layout: all checks passed")
