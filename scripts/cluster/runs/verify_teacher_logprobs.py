#!/usr/bin/env python3
"""Do sglang's input logprobs, in vLLM's layout, actually hold the right numbers?

The translation added to SGLangHttpServer.generate rebuilds vLLM's
``prompt_logprobs`` from sglang's ``meta_info``. Unit tests pin the SHAPE, which
is not the risk: a one-position shift, or a rank order read backwards, produces a
perfectly shaped tensor of wrong numbers, and it feeds the distillation KL
directly. Nothing downstream would notice.

So compare against an independent reference -- a plain HuggingFace forward pass
over the same token ids -- and require agreement.

    python runs/verify_teacher_logprobs.py --model <path> [--topk 8]

Exit status is 0 only if every position agrees within tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

VERL_ROOT = Path("/data/project/private/minstar/workspace/verl_ttopd")
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.workers.rollout.sglang_rollout.async_sglang_server import (  # noqa: E402
    prompt_logprobs_from_meta,
)

PROMPTS = [
    "The patient presents with acute chest pain radiating to the left arm.",
    "Question: Which antibiotic covers MRSA?\nOption A: vancomycin\nOption B: amoxicillin",
    "think",
]


def reference_logprobs(model, input_ids: list[int], topk: int):
    """Logprobs of each token given its prefix, straight from the model."""
    ids = torch.tensor([input_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    logprobs = torch.log_softmax(logits[0], dim=-1)
    # Position i of the output predicts token i+1.
    taken, top_vals, top_ids = [], [], []
    for i in range(len(input_ids) - 1):
        row = logprobs[i]
        taken.append(row[input_ids[i + 1]].item())
        if topk > 0:
            v, k = torch.topk(row, topk)
            top_vals.append(v.tolist())
            top_ids.append(k.tolist())
    return taken, top_vals, top_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--topk", type=int, default=0, help="0 = the taken token only")
    ap.add_argument("--noise-mult", type=float, default=3.0,
                    help="allowed multiple of the measured bf16-vs-fp32 floor")
    ap.add_argument("--atol", type=float, default=5e-2,
                    help="sglang runs in bf16 with a different kernel; exact equality is not the bar")
    args = ap.parse_args()

    import sglang
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = sglang.Engine(model_path=args.model, tp_size=1, attention_backend="triton",
                           mem_fraction_static=0.45, log_level="error", skip_server_warmup=True)

    print("loading the HF reference model (same weights, independent code path)")
    ref = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

    # How much do two runs of the SAME model disagree purely from precision? Without
    # this the sglang-vs-HF gap has no scale: 0.13 nats is either a broken
    # translation or ordinary bf16 kernel noise, and the number alone cannot say
    # which. fp32 on the identical weights and token ids gives the floor.
    print("loading an fp32 copy to measure the bf16 noise floor")
    ref32 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="cuda", trust_remote_code=True
    ).eval()

    failures = 0
    for text in PROMPTS:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 3:
            continue
        out = engine.generate(
            input_ids=ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=args.topk if args.topk > 0 else 0,
        )
        meta = out["meta_info"] if isinstance(out, dict) else out[0]["meta_info"]
        got_lp, got_ids = prompt_logprobs_from_meta(meta, args.topk, len(ids))

        taken, top_vals, top_ids = reference_logprobs(ref, ids, args.topk)
        taken32, top_vals32, _ = reference_logprobs(ref32, ids, args.topk)
        noise = max((abs(a - b) for a, b in zip(taken, taken32, strict=True)), default=0.0)

        print(f"\nprompt: {text[:60]!r}  ({len(ids)} tokens)")
        print(f"  rows returned: {len(got_lp)} (expected {len(ids)})")
        if len(got_lp) != len(ids):
            print("  FAIL: row count"); failures += 1; continue

        if args.topk == 0:
            # Column 0 must be the logprob of the NEXT token, and the id column
            # must name that token -- this is what catches an off-by-one.
            bad_id = [i for i in range(len(ids) - 1) if got_ids[i][0] != ids[i + 1]]
            diffs = [abs(got_lp[i][0] - taken[i]) for i in range(len(ids) - 1)]
            worst = max(diffs) if diffs else 0.0
            print(f"  token-id alignment mismatches: {len(bad_id)}")
            print(f"  max |sglang  - HF bf16|:       {worst:.4f}")
            print(f"  max |HF bf16 - HF fp32|:       {noise:.4f}   <- precision floor")
            # The bar is the floor, not an absolute constant: a translation that
            # agrees to within what two runs of the same model disagree by is right.
            bar = max(args.atol, noise * args.noise_mult)
            if bad_id:
                print(f"  FAIL: rows {bad_id[:5]} name the wrong token — off-by-one")
                failures += 1
            elif worst > bar:
                print(f"  FAIL: {worst:.4f} exceeds {bar:.4f} ({args.noise_mult}x the floor)")
                failures += 1
            else:
                print(f"  ok (within {bar:.4f})")
        else:
            worst = 0.0
            order_bad = 0
            for i in range(len(ids) - 1):
                if got_ids[i][0] != top_ids[i][0]:
                    order_bad += 1
                worst = max(worst, abs(got_lp[i][0] - top_vals[i][0]))
                if got_lp[i] != sorted(got_lp[i], reverse=True):
                    order_bad += 1
            print(f"  rank-order problems: {order_bad}")
            print(f"  max |sglang - HF| on rank-1: {worst:.4f}")
            print(f"  precision floor:             {noise:.4f}")
            if order_bad or worst > max(args.atol, noise * args.noise_mult):
                print("  FAIL"); failures += 1
            else:
                print("  ok")

    print("\n" + "=" * 60)
    if failures:
        print(f"{failures} prompt(s) FAILED — do not trust teacher logprobs")
        return 1
    print("sglang input logprobs agree with the HF reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
