#!/usr/bin/env python
"""Measure ReAct format adherence — old configuration vs new — on real tasks.

Rebuttal. The react arm's accuracy is uninterpretable without
this number: on Qwen3.5-9B / MedQA the shipped arm scored 0.744 while emitting
ReAct on 10 of 4,429 turns (react_rate 0.002), so what it measured was a clash
between two tool-calling contracts in one prompt, not ReAct.

This driver runs the SAME tasks through the SAME multi-turn loop
(scripts/eval_benchmark_multiturn._run_single_task_multiturn) and the SAME
instrumentation (aggregate_format_adherence) under two configurations:

  old   PRE_COMMIT's build_system_prompt + the tool catalog delivered through
        apply_chat_template(tools=...), which also injects the model's native
        tool-calling contract. Reproduced by loading that commit's module from
        git, not by re-implementing it.
  new   the current build_system_prompt, which renders the identical tool
        catalog into the prompt text, with the template injection withheld by
        native_tools_for_prompt_mode — so ReAct is the only contract present.

Everything else is held fixed: same task list in the same order, same turn
budget, same decoding, same seed, same parser. The two arms therefore differ in
exactly one thing, which is what makes the react_rate delta attributable.

Backend note: this runs the backbone on CPU, because the paired comparison has
to happen without touching the GPU queue. CPU decoding is ~0.3 tok/s for this
backbone, so the sample is small by necessity and `n` is reported with every
number. Prompt rendering matches the sglang eval path exactly
(apply_chat_template(..., enable_thinking=False)); only the executor differs.

Run:
    PYTHONPATH=<repo> .venv/bin/python scripts/rebuttal/measure_react_adherence.py \
        --model <path> --n 6 --out <json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from loguru import logger  # noqa: E402

import bioagents.evaluation.agent_runner as AR  # noqa: E402
from bioagents.evaluation.agent_runner import (  # noqa: E402
    AgentRunner,
    RunConfig,
    aggregate_format_adherence,
)

# The tree this change was applied to. A pinned hash, never symbolic "HEAD":
# after the change is committed, HEAD IS the change, and the "old" arm would
# silently become a second copy of the new one.
PRE_COMMIT = "10fdc57"


def head_module(ref: str = PRE_COMMIT):
    """Load a git ref's agent_runner.py as a live module (the old arm's code)."""
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         f"{ref}:bioagents/evaluation/agent_runner.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "agent_runner_head.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("_agent_runner_head_meas", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_agent_runner_head_meas"] = mod
    spec.loader.exec_module(mod)
    return mod


class CPURunner(AgentRunner):
    """AgentRunner that decodes on CPU with the sglang path's exact rendering.

    ``_generate_sglang`` renders the prompt locally with
    ``apply_chat_template(..., enable_thinking=False)`` and then posts it to the
    server. This does the first half identically and executes the second half
    in-process, so the string the model conditions on is the one the GPU eval
    would have sent.
    """

    def __init__(self, config: RunConfig, model, tokenizer, turn_log: Path):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
        self._is_vl_model = False
        self.processor = None
        self.n_calls = 0
        self.gen_seconds = 0.0
        # Every generation is appended verbatim. The adherence histogram says
        # WHICH branch parsed each turn; this says what the model actually
        # wrote, which is the only thing that settles a format claim.
        self.turn_log = turn_log

    def load_model(self):
        pass

    def generate(self, messages, tools=None):
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tools=tools if tools else None,
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = self.tokenizer(text, return_tensors="pt")
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=max(self.config.temperature, 0.01),
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        self.n_calls += 1
        self.gen_seconds += dt
        n_new = out.shape[1] - ids["input_ids"].shape[1]
        txt = self.tokenizer.decode(
            out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        _call, label = AR.parse_tool_call_with_format(txt)
        with self.turn_log.open("a") as fh:
            fh.write(json.dumps({
                "call": self.n_calls, "in_tokens": int(ids["input_ids"].shape[1]),
                "out_tokens": int(n_new), "seconds": round(dt, 1),
                "format": label, "tool": (_call or {}).get("name"),
                "output": txt,
            }, ensure_ascii=False) + "\n")
        logger.info(
            f"    gen #{self.n_calls}: {ids['input_ids'].shape[1]} in -> "
            f"{n_new} out in {dt:.0f}s  format={label}"
        )
        return txt


def run_arm(arm: str, tasks, model, tokenizer, args) -> dict:
    """Run one configuration over ``tasks`` and return the eval's own summary.

    The scoring path is not reimplemented here: this calls the benchmark
    driver, ``run_benchmark_multiturn``, so accuracy, the adherence aggregate
    and the written results JSON are produced by exactly the code the GPU eval
    runs.

    The old arm is installed by pointing the module attributes that loop
    resolves AT CALL TIME back at PRE_COMMIT's implementations — its real
    build_system_prompt, and the identity tool gate it had before
    native_tools_for_prompt_mode existed. Nothing else is touched.
    """
    import eval_benchmark_multiturn as EB

    saved = (AR.build_system_prompt, AR.native_tools_for_prompt_mode)
    if arm == "old":
        H = head_module()
        AR.build_system_prompt = H.build_system_prompt
        AR.native_tools_for_prompt_mode = lambda tools, mode: tools

    cfg = RunConfig(
        model_name_or_path=args.model,
        backend="transformers",
        domain="medical_qa",
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        no_think=True,          # matches runs/eval_agentic.slurm (--no-think)
        prompt_mode="react",
        log_dir=str(Path(args.out).parent / "logs"),
    )
    out_dir = Path(args.out).parent / f"arm_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = CPURunner(cfg, model, tokenizer, out_dir / "raw_turns.jsonl")

    t0 = time.time()
    try:
        summary = EB.run_benchmark_multiturn(
            benchmark_name="medqa", tasks=list(tasks), runner=runner,
            domain="medical_qa", max_turns=args.max_turns, output_dir=out_dir,
        )
    finally:
        AR.build_system_prompt, AR.native_tools_for_prompt_mode = saved

    fa = summary["format_adherence"]
    fa["gen_calls"] = runner.n_calls
    fa["gen_seconds"] = round(runner.gen_seconds, 1)
    logger.info(
        f"=== arm {arm}: react_rate={fa['react_rate']:.3f} "
        f"({fa['n_react']}/{fa['n_turns']} turns) acc={summary['accuracy']:.3f} "
        f"({summary['correct']}/{summary['total']}) formats={fa['formats']} "
        f"wall={time.time() - t0:.0f}s"
    )
    return {
        "adherence": fa,
        "accuracy": summary["accuracy"],
        "correct": summary["correct"],
        "total": summary["total"],
        "avg_turns": summary["avg_turns"],
        "per_task": [
            {"id": r.get("task_id") or r.get("id"),
             "correct": r.get("correct"),
             "react_rate": r.get("react_rate"),
             "turns": r.get("turns")}
            for r in summary["results"]
        ],
    }


def describe_arms(task, tokenizer) -> dict:
    """Log what each arm actually puts in front of the model, before running.

    Cheap, model-free, and it is the evidence that the two arms differ in one
    thing: the old arm's prompt carries the native tool-call contract, the new
    one does not, and both carry the same tool catalog.
    """
    from bioagents.gym.agent_env import BioAgentGymEnv

    env = BioAgentGymEnv(domain="medical_qa", max_turns=5)
    env._task_map[task["id"]] = task
    env._tasks.append(task)
    _obs, info = env.reset(options={"task_id": task["id"]})
    tools = [t for t in info["tools"]
             if t.get("function", {}).get("name") != "think"]
    H = head_module()
    NATIVE = "<function=example_function_name>"

    out = {}
    for arm, build, gate in (
        ("new", AR.build_system_prompt, AR.native_tools_for_prompt_mode),
        ("old", H.build_system_prompt, lambda t, m: t),
    ):
        sysp = build(info["policy"], tools, domain="medical_qa", task=task,
                     prompt_mode="react")
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": sysp},
             {"role": "user", "content": "TICKET"}],
            tools=gate(tools, "react") or None,
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        n_named = sum(
            1 for t in tools
            if f'"name": "{t.get("function", t).get("name")}"' in rendered
        )
        out[arm] = {
            "prompt_tokens": len(tokenizer(rendered)["input_ids"]),
            "native_tool_contract": NATIVE in rendered,
            "react_contract": "Action Input:" in rendered,
            "tools_in_prompt": f"{n_named}/{len(tools)}",
        }
        logger.info(f"  arm {arm}: {out[arm]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=6, help="tasks per arm")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--arms", default="new,old")
    ap.add_argument("--threads", type=int, default=56)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    from eval_benchmark_multiturn import load_textqa_benchmark

    tasks = load_textqa_benchmark("medqa")[: args.n]
    logger.info(f"{len(tasks)} MedQA tasks; model on CPU with {args.threads} threads")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()

    out = {"model": args.model, "n_tasks": len(tasks),
           "task_ids": [t["id"] for t in tasks],
           "max_turns": args.max_turns, "max_new_tokens": args.max_new_tokens,
           "temperature": args.temperature, "seed": args.seed, "arms": {}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("what each arm puts in front of the model (task 0):")
    out["arm_prompts"] = describe_arms(tasks[0], tokenizer)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    for arm in args.arms.split(","):
        arm = arm.strip()
        if not arm:
            continue
        logger.info(f"=== arm: {arm} ===")
        torch.manual_seed(args.seed)
        out["arms"][arm] = run_arm(arm, tasks, model, tokenizer, args)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n" + "=" * 84)
    print(f"{'arm':>6} {'tasks':>6} {'turns':>6} {'n_react':>8} {'react_rate':>11} "
          f"{'acc':>8}  formats")
    for arm, d in out["arms"].items():
        a = d["adherence"]
        print(f"{arm:>6} {a['n_tasks']:>6} {a['n_turns']:>6} {a['n_react']:>8} "
              f"{a['react_rate']:>11.3f} {d['accuracy']:>8.3f}  {a['formats']}")
    print("=" * 84)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
