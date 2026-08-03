#!/usr/bin/env python3
"""Score a benchmark under the Reflexion self-critique ladder.

Reviewers asked for baselines that separate the paper's method from mere
prompting. `--prompt-mode react` and `--prompt-mode strong_tool` in
`eval_benchmark_multiturn.py` cover single-attempt prompting; this covers the
upper bound, where the agent is allowed to look at its own failed attempt and
try again. If that upper bound reaches the RL arms, the paper's contribution
claim has to be restated.

Reflexion: Shinn et al., NeurIPS 2023. The official repo was NOT run — it is
hardcoded to a 2023-era OpenAI completions API. `reflexion_runner.py`
reimplements the loop against this environment; its
DEVIATIONS_FROM_PUBLISHED_REFLEXION lists every difference and this script
prints them into the output so a result can never be quoted without them.

The four strategies form the ladder the paper cites:

    NONE                        one attempt; the reference point
    LAST_ATTEMPT                retry with the failed transcript, no reflection
    REFLEXION                   retry with a verbal self-reflection
    LAST_ATTEMPT_AND_REFLEXION  retry with both

LAST_ATTEMPT is what separates "reflection helped" from "a second sample
helped" — without it a Reflexion gain is unattributable, so it is in the default
sweep even though it costs a full extra pass.

    python scripts/rebuttal/run_reflexion_eval.py \\
        --model_path <ckpt> --server-url http://127.0.0.1:31000 \\
        --benchmarks medqa --strategies all --output-dir eval_results/reflexion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig  # noqa: E402
from bioagents.evaluation.reflexion_runner import (  # noqa: E402
    DEVIATIONS_FROM_PUBLISHED_REFLEXION,
    ReflexionConfig,
    ReflexionRunner,
    ReflexionStrategy,
)


def load_benchmark(name: str, limit: int) -> list[dict]:
    """Reuse the eval harness's own loader so tasks are identical across arms."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_mt", str(REPO / "scripts" / "eval_benchmark_multiturn.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if name not in mod.BENCHMARK_FILES:
        raise SystemExit(f"[fatal] unknown benchmark '{name}'. known: {sorted(mod.BENCHMARK_FILES)}")

    # Mirror the dispatch in eval_benchmark_multiturn.main (~line 944) rather than
    # reparsing anything here: the Reflexion arm has to see byte-identical tasks to
    # the other conditions or its numbers are not comparable to them.
    VQA = {"vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"}
    EHR = {"mimic_iii", "eicu"}
    if name in VQA:
        tasks = mod.load_vqa_benchmark(name)
    elif name in EHR:
        tasks = mod.load_ehr_benchmark(name)
    else:
        tasks = mod.load_textqa_benchmark(name)
    return tasks[:limit] if limit else tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--backend", default="sglang", choices=["transformers", "vllm", "sglang"])
    ap.add_argument("--server-url", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=["medqa"])
    ap.add_argument("--strategies", nargs="+", default=["all"],
                    help="subset of NONE/LAST_ATTEMPT/REFLEXION/LAST_ATTEMPT_AND_REFLEXION, or 'all'")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=5,
                    help="must match the other arms; the submitted paper could not difference "
                         "its own columns partly because this varied between conditions")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--output-dir", default="eval_results/reflexion")
    args = ap.parse_args()

    names = [s.name for s in ReflexionStrategy]
    chosen = names if args.strategies == ["all"] else [s.upper() for s in args.strategies]
    bad = [s for s in chosen if s not in names]
    if bad:
        raise SystemExit(f"[fatal] unknown strategy {bad}; known: {names}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runner = AgentRunner(RunConfig(
        model_name_or_path=args.model_path,
        backend=args.backend,
        server_url=args.server_url,
        max_turns=args.max_turns,
        temperature=args.temperature,
    ))
    runner.load_model()

    import importlib.util as _il
    _spec = _il.spec_from_file_location("eval_mt", str(REPO / "scripts" / "eval_benchmark_multiturn.py"))
    _evalmod = _il.module_from_spec(_spec)
    _spec.loader.exec_module(_evalmod)
    from bioagents.gym.agent_env import BioAgentGymEnv

    summary: dict[str, dict] = {}
    for bench in args.benchmarks:
        tasks = load_benchmark(bench, args.max_samples)
        domain = _evalmod.BENCHMARK_DOMAIN[bench]
        runner.config.domain = domain
        print(f"\n=== {bench}: {len(tasks)} tasks (domain={domain}) ===")
        for strat in chosen:
            # STOP_AFTER_FIRST semantics: NONE is one attempt regardless of the cap,
            # so passing max_attempts through unchanged keeps the ladder honest.
            cfg = ReflexionConfig(
                strategy=ReflexionStrategy[strat],
                max_attempts=args.max_attempts,
                output_dir=str(out_root / bench / strat.lower()),
            )
            rx = ReflexionRunner(runner, cfg)
            t0 = time.time()
            results = []
            for task in tasks:
                # Fresh env per task, with the benchmark task injected into its map —
                # exactly what eval_benchmark_multiturn does (~line 520). Benchmark
                # tasks are not in the domain's own pool, so env.reset(task_id=...)
                # cannot find them otherwise, and reusing one env across tasks would
                # also diverge from how every other condition is scored.
                env = BioAgentGymEnv(domain=domain, max_turns=args.max_turns)
                env._task_map[task["id"]] = task
                env._tasks.append(task)
                results.append(rx.run_task(task, env))
            stats = rx.summarize(results)
            stats["wall_clock_s"] = round(time.time() - t0, 1)
            summary.setdefault(bench, {})[strat] = stats
            print(f"  {strat:28} " + " ".join(
                f"{k}={v}" for k, v in stats.items()
                if k in ("n_tasks", "n_solved", "solve_rate", "call_overhead_ratio", "extra_model_calls")
            ))

    payload = {
        "model_path": args.model_path,
        "backend": args.backend,
        "max_turns": args.max_turns,
        "max_attempts": args.max_attempts,
        "temperature": args.temperature,
        "summary": summary,
        # Carried in the artifact so no number can be quoted without them.
        "reimplementation_notice": (
            "Reflexion (Shinn et al., NeurIPS 2023) reimplemented against this environment. "
            "The official repository was NOT run."
        ),
        "deviations_from_published_reflexion": DEVIATIONS_FROM_PUBLISHED_REFLEXION,
    }
    dest = out_root / f"reflexion_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {dest}")

    print("\nCost is part of the result — a prompting baseline that wins on 5x the")
    print("inference budget is a finding, not a footnote. call_overhead_ratio above")
    print("is relative to a single-attempt pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
