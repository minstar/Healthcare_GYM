"""EHR Benchmark Evaluation for BIOAgents Healthcare AI GYM.

Evaluates trained models on real-world EHR data from two clinical databases:

  1. MIMIC-III Clinical Database v1.4 (50 ICU patients, 50 tasks)
     - Chart review, critical value ID, medication reconciliation,
       lab trend analysis, discharge readiness
     Source: PhysioNet / MIT Lab for Computational Physiology

  2. eICU Collaborative Research Database v2.0 (50 ICU patients, 50 tasks)
     - ICU assessment, vital monitoring, mortality prediction,
       lab trend analysis, medication review
     Source: PhysioNet / Philips Healthcare

These benchmarks test the agent's ability to navigate real EHR data
using the BIOAgents tool-use framework, providing a grounded evaluation
that complements synthetic task evaluation.

Usage:
    evaluator = EHRBenchmarkEvaluator(config)
    results = evaluator.evaluate_all()
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Benchmark data paths
BENCHMARK_DATA = {
    "mimic_iii": {
        "path": "data/ehr_benchmarks/mimic_iii_bench.json",
        "name": "MIMIC-III Clinical Database v1.4",
        "source": "PhysioNet",
        "description": "ICU patients from Beth Israel Deaconess Medical Center",
    },
    "eicu": {
        "path": "data/ehr_benchmarks/eicu_bench.json",
        "name": "eICU Collaborative Research Database v2.0",
        "source": "PhysioNet",
        "description": "Multi-center ICU patients from Philips Healthcare",
    },
}


@dataclass
class EHRBenchmarkConfig:
    """Configuration for EHR benchmark evaluation."""
    model_name_or_path: str
    model_name: str = "BIOAgent"
    backend: str = "transformers"
    benchmarks: list[str] = field(
        default_factory=lambda: ["mimic_iii", "eicu"]
    )
    max_samples: int = 0        # 0 = all
    max_turns: int = 15
    temperature: float = 0.1
    max_new_tokens: int = 1024
    output_dir: str = "results/ehr_benchmarks"
    task_split: str = "test"


@dataclass
class EHRTaskResult:
    """Result for a single EHR benchmark task."""
    task_id: str
    source: str
    category: str
    action_score: float = 0.0
    tool_calls_made: list = field(default_factory=list)
    expected_tools: list = field(default_factory=list)
    total_turns: int = 0
    final_answer: str = ""
    rubric_scores: dict = field(default_factory=dict)
    completed: bool = False
    error: Optional[str] = None
    latency: float = 0.0


class EHRBenchmarkEvaluator:
    """Evaluate models on real-world EHR benchmarks (MIMIC-III, eICU).

    This evaluator loads pre-built benchmark data (from build_ehr_benchmark.py),
    creates BIOAgents EHR environments with real patient data, and runs agent
    tasks to measure tool-use accuracy, clinical reasoning, and task completion.
    """

    def __init__(self, config: EHRBenchmarkConfig):
        self.config = config
        self._runner = None
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_runner(self):
        """Lazy-load the AgentRunner."""
        if self._runner is None:
            from bioagents.evaluation.agent_runner import AgentRunner, RunConfig

            run_config = RunConfig(
                model_name_or_path=self.config.model_name_or_path,
                backend=self.config.backend,
                domain="ehr_management",
                max_turns=self.config.max_turns,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
            )
            self._runner = AgentRunner(run_config)
            self._runner.load_model()
        return self._runner

    def load_benchmark(self, benchmark_key: str) -> tuple[dict, list]:
        """Load a benchmark dataset (db + tasks).

        Returns:
            (db_dict, tasks_list)
        """
        info = BENCHMARK_DATA.get(benchmark_key)
        if info is None:
            raise ValueError(
                f"Unknown benchmark: {benchmark_key}. "
                f"Available: {list(BENCHMARK_DATA.keys())}"
            )

        data_path = PROJECT_ROOT / info["path"]
        if not data_path.exists():
            raise FileNotFoundError(
                f"Benchmark data not found: {data_path}\n"
                f"Run: python scripts/build_ehr_benchmark.py"
            )

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        db = data["db"]
        tasks = data["tasks"]

        # Filter by split
        if self.config.task_split:
            tasks = [t for t in tasks if t.get("split", "test") == self.config.task_split]

        # Limit samples
        if self.config.max_samples > 0:
            tasks = tasks[:self.config.max_samples]

        logger.info(
            f"Loaded {benchmark_key}: {len(db['records'])} records, "
            f"{len(tasks)} tasks ({info['name']})"
        )
        return db, tasks

    def evaluate_benchmark(self, benchmark_key: str) -> dict:
        """Evaluate on a single EHR benchmark.

        Creates a real EHR environment, loads patient data from the benchmark,
        and runs each task with the agent.

        Returns:
            Results dictionary with per-task and aggregate metrics.
        """
        from bioagents.domains.ehr_management.data_model import EHRDB
        from bioagents.domains.ehr_management.environment import get_environment
        from bioagents.gym.agent_env import BioAgentGymEnv

        db_dict, tasks = self.load_benchmark(benchmark_key)
        runner = self._get_runner()

        info = BENCHMARK_DATA[benchmark_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"  EHR Benchmark: {info['name']}")
        logger.info(f"  Tasks: {len(tasks)}")
        logger.info(f"{'='*60}")

        # Create EHRDB from benchmark data
        ehr_db = EHRDB.model_validate(db_dict)

        # Create environment with the benchmark DB
        env = get_environment(db=ehr_db, max_turns=self.config.max_turns)

        # We need a gym-compatible env for the agent runner
        # Create a lightweight wrapper
        gym_env = _EHRBenchGymEnv(
            ehr_db=ehr_db,
            tasks=tasks,
            max_turns=self.config.max_turns,
        )

        task_results = []
        t0 = time.time()

        for i, task in enumerate(tasks):
            task_id = task["id"]

            try:
                result = runner.run_task(task, gym_env)
                task_result = EHRTaskResult(
                    task_id=task_id,
                    source=benchmark_key,
                    category=task.get("category", ""),
                    action_score=result.action_score,
                    tool_calls_made=[
                        t.parsed_tool_call.get("name", "") if t.parsed_tool_call else ""
                        for t in result.turns if t.parsed_tool_call
                    ],
                    expected_tools=[
                        a.get("tool", a.get("name", ""))
                        for a in task.get("expected_actions", task.get("evaluation_criteria", {}).get("actions", []))
                    ],
                    total_turns=result.total_turns,
                    final_answer=result.trajectory.get("final_answer", ""),
                    completed=result.completed,
                    latency=result.total_latency,
                )
            except Exception as e:
                logger.error(f"  Error on task {task_id}: {e}")
                task_result = EHRTaskResult(
                    task_id=task_id,
                    source=benchmark_key,
                    category=task.get("category", ""),
                    error=str(e),
                )

            task_results.append(task_result)

            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                completed = sum(1 for r in task_results if r.completed)
                avg_score = sum(r.action_score for r in task_results) / max(len(task_results), 1)
                elapsed = time.time() - t0
                logger.info(
                    f"  [{benchmark_key}] {i+1}/{len(tasks)} tasks, "
                    f"completed={completed}, avg_action_score={avg_score:.3f}, "
                    f"elapsed={elapsed:.0f}s"
                )

        # ── Aggregate results ─────────────────────────────────────────────
        completed_results = [r for r in task_results if r.completed]
        n_completed = len(completed_results)
        n_total = len(task_results)
        n_errors = sum(1 for r in task_results if r.error)

        avg_action_score = (
            sum(r.action_score for r in completed_results) / max(n_completed, 1)
        )
        avg_turns = (
            sum(r.total_turns for r in completed_results) / max(n_completed, 1)
        )
        avg_latency = (
            sum(r.latency for r in completed_results) / max(n_completed, 1)
        )

        # Per-category breakdown
        category_scores = {}
        for r in completed_results:
            cat = r.category
            if cat not in category_scores:
                category_scores[cat] = {"scores": [], "count": 0}
            category_scores[cat]["scores"].append(r.action_score)
            category_scores[cat]["count"] += 1
        for cat in category_scores:
            scores = category_scores[cat]["scores"]
            category_scores[cat]["avg_score"] = sum(scores) / max(len(scores), 1)

        # Tool usage analysis
        all_tools_used = []
        for r in completed_results:
            all_tools_used.extend(r.tool_calls_made)
        tool_usage = {}
        for t in set(all_tools_used):
            if t:
                tool_usage[t] = all_tools_used.count(t)

        results = {
            "benchmark": benchmark_key,
            "name": info["name"],
            "source": info["source"],
            "total_tasks": n_total,
            "completed": n_completed,
            "errors": n_errors,
            "avg_action_score": avg_action_score,
            "avg_turns": avg_turns,
            "avg_latency": avg_latency,
            "category_breakdown": category_scores,
            "tool_usage": tool_usage,
            "per_task": [
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "action_score": r.action_score,
                    "total_turns": r.total_turns,
                    "tools_used": r.tool_calls_made,
                    "completed": r.completed,
                    "error": r.error,
                }
                for r in task_results
            ],
        }

        return results

    def evaluate_all(self) -> dict:
        """Evaluate on all configured EHR benchmarks."""
        all_results = {}

        for benchmark_key in self.config.benchmarks:
            try:
                results = self.evaluate_benchmark(benchmark_key)
                all_results[benchmark_key] = results
                logger.info(
                    f"  {benchmark_key}: action_score={results['avg_action_score']:.3f} "
                    f"({results['completed']}/{results['total_tasks']} completed)"
                )
            except Exception as e:
                logger.error(f"  Error evaluating {benchmark_key}: {e}")
                import traceback
                traceback.print_exc()
                all_results[benchmark_key] = {"error": str(e)}

        # Save results
        self._save_results(all_results)
        self._print_summary(all_results)

        return all_results

    def _save_results(self, all_results: dict):
        """Save EHR benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(self.config.output_dir)
            / f"ehr_bench_{self.config.model_name}_{timestamp}.json"
        )

        report = {
            "model_name": self.config.model_name,
            "model_path": self.config.model_name_or_path,
            "timestamp": datetime.now().isoformat(),
            "category": "ehr_benchmark",
            "benchmarks": {},
        }

        for key, results in all_results.items():
            if "error" not in results:
                report["benchmarks"][key] = {
                    "name": results["name"],
                    "total_tasks": results["total_tasks"],
                    "completed": results["completed"],
                    "avg_action_score": results["avg_action_score"],
                    "avg_turns": results["avg_turns"],
                    "category_breakdown": results.get("category_breakdown", {}),
                }
            else:
                report["benchmarks"][key] = {"error": results["error"]}

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"EHR benchmark results saved to {output_path}")

    def _print_summary(self, all_results: dict):
        """Print EHR benchmark evaluation summary."""
        print(f"\n{'='*70}")
        print(f"  EHR BENCHMARK RESULTS: {self.config.model_name}")
        print(f"{'='*70}")

        for key, results in all_results.items():
            if "error" in results:
                print(f"\n  {key}: ERROR - {results['error'][:80]}")
                continue

            info = BENCHMARK_DATA.get(key, {})
            print(f"\n  {info.get('name', key)}")
            print(f"  {'─'*50}")
            print(f"    Tasks: {results['completed']}/{results['total_tasks']} completed")
            print(f"    Avg Action Score: {results['avg_action_score']:.3f}")
            print(f"    Avg Turns: {results['avg_turns']:.1f}")
            print(f"    Avg Latency: {results['avg_latency']:.1f}s")

            # Category breakdown
            cats = results.get("category_breakdown", {})
            if cats:
                print(f"    {'Category':<30} {'Score':>8} {'N':>5}")
                print(f"    {'─'*43}")
                for cat, info_cat in sorted(cats.items()):
                    print(
                        f"    {cat:<30} "
                        f"{info_cat['avg_score']:>8.3f} "
                        f"{info_cat['count']:>5}"
                    )

        # Overall
        valid = [r for r in all_results.values() if "error" not in r]
        if valid:
            overall_score = sum(r["avg_action_score"] for r in valid) / len(valid)
            total_tasks = sum(r["total_tasks"] for r in valid)
            total_completed = sum(r["completed"] for r in valid)
            print(f"\n  {'═'*50}")
            print(f"  OVERALL: {overall_score:.3f} action_score")
            print(f"  Total: {total_completed}/{total_tasks} tasks across {len(valid)} EHR databases")
        print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight Gym Env Wrapper for EHR Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

class _EHRBenchGymEnv:
    """Minimal Gym-compatible wrapper for EHR benchmark evaluation.

    Wraps the EHR Environment so the AgentRunner can use it with:
        obs, info = env.reset(options={"task_id": ...})
        obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, ehr_db, tasks: list, max_turns: int = 15):
        from bioagents.domains.ehr_management.tools import EHRTools
        from bioagents.domains.ehr_management.data_model import POLICY_PATH

        self._db = ehr_db
        self._tasks = tasks
        self._task_map = {t["id"]: t for t in tasks}
        self._max_turns = max_turns
        self._tools = EHRTools(ehr_db)
        self._tool_call_log = []
        self._turn_count = 0

        # Load policy
        try:
            with open(POLICY_PATH, "r", encoding="utf-8") as f:
                self._policy = f.read()
        except FileNotFoundError:
            self._policy = (
                "You are an EHR analysis AI. Use the available tools to navigate "
                "electronic health records, identify clinical findings, and provide "
                "evidence-based assessments."
            )

    def reset(self, options: dict = None):
        """Reset environment for a new task."""
        self._tool_call_log = []
        self._turn_count = 0

        task_id = options.get("task_id") if options else None
        task = self._task_map.get(task_id, self._tasks[0])

        obs = task.get("ticket", "")
        info = {
            "policy": self._policy,
            "tools": self._tools.get_tool_definitions_dict(),
            "max_turns": self._max_turns,
            "task": task,
        }
        return obs, info

    def step(self, action: str):
        """Execute an action."""
        import json as _json

        self._turn_count += 1
        observation = ""
        reward = 0.0
        terminated = False
        truncated = self._turn_count >= self._max_turns

        try:
            parsed = _json.loads(action)
            if isinstance(parsed, dict) and "name" in parsed:
                tool_name = parsed["name"]
                arguments = parsed.get("arguments", {})

                self._tool_call_log.append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "turn": self._turn_count,
                })

                if tool_name == "submit_answer":
                    terminated = True
                    observation = f"Answer submitted: {arguments.get('answer', '')}"
                else:
                    try:
                        result = self._tools.use_tool(tool_name, **arguments)
                        if isinstance(result, (dict, list)):
                            observation = _json.dumps(result, default=str, ensure_ascii=False)
                        else:
                            observation = str(result)
                    except Exception as e:
                        observation = f"Tool error: {e}"
            else:
                observation = action
        except (_json.JSONDecodeError, TypeError):
            observation = action

        info = {
            "turn_count": self._turn_count,
        }
        return observation, reward, terminated, truncated, info

    def get_trajectory(self):
        """Return trajectory information."""
        return {
            "tool_calls": self._tool_call_log,
            "total_turns": self._turn_count,
            "final_reward": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="EHR Benchmark Evaluation (MIMIC-III & eICU)"
    )
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--model-name", default="BIOAgent")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mimic_iii", "eicu"],
        choices=["mimic_iii", "eicu"],
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--backend", default="transformers",
                        choices=["transformers", "vllm"])
    parser.add_argument("--output-dir", default="results/ehr_benchmarks")
    args = parser.parse_args()

    config = EHRBenchmarkConfig(
        model_name_or_path=args.model,
        model_name=args.model_name,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        backend=args.backend,
        output_dir=args.output_dir,
    )
    evaluator = EHRBenchmarkEvaluator(config)
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
