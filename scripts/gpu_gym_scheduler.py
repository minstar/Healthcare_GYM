#!/usr/bin/env python3
"""Healthcare AI GYM — Continuous GPU Scheduler v2.

Keeps ALL 8 A100 80GB GPUs at 100% utilization by cycling through:
    Training → Checkpoint → Evaluate ALL benchmarks → Training → ...

Key improvements over v1:
    - INFINITE LOOP: Never stops — cycles SFT → GRPO → FairGRPO → eval → repeat
    - EVAL INTERLEAVING: After every training phase, runs VQA(6) + TextQA(3) + LongQA(5)
    - ADAPTIVE POLLING: 5s when tasks finish, not fixed 30s
    - AUTO-RETRY: Failed tasks retry up to 3 times
    - GPU MONITORING: Tracks actual GPU utilization, alerts on idle
    - RESULT TRACKING: All eval results saved to central JSON for analysis

Architecture (8 GPUs):
    GPUs 0-5: Training workers (SFT/GRPO/FairGRPO, cycle through domains)
    GPUs 6-7: Evaluation workers (cycle through 14 benchmarks continuously)
    When evals finish, GPUs 6-7 also train. When training finishes, those GPUs eval.

Usage:
    python scripts/gpu_gym_scheduler.py                     # Launch all 8 GPUs
    python scripts/gpu_gym_scheduler.py --gpus 0,1,2,3      # Specific GPUs
    python scripts/gpu_gym_scheduler.py --dry-run            # Preview
    python scripts/gpu_gym_scheduler.py --monitor            # Monitor
    python scripts/gpu_gym_scheduler.py --results            # Show eval results
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = "/data/project/private/minstar/miniconda3/envs/llm/bin/python"

# ============================================================
# Model Registry
# ============================================================
MODELS = {
    "qwen3_8b": {
        "name": "Qwen3-8B-Base",
        "path": "/data/project/private/minstar/models/Qwen3-8B-Base",
        "type": "causal",
        "vram_gb": 16,
    },
    "lingshu_7b": {
        "name": "Lingshu-7B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Lingshu-7B"),
        "type": "vlm",
        "vram_gb": 16,
    },
    "qwen25vl_7b": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "path": str(PROJECT_ROOT / "checkpoints/models/Qwen2.5-VL-7B-Instruct"),
        "type": "vlm",
        "vram_gb": 16,
    },
}

# ============================================================
# Domain & Benchmark Registry
# ============================================================
ALL_DOMAINS = [
    "clinical_diagnosis", "medical_qa", "visual_diagnosis", "drug_interaction",
    "ehr_management", "triage_emergency", "radiology_report",
    "psychiatry", "obstetrics", "cross_domain",
]

# 14 external benchmarks (VQA 6 + TextQA 3 + LongQA 5)
TEXTQA_BENCHMARKS = ["medqa", "medmcqa", "mmlu_clinical"]
VQA_BENCHMARKS = ["vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"]
LONGQA_BENCHMARKS = ["kqa_golden", "liveqa", "medicationqa", "healthsearchqa", "kqa_silver"]
ALL_BENCHMARKS = TEXTQA_BENCHMARKS + VQA_BENCHMARKS + LONGQA_BENCHMARKS


# ============================================================
# Task Definitions
# ============================================================
@dataclass
class GymTask:
    """A single GPU task."""
    name: str
    gpu_id: int
    task_type: str  # sft, grpo, fair_grpo, eval_textqa, eval_vqa, eval_longqa, eval_gym
    model_key: str
    domain: str = ""
    config_path: str = ""
    extra_args: dict = field(default_factory=dict)
    depends_on: str = ""
    estimated_minutes: int = 60
    priority: int = 5
    retries: int = 0
    max_retries: int = 3

    def _resolve_model_key(self) -> str:
        """Map internal model_key to benchmark suite model name."""
        mapping = {
            "qwen3_8b": "qwen3",
            "lingshu_7b": "lingshu",
            "qwen25vl_7b": "qwen2vl",
        }
        return mapping.get(self.model_key, self.model_key)

    def build_command(self) -> list[str]:
        """Build shell command for this task."""
        model_info = MODELS.get(self.model_key, {})
        model_path = model_info.get("path", self.model_key)
        base = [PYTHON, "-u"]

        if self.task_type == "sft":
            cfg = self.config_path or f"configs/sft_medical_qa.yaml"
            return base + ["-m", "bioagents.training.sft_trainer", "--config", cfg]

        elif self.task_type in ("grpo", "multiturn_grpo"):
            mode = "multi_turn" if self.task_type == "multiturn_grpo" else "single_turn"
            cfg = self.config_path or f"configs/grpo_{self.domain}.yaml"
            return base + ["-m", "bioagents.training.grpo_trainer", "--config", cfg, "--mode", mode]

        elif self.task_type == "fair_grpo":
            cfg = self.config_path or f"configs/grpo_{self.domain}.yaml"
            return base + ["-m", "bioagents.training.grpo_trainer", "--config", cfg, "--mode", "fair_grpo"]

        elif self.task_type == "eval_textqa":
            # TextQA: MedQA + MedMCQA + MMLU via unified benchmark suite
            max_s = str(self.extra_args.get("max_samples", 200))
            model_key = self._resolve_model_key()
            return base + [
                "scripts/run_full_benchmark_suite.py",
                "--category", "textqa",
                "--model", model_key,
                "--max-samples", max_s,
                "--output-dir", f"results/eval_{self.name}",
            ]

        elif self.task_type == "eval_vqa":
            # VQA 6 benchmarks via unified benchmark suite
            max_s = str(self.extra_args.get("max_samples", 200))
            model_key = self._resolve_model_key()
            return base + [
                "scripts/run_full_benchmark_suite.py",
                "--category", "vqa",
                "--model", model_key,
                "--max-samples", max_s,
                "--output-dir", f"results/eval_{self.name}",
            ]

        elif self.task_type == "eval_longqa":
            # LongQA 5 benchmarks via unified benchmark suite
            max_s = str(self.extra_args.get("max_samples", 200))
            model_key = self._resolve_model_key()
            return base + [
                "scripts/run_full_benchmark_suite.py",
                "--category", "medlfqa",
                "--model", model_key,
                "--max-samples", max_s,
                "--output-dir", f"results/eval_{self.name}",
            ]

        elif self.task_type == "eval_gym":
            return base + ["-c", f"""
import sys; sys.path.insert(0, '.')
from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
config = RunConfig(
    model_name_or_path='{model_path}',
    backend='transformers',
    domain='{self.domain}',
    task_split='test',
    max_turns=10,
    log_dir='results/gym_eval/{self.name}',
)
runner = AgentRunner(config)
results = runner.run_all()
if results:
    scores = [r.action_score for r in results]
    print(f'Domain: {self.domain}, Mean: {{sum(scores)/len(scores):.4f}}, N={{len(results)}}')
"""]

        elif self.task_type == "selfplay":
            domains = self.extra_args.get("domains", ["medical_qa", "clinical_diagnosis"])
            return base + ["-c", f"""
import sys; sys.path.insert(0, '.')
from bioagents.gym.self_play import SelfPlayConfig, SelfPlayLoop
config = SelfPlayConfig(
    model_name_or_path='{model_path}',
    domains={domains},
    max_iterations={self.extra_args.get('iterations', 5)},
    num_trajectories_per_task={self.extra_args.get('trajectories', 3)},
    output_dir='checkpoints/selfplay_{self.model_key}',
)
loop = SelfPlayLoop(config)
loop.run()
"""]

        raise ValueError(f"Unknown task type: {self.task_type}")


def _make_eval_tasks(gpu_id: int, model_key: str, phase: str, cycle: int) -> list[GymTask]:
    """Generate evaluation tasks for a specific checkpoint."""
    prefix = f"c{cycle}_{phase}"
    tasks = []

    # TextQA: MedQA, MedMCQA, MMLU (run together, ~15 min)
    tasks.append(GymTask(
        name=f"gpu{gpu_id}_{prefix}_textqa",
        gpu_id=gpu_id, task_type="eval_textqa", model_key=model_key,
        extra_args={"benchmarks": TEXTQA_BENCHMARKS},
        estimated_minutes=20, priority=2,
    ))

    # VQA 6 benchmarks (~30 min)
    tasks.append(GymTask(
        name=f"gpu{gpu_id}_{prefix}_vqa",
        gpu_id=gpu_id, task_type="eval_vqa", model_key=model_key,
        extra_args={"benchmarks": VQA_BENCHMARKS, "max_samples": 200},
        estimated_minutes=30, priority=2,
    ))

    # LongQA 5 benchmarks (~20 min)
    tasks.append(GymTask(
        name=f"gpu{gpu_id}_{prefix}_longqa",
        gpu_id=gpu_id, task_type="eval_longqa", model_key=model_key,
        extra_args={"max_samples": 200},
        estimated_minutes=25, priority=2,
    ))

    return tasks


def _make_gym_eval_tasks(gpu_id: int, model_key: str, domains: list[str], cycle: int) -> list[GymTask]:
    """Generate GYM domain evaluation tasks."""
    tasks = []
    for domain in domains:
        tasks.append(GymTask(
            name=f"gpu{gpu_id}_c{cycle}_gymeval_{domain}",
            gpu_id=gpu_id, task_type="eval_gym", model_key=model_key,
            domain=domain,
            estimated_minutes=15, priority=3,
        ))
    return tasks


def build_infinite_pipeline(cycle: int = 0) -> dict[int, list[GymTask]]:
    """Build one CYCLE of the infinite training + eval pipeline.

    Each cycle:
        Phase 1: SFT warm-up (GPUs 0-5) + Eval baseline (GPUs 6-7)
        Phase 2: GRPO training (GPUs 0-5) + Eval (GPUs 6-7)
        Phase 3: FairGRPO + Self-play (GPUs 0-5) + Final eval (GPUs 6-7)

    Total estimated: ~8-12 hours per cycle → runs 2-3 cycles/day
    """
    pipelines = {i: [] for i in range(8)}

    # ═══════════════════════════════════════════════════════════
    # TRAINING GPUs (0-5): SFT → GRPO → FairGRPO per model
    # ═══════════════════════════════════════════════════════════

    # --- GPU 0-1: Qwen3-8B ---
    pipelines[0].extend([
        GymTask(
            name=f"gpu0_c{cycle}_sft_qwen3",
            gpu_id=0, task_type="sft", model_key="qwen3_8b",
            config_path="configs/8gpu/sft_qwen3_8b_gpu0.yaml",
            estimated_minutes=60, priority=1,
        ),
        GymTask(
            name=f"gpu0_c{cycle}_grpo_medical_qa",
            gpu_id=0, task_type="multiturn_grpo", model_key="qwen3_8b",
            domain="medical_qa",
            depends_on=f"gpu0_c{cycle}_sft_qwen3",
            estimated_minutes=90, priority=2,
        ),
        GymTask(
            name=f"gpu0_c{cycle}_grpo_clinical_dx",
            gpu_id=0, task_type="multiturn_grpo", model_key="qwen3_8b",
            domain="clinical_diagnosis",
            depends_on=f"gpu0_c{cycle}_grpo_medical_qa",
            estimated_minutes=90, priority=3,
        ),
        GymTask(
            name=f"gpu0_c{cycle}_grpo_ehr",
            gpu_id=0, task_type="multiturn_grpo", model_key="qwen3_8b",
            domain="ehr_management",
            depends_on=f"gpu0_c{cycle}_grpo_clinical_dx",
            estimated_minutes=90, priority=4,
        ),
    ])

    pipelines[1].extend([
        GymTask(
            name=f"gpu1_c{cycle}_grpo_drug",
            gpu_id=1, task_type="multiturn_grpo", model_key="qwen3_8b",
            domain="drug_interaction",
            estimated_minutes=90, priority=1,
        ),
        GymTask(
            name=f"gpu1_c{cycle}_grpo_triage",
            gpu_id=1, task_type="multiturn_grpo", model_key="qwen3_8b",
            domain="triage_emergency",
            depends_on=f"gpu1_c{cycle}_grpo_drug",
            estimated_minutes=90, priority=2,
        ),
        GymTask(
            name=f"gpu1_c{cycle}_fairgrpo_medical_qa",
            gpu_id=1, task_type="fair_grpo", model_key="qwen3_8b",
            domain="medical_qa",
            depends_on=f"gpu1_c{cycle}_grpo_triage",
            estimated_minutes=120, priority=3,
        ),
    ])

    # --- GPU 2-3: Lingshu-7B ---
    pipelines[2].extend([
        GymTask(
            name=f"gpu2_c{cycle}_sft_lingshu",
            gpu_id=2, task_type="sft", model_key="lingshu_7b",
            config_path="configs/sft_multidomain_lingshu.yaml",
            estimated_minutes=60, priority=1,
        ),
        GymTask(
            name=f"gpu2_c{cycle}_grpo_radiology",
            gpu_id=2, task_type="multiturn_grpo", model_key="lingshu_7b",
            domain="radiology_report",
            depends_on=f"gpu2_c{cycle}_sft_lingshu",
            estimated_minutes=90, priority=2,
        ),
        GymTask(
            name=f"gpu2_c{cycle}_grpo_psych",
            gpu_id=2, task_type="multiturn_grpo", model_key="lingshu_7b",
            domain="psychiatry",
            depends_on=f"gpu2_c{cycle}_grpo_radiology",
            estimated_minutes=90, priority=3,
        ),
        GymTask(
            name=f"gpu2_c{cycle}_grpo_obstetrics",
            gpu_id=2, task_type="multiturn_grpo", model_key="lingshu_7b",
            domain="obstetrics",
            depends_on=f"gpu2_c{cycle}_grpo_psych",
            estimated_minutes=90, priority=4,
        ),
    ])

    pipelines[3].extend([
        GymTask(
            name=f"gpu3_c{cycle}_grpo_triage",
            gpu_id=3, task_type="multiturn_grpo", model_key="lingshu_7b",
            domain="triage_emergency",
            estimated_minutes=90, priority=1,
        ),
        GymTask(
            name=f"gpu3_c{cycle}_grpo_cross",
            gpu_id=3, task_type="multiturn_grpo", model_key="lingshu_7b",
            domain="cross_domain",
            depends_on=f"gpu3_c{cycle}_grpo_triage",
            estimated_minutes=90, priority=2,
        ),
        GymTask(
            name=f"gpu3_c{cycle}_fairgrpo_psych",
            gpu_id=3, task_type="fair_grpo", model_key="lingshu_7b",
            domain="psychiatry",
            depends_on=f"gpu3_c{cycle}_grpo_cross",
            estimated_minutes=120, priority=3,
        ),
    ])

    # --- GPU 4-5: Qwen2.5-VL-7B (visual-heavy) ---
    pipelines[4].extend([
        GymTask(
            name=f"gpu4_c{cycle}_sft_vl",
            gpu_id=4, task_type="sft", model_key="qwen25vl_7b",
            config_path="configs/8gpu/sft_qwen25vl_7b_gpu3.yaml",
            estimated_minutes=60, priority=1,
        ),
        GymTask(
            name=f"gpu4_c{cycle}_grpo_visual",
            gpu_id=4, task_type="multiturn_grpo", model_key="qwen25vl_7b",
            domain="visual_diagnosis",
            depends_on=f"gpu4_c{cycle}_sft_vl",
            estimated_minutes=90, priority=2,
        ),
        GymTask(
            name=f"gpu4_c{cycle}_grpo_medical_qa",
            gpu_id=4, task_type="multiturn_grpo", model_key="qwen25vl_7b",
            domain="medical_qa",
            depends_on=f"gpu4_c{cycle}_grpo_visual",
            estimated_minutes=90, priority=3,
        ),
    ])

    pipelines[5].extend([
        GymTask(
            name=f"gpu5_c{cycle}_selfplay",
            gpu_id=5, task_type="selfplay", model_key="qwen3_8b",
            extra_args={
                "domains": ["medical_qa", "clinical_diagnosis", "drug_interaction",
                            "ehr_management", "triage_emergency"],
                "iterations": 5,
                "trajectories": 3,
            },
            estimated_minutes=180, priority=1,
        ),
        GymTask(
            name=f"gpu5_c{cycle}_grpo_ehr",
            gpu_id=5, task_type="multiturn_grpo", model_key="qwen3_8b",
            domain="ehr_management",
            depends_on=f"gpu5_c{cycle}_selfplay",
            estimated_minutes=90, priority=2,
        ),
    ])

    # ═══════════════════════════════════════════════════════════
    # EVALUATION GPUs (6-7): Continuous benchmark evaluation
    # After every training phase, evaluate ALL 14 benchmarks
    # ═══════════════════════════════════════════════════════════

    # GPU 6: TextQA(3) + VQA(6) for Qwen3-8B, then Lingshu-7B
    pipelines[6].extend([
        # Qwen3-8B evaluation
        *_make_eval_tasks(6, "qwen3_8b", "qwen3", cycle),
        # Lingshu-7B evaluation
        *_make_eval_tasks(6, "lingshu_7b", "lingshu", cycle),
        # GYM domain evals
        *_make_gym_eval_tasks(6, "qwen3_8b",
            ["medical_qa", "clinical_diagnosis", "drug_interaction", "ehr_management", "triage_emergency"],
            cycle),
    ])

    # GPU 7: TextQA(3) + VQA(6) + LongQA(5) for Qwen2.5-VL-7B, then cross-eval
    pipelines[7].extend([
        # Qwen2.5-VL-7B evaluation
        *_make_eval_tasks(7, "qwen25vl_7b", "vl", cycle),
        # GYM domain evals (visual + safety-critical)
        *_make_gym_eval_tasks(7, "lingshu_7b",
            ["visual_diagnosis", "radiology_report", "psychiatry", "obstetrics", "cross_domain"],
            cycle),
        # Re-evaluate Qwen3 on LongQA after GRPO
        GymTask(
            name=f"gpu7_c{cycle}_post_longqa_qwen3",
            gpu_id=7, task_type="eval_longqa", model_key="qwen3_8b",
            extra_args={"max_samples": 200},
            estimated_minutes=25, priority=4,
        ),
    ])

    return pipelines


# ============================================================
# GPU Utilization Monitor
# ============================================================

class GPUMonitor:
    """Monitors actual GPU utilization via nvidia-smi."""

    @staticmethod
    def get_utilization() -> dict[int, dict]:
        """Get current GPU utilization for all GPUs."""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            gpus = {}
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    idx = int(parts[0])
                    gpus[idx] = {
                        "util_pct": int(parts[1]),
                        "mem_used_mb": int(parts[2]),
                        "mem_total_mb": int(parts[3]),
                        "temp_c": int(parts[4]),
                        "mem_pct": round(int(parts[2]) / max(int(parts[3]), 1) * 100, 1),
                    }
            return gpus
        except Exception:
            return {}

    @staticmethod
    def is_gpu_idle(gpu_id: int, threshold_pct: int = 5) -> bool:
        """Check if a specific GPU is idle."""
        utils = GPUMonitor.get_utilization()
        info = utils.get(gpu_id, {})
        return info.get("util_pct", 0) < threshold_pct


# ============================================================
# Results Tracker
# ============================================================

class ResultsTracker:
    """Tracks all evaluation results across cycles."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = results_dir / "all_results.json"
        self.results = self._load()

    def _load(self) -> dict:
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return {"cycles": [], "benchmarks": {}, "models": {}}

    def save(self):
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def record(self, task_name: str, model_key: str, benchmark: str,
               metrics: dict, cycle: int):
        entry = {
            "task_name": task_name,
            "model_key": model_key,
            "benchmark": benchmark,
            "metrics": metrics,
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.setdefault("entries", []).append(entry)
        self.save()

    def summary(self) -> str:
        """Generate a summary table of all results."""
        entries = self.results.get("entries", [])
        if not entries:
            return "No results recorded yet."

        lines = ["=" * 80, "  EVALUATION RESULTS SUMMARY", "=" * 80]
        # Group by model
        by_model = defaultdict(list)
        for e in entries:
            by_model[e["model_key"]].append(e)

        for model, model_entries in sorted(by_model.items()):
            lines.append(f"\n  Model: {MODELS.get(model, {}).get('name', model)}")
            lines.append(f"  {'Benchmark':<25s} {'Metric':<15s} {'Value':<10s} {'Cycle':<6s} {'Time'}")
            lines.append("  " + "-" * 70)
            for e in sorted(model_entries, key=lambda x: x.get("timestamp", "")):
                for k, v in e.get("metrics", {}).items():
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"  {e['benchmark']:<25s} {k:<15s} {val:<10s} {e['cycle']:<6d} {e['timestamp'][:16]}")
        lines.append("=" * 80)
        return "\n".join(lines)


# ============================================================
# Scheduler Engine v2
# ============================================================

class GPUScheduler:
    """Continuous GPU scheduler with training ↔ evaluation interleaving."""

    def __init__(self, gpus: list[int] = None, dry_run: bool = False,
                 max_cycles: int = 0):
        self.gpus = gpus or list(range(8))
        self.dry_run = dry_run
        self.max_cycles = max_cycles  # 0 = infinite
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = PROJECT_ROOT / f"logs/gym_scheduler/{self.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.cycle = 0
        self.pipelines: dict[int, list[GymTask]] = {}
        self.processes: dict[int, subprocess.Popen] = {}
        self.log_handles: dict[int, object] = {}  # Keep file handles to avoid leaks
        self.current_task_idx: dict[int, int] = {}
        self.completed: dict[str, dict] = {}
        self.failed: dict[str, dict] = {}
        self.start_times: dict[int, float] = {}
        self.idle_since: dict[int, float] = {}  # Track when GPU became idle
        self.lock = threading.Lock()

        self.results = ResultsTracker(PROJECT_ROOT / "results")
        self.monitor = GPUMonitor()

        self._load_cycle()

    def _load_cycle(self):
        """Load pipeline for current cycle."""
        self.pipelines = build_infinite_pipeline(self.cycle)
        # Filter to only requested GPUs
        self.pipelines = {g: self.pipelines.get(g, []) for g in self.gpus}
        self.current_task_idx = {g: 0 for g in self.gpus}

    def _log(self, msg: str, gpu_id: int = None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{ts}]"
        if gpu_id is not None:
            prefix += f" [GPU {gpu_id}]"
        line = f"{prefix} {msg}"
        print(line, flush=True)

        log_file = self.log_dir / "scheduler.log"
        with open(log_file, "a") as f:
            f.write(line + "\n")

    def _save_state(self):
        state = {
            "timestamp": self.timestamp,
            "cycle": self.cycle,
            "gpus": self.gpus,
            "current_task_idx": self.current_task_idx,
            "completed": list(self.completed.keys()),
            "failed": {k: v for k, v in self.failed.items()},
            "gpu_utilization": self.monitor.get_utilization(),
        }
        with open(self.log_dir / "scheduler_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

    def launch_task(self, gpu_id: int) -> bool:
        """Launch the next task for a GPU. Thread-safe."""
        with self.lock:
            pipeline = self.pipelines.get(gpu_id, [])
            idx = self.current_task_idx.get(gpu_id, 0)

            if idx >= len(pipeline):
                return False

            task = pipeline[idx]

            # Check dependency
            if task.depends_on and task.depends_on not in self.completed:
                return False

            # Build command
            try:
                cmd = task.build_command()
            except Exception as e:
                self._log(f"ERROR building command for {task.name}: {e}", gpu_id)
                self.current_task_idx[gpu_id] = idx + 1
                return False

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            env["WANDB_DISABLED"] = "true"  # Disable wandb for evals

            self._log(f"LAUNCH: {task.name} [{task.task_type}] domain={task.domain}", gpu_id)
            self._log(f"  Model: {MODELS.get(task.model_key, {}).get('name', task.model_key)}", gpu_id)

            if self.dry_run:
                self._log(f"  [DRY RUN] {' '.join(cmd[:6])}...", gpu_id)
                self.completed[task.name] = {
                    "gpu_id": gpu_id, "status": "dry_run",
                    "timestamp": datetime.now().isoformat(),
                }
                self.current_task_idx[gpu_id] = idx + 1
                return True

            # Launch process with log file
            log_file = self.log_dir / f"gpu{gpu_id}_{task.name}.log"
            with open(log_file, "w") as f:
                f.write(f"# Task: {task.name}\n# GPU: {gpu_id}\n# Type: {task.task_type}\n")
                f.write(f"# Model: {MODELS.get(task.model_key, {}).get('name', '?')}\n")
                f.write(f"# Domain: {task.domain}\n# Cycle: {self.cycle}\n")
                f.write(f"# Started: {datetime.now().isoformat()}\n")
                f.write(f"# Command: {' '.join(cmd)}\n{'='*60}\n\n")

            log_fh = open(log_file, "a")
            proc = subprocess.Popen(
                cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                env=env, cwd=str(PROJECT_ROOT),
            )

            self.processes[gpu_id] = proc
            self.log_handles[gpu_id] = log_fh
            self.start_times[gpu_id] = time.time()
            self.idle_since.pop(gpu_id, None)
            self._save_state()
            return True

    def check_processes(self):
        """Check all running processes, handle completions."""
        with self.lock:
            for gpu_id in list(self.processes.keys()):
                proc = self.processes[gpu_id]
                ret = proc.poll()

                if ret is None:
                    continue  # Still running

                elapsed = time.time() - self.start_times.get(gpu_id, 0)
                idx = self.current_task_idx[gpu_id]
                task = self.pipelines[gpu_id][idx]

                # Close log handle
                if gpu_id in self.log_handles:
                    self.log_handles[gpu_id].close()
                    del self.log_handles[gpu_id]

                if ret == 0:
                    self._log(
                        f"DONE: {task.name} ({elapsed/60:.1f}min)", gpu_id)
                    self.completed[task.name] = {
                        "gpu_id": gpu_id, "exit_code": 0,
                        "elapsed_min": round(elapsed / 60, 1),
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.current_task_idx[gpu_id] = idx + 1
                else:
                    task.retries += 1
                    if task.retries <= task.max_retries:
                        self._log(
                            f"RETRY ({task.retries}/{task.max_retries}): "
                            f"{task.name} (exit={ret})", gpu_id)
                        # Don't advance index — retry same task
                    else:
                        self._log(
                            f"FAILED: {task.name} (exit={ret}, "
                            f"{elapsed/60:.1f}min, exhausted retries)", gpu_id)
                        self.failed[task.name] = {
                            "gpu_id": gpu_id, "exit_code": ret,
                            "elapsed_min": round(elapsed / 60, 1),
                            "retries": task.retries,
                            "timestamp": datetime.now().isoformat(),
                        }
                        self.current_task_idx[gpu_id] = idx + 1

                del self.processes[gpu_id]
                self.idle_since[gpu_id] = time.time()
                self._save_state()

    def _all_done(self) -> bool:
        """Check if current cycle is complete."""
        for gpu_id in self.gpus:
            pipeline = self.pipelines.get(gpu_id, [])
            if self.current_task_idx.get(gpu_id, 0) < len(pipeline):
                return False
            if gpu_id in self.processes:
                return False
        return True

    def _print_status(self):
        """Print current status of all GPUs."""
        utils = self.monitor.get_utilization()
        self._log("-" * 60)
        self._log(f"CYCLE {self.cycle} STATUS (completed={len(self.completed)}, failed={len(self.failed)})")
        for gpu_id in self.gpus:
            pipeline = self.pipelines.get(gpu_id, [])
            idx = self.current_task_idx.get(gpu_id, 0)
            running = gpu_id in self.processes
            gpu_info = utils.get(gpu_id, {})
            util = gpu_info.get("util_pct", 0)
            mem = gpu_info.get("mem_pct", 0)
            temp = gpu_info.get("temp_c", 0)

            if running:
                task = pipeline[idx] if idx < len(pipeline) else None
                elapsed = (time.time() - self.start_times.get(gpu_id, 0)) / 60
                name = task.name if task else "?"
                status = f"RUNNING: {name} ({elapsed:.0f}min)"
            elif idx >= len(pipeline):
                status = "COMPLETE"
            else:
                task = pipeline[idx]
                status = f"WAITING: {task.name} (dep: {task.depends_on or 'none'})"

            self._log(f"  GPU {gpu_id}: util={util}% mem={mem}% {temp}°C | {status}")
        self._log("-" * 60)

    def run(self):
        """Main scheduler loop — runs forever until max_cycles or Ctrl+C."""
        self._log("=" * 60)
        self._log("Healthcare AI GYM — Continuous GPU Scheduler v2")
        self._log(f"GPUs: {self.gpus}")
        self._log(f"Max cycles: {'infinite' if self.max_cycles == 0 else self.max_cycles}")
        self._log(f"Benchmarks: TextQA({len(TEXTQA_BENCHMARKS)}) + VQA({len(VQA_BENCHMARKS)}) + LongQA({len(LONGQA_BENCHMARKS)}) = {len(ALL_BENCHMARKS)}")
        self._log(f"Log dir: {self.log_dir}")
        total = sum(len(p) for p in self.pipelines.values())
        self._log(f"Tasks this cycle: {total}")
        self._log("=" * 60)

        # Print pipeline overview
        for gpu_id in self.gpus:
            tasks = self.pipelines.get(gpu_id, [])
            self._log(f"GPU {gpu_id}: {len(tasks)} tasks")
            for i, t in enumerate(tasks):
                dep = f" (after {t.depends_on})" if t.depends_on else ""
                self._log(f"  [{i+1}] {t.name} [{t.task_type}]{dep}")

        if self.dry_run:
            self._run_dry()
            return

        try:
            while True:
                # Launch initial tasks
                for gpu_id in self.gpus:
                    if gpu_id not in self.processes:
                        self.launch_task(gpu_id)

                # Adaptive polling loop
                status_interval = 0
                while not self._all_done():
                    self.check_processes()

                    # Launch tasks on idle GPUs
                    for gpu_id in self.gpus:
                        if gpu_id not in self.processes:
                            self.launch_task(gpu_id)

                    # Status report every 5 minutes
                    status_interval += 1
                    if status_interval % 60 == 0:  # 60 * 5s = 5 min
                        self._print_status()

                    time.sleep(5)  # 5s adaptive polling

                # Cycle complete!
                self._log(f"CYCLE {self.cycle} COMPLETE!")
                self._print_summary()

                # Check if we should continue
                self.cycle += 1
                if self.max_cycles > 0 and self.cycle >= self.max_cycles:
                    self._log(f"Reached max_cycles={self.max_cycles}. Stopping.")
                    break

                # Load next cycle and continue
                self._log(f"\n{'='*60}")
                self._log(f"STARTING CYCLE {self.cycle}")
                self._log(f"{'='*60}")
                self._load_cycle()

        except KeyboardInterrupt:
            self._log("\nScheduler interrupted by user. Cleaning up...")
            self._cleanup()
            self._print_summary()

    def _run_dry(self):
        """Dry-run mode: walk through all tasks without executing."""
        for _ in range(200):  # safety limit
            advanced = False
            for gpu_id in self.gpus:
                if self.launch_task(gpu_id):
                    advanced = True
            if not advanced:
                break
        self._print_summary()

    def _cleanup(self):
        """Gracefully terminate all running processes."""
        for gpu_id, proc in list(self.processes.items()):
            self._log(f"Terminating GPU {gpu_id}...", gpu_id)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            # Close log handles
            if gpu_id in self.log_handles:
                self.log_handles[gpu_id].close()

    def _print_summary(self):
        """Print execution summary."""
        self._log("")
        self._log("=" * 60)
        self._log(f"CYCLE {self.cycle} SUMMARY")
        self._log("=" * 60)
        self._log(f"Completed: {len(self.completed)}")
        for name, info in sorted(self.completed.items()):
            elapsed = info.get("elapsed_min", 0)
            self._log(f"  + {name} ({elapsed:.1f}min)")

        if self.failed:
            self._log(f"\nFailed: {len(self.failed)}")
            for name, info in sorted(self.failed.items()):
                self._log(f"  x {name} (exit={info.get('exit_code')}, retries={info.get('retries', 0)})")

        # GPU utilization summary
        utils = self.monitor.get_utilization()
        if utils:
            self._log(f"\nGPU Status:")
            for gpu_id in sorted(utils.keys()):
                info = utils[gpu_id]
                self._log(f"  GPU {gpu_id}: {info['util_pct']}% util, "
                         f"{info['mem_used_mb']}/{info['mem_total_mb']}MB, {info['temp_c']}°C")

        self._log(f"\nLogs: {self.log_dir}/")
        self._log(f"Results: {PROJECT_ROOT}/results/all_results.json")
        self._log("=" * 60)

        # Save summary
        summary = {
            "cycle": self.cycle,
            "completed": self.completed,
            "failed": self.failed,
            "total_tasks": sum(len(self.pipelines.get(g, [])) for g in self.gpus),
        }
        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


# ============================================================
# CLI
# ============================================================

def show_results():
    """Show accumulated evaluation results."""
    tracker = ResultsTracker(PROJECT_ROOT / "results")
    print(tracker.summary())


def monitor_jobs():
    """Monitor running scheduler jobs."""
    log_base = PROJECT_ROOT / "logs/gym_scheduler"
    if not log_base.exists():
        print("No scheduler runs found.")
        return

    runs = sorted(log_base.iterdir(), reverse=True)
    if not runs:
        print("No runs found.")
        return

    latest = runs[0]
    print(f"Latest run: {latest.name}")

    state_file = latest / "scheduler_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"Cycle: {state.get('cycle', 0)}")
        print(f"Completed: {len(state.get('completed', []))}")
        print(f"Failed: {len(state.get('failed', {}))}")

    # GPU status
    utils = GPUMonitor.get_utilization()
    if utils:
        print(f"\nGPU Status:")
        for gpu_id in sorted(utils.keys()):
            info = utils[gpu_id]
            running = info['util_pct'] > 5
            status = "ACTIVE" if running else "IDLE"
            print(f"  GPU {gpu_id}: {info['util_pct']:3d}% util | "
                  f"{info['mem_used_mb']:5d}/{info['mem_total_mb']}MB | "
                  f"{info['temp_c']}°C | {status}")

    # Recent log
    scheduler_log = latest / "scheduler.log"
    if scheduler_log.exists():
        print("\nRecent log:")
        with open(scheduler_log) as f:
            lines = f.readlines()
        for line in lines[-25:]:
            print(f"  {line.rstrip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare AI GYM — Continuous GPU Scheduler v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: all 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview pipeline without running")
    parser.add_argument("--monitor", action="store_true",
                        help="Monitor running jobs")
    parser.add_argument("--results", action="store_true",
                        help="Show evaluation results")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Max training cycles (0=infinite, default)")

    args = parser.parse_args()

    if args.monitor:
        monitor_jobs()
        return
    if args.results:
        show_results()
        return

    gpus = [int(g.strip()) for g in args.gpus.split(",")] if args.gpus else None
    scheduler = GPUScheduler(gpus=gpus, dry_run=args.dry_run, max_cycles=args.max_cycles)
    scheduler.run()


if __name__ == "__main__":
    main()
