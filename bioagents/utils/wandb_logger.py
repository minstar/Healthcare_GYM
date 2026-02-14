"""Centralized Weights & Biases Logger for Healthcare AI GYM.

All W&B logging flows through this module to ensure:
1. Consistent project name (`pt2-minstar-gym-rl`)
2. Structured run naming: {agent_id}/{domain}/{strategy}/{timestamp}
3. Standard metric namespacing
4. Graceful fallback if wandb is not installed
5. Artifact management for checkpoints & trajectories

Usage:
    from bioagents.utils.wandb_logger import GymWandbLogger

    # Initialize for a training run
    wb = GymWandbLogger.init_run(
        agent_id="qwen3_weakness_fixer",
        run_type="grpo_train",
        domain="clinical_diagnosis",
        reward_strategy="adaptive",
        model_name="Qwen3-8B-Base",
        config={...},
    )

    # Log metrics
    wb.log_step({"reward/mean": 0.72, "reward/accuracy": 0.65}, step=100)
    wb.log_cycle(cycle_result)
    wb.finish()
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ============================================================
# Constants
# ============================================================

WANDB_PROJECT = "pt2-minstar-gym-rl"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)  # Use default entity


def _wandb_available() -> bool:
    """Check if wandb is installed and importable."""
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================
# GymWandbLogger
# ============================================================

class GymWandbLogger:
    """Centralized W&B logger for the Healthcare AI GYM.

    Handles:
    - Run initialization with structured naming
    - Metric logging with namespace prefixes
    - Cycle-level summaries (for AutonomousAgent)
    - Training step metrics (for GRPO/SFT trainers)
    - Artifact logging (checkpoints, trajectories)
    - Graceful no-op when wandb is unavailable
    """

    def __init__(self, run=None, enabled: bool = True):
        self._run = run
        self._enabled = enabled and _wandb_available()
        self._step_counter = 0

    @classmethod
    def init_run(
        cls,
        agent_id: str = "",
        run_type: str = "train",
        domain: str = "",
        reward_strategy: str = "grpo",
        model_name: str = "",
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        resume: Optional[str] = None,
        enabled: bool = True,
    ) -> "GymWandbLogger":
        """Initialize a W&B run with structured naming.

        Args:
            agent_id: Agent identifier (e.g., "qwen3_weakness_fixer")
            run_type: "grpo_train", "sft_train", "eval", "benchmark", "gym_session"
            domain: Target domain (e.g., "clinical_diagnosis")
            reward_strategy: Reward strategy (grpo/mrpo/sarl/adaptive)
            model_name: Model name for display
            config: Run configuration dict
            tags: Additional tags
            group: W&B run group
            resume: Resume run ID
            enabled: Whether to actually log to W&B

        Returns:
            GymWandbLogger instance (no-op if disabled or wandb unavailable)
        """
        if not enabled or not _wandb_available():
            logger.info("[W&B] Disabled or wandb not installed — running in offline mode")
            return cls(run=None, enabled=False)

        import wandb

        # Build structured run name
        timestamp = datetime.now().strftime("%m%d_%H%M")
        name_parts = []
        if agent_id:
            name_parts.append(agent_id)
        if domain:
            name_parts.append(domain)
        if reward_strategy and reward_strategy != "grpo":
            name_parts.append(reward_strategy)
        name_parts.append(run_type)
        name_parts.append(timestamp)
        run_name = "/".join(name_parts)

        # Build tags
        all_tags = [run_type]
        if domain:
            all_tags.append(f"domain:{domain}")
        if reward_strategy:
            all_tags.append(f"strategy:{reward_strategy}")
        if model_name:
            # Extract short model name
            short_name = model_name.split("/")[-1].split("-")[0]
            all_tags.append(f"model:{short_name}")
        if tags:
            all_tags.extend(tags)

        # Build config
        run_config = {
            "agent_id": agent_id,
            "run_type": run_type,
            "domain": domain,
            "reward_strategy": reward_strategy,
            "model_name": model_name,
        }
        if config:
            run_config.update(config)

        # Group defaults: agent_id for agent-level grouping
        if group is None and agent_id:
            group = agent_id

        try:
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_name,
                config=run_config,
                tags=all_tags,
                group=group,
                resume=resume,
                reinit=True,
            )
            logger.info(
                f"[W&B] Run initialized: {run_name} "
                f"(project={WANDB_PROJECT}, id={run.id})"
            )
            return cls(run=run, enabled=True)

        except Exception as e:
            logger.warning(f"[W&B] Failed to init run: {e}")
            return cls(run=None, enabled=False)

    @property
    def run(self):
        return self._run

    @property
    def is_active(self) -> bool:
        return self._enabled and self._run is not None

    # ---- Step-level logging ----

    def log_step(self, metrics: dict, step: Optional[int] = None, prefix: str = ""):
        """Log metrics for a training step."""
        if not self.is_active:
            return

        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if step is not None:
            self._run.log(metrics, step=step)
        else:
            self._run.log(metrics)

    # ---- Epoch-level logging ----

    def log_epoch(
        self,
        epoch: int,
        mean_reward: float,
        num_trajectories: int,
        positive_trajectories: int,
        loss: float = 0.0,
        reward_detail: Optional[dict] = None,
        **kwargs,
    ):
        """Log epoch-level training metrics."""
        if not self.is_active:
            return

        metrics = {
            "epoch": epoch,
            "train/mean_reward": mean_reward,
            "train/num_trajectories": num_trajectories,
            "train/positive_trajectories": positive_trajectories,
            "train/loss": loss,
            "train/positive_ratio": (
                positive_trajectories / max(num_trajectories, 1)
            ),
        }
        if reward_detail:
            for k, v in reward_detail.items():
                if isinstance(v, (int, float)):
                    metrics[f"train/reward_{k}"] = v

        metrics.update(kwargs)
        self._run.log(metrics)

    # ---- Cycle-level logging (Autonomous Agent) ----

    def log_cycle(self, cycle_result: dict):
        """Log a full autonomous agent cycle.

        Args:
            cycle_result: Output from AutonomousAgent.run_one_cycle()
        """
        if not self.is_active:
            return

        cycle_num = cycle_result.get("cycle", 0)

        # Core metrics
        metrics = {
            "cycle": cycle_num,
            "cycle/gpu_id": cycle_result.get("gpu_id", -1),
            "cycle/duration_s": cycle_result.get("duration_ms", 0) / 1000,
        }

        # Reflection
        reflection = cycle_result.get("reflection", {})
        if reflection:
            metrics["reflection/overall_score"] = reflection.get("overall_score", 0)
            metrics["reflection/num_strengths"] = len(reflection.get("strengths", []))
            metrics["reflection/num_weaknesses"] = len(reflection.get("weaknesses", []))
            metrics["reflection/num_plateaus"] = len(reflection.get("plateaus", []))

        # Decision
        decision = cycle_result.get("decision", {})
        if decision:
            metrics["decision/domain"] = decision.get("domain", "")
            metrics["decision/motivation"] = decision.get("motivation", "")
            metrics["decision/reward_strategy"] = decision.get("reward_strategy", "grpo")

        # Workout
        workout = cycle_result.get("workout", {})
        if workout:
            metrics["workout/pre_score"] = workout.get("pre_score", 0)
            metrics["workout/post_score"] = workout.get("post_score", 0)
            metrics["workout/improvement"] = workout.get("improvement", 0)
            metrics["workout/tasks_completed"] = workout.get("tasks_completed", 0)
            metrics["workout/success"] = 1 if workout.get("success") else 0
            metrics["workout/num_errors"] = len(workout.get("errors", []))

        # Benchmarks
        benchmarks = cycle_result.get("benchmarks", {})
        if benchmarks:
            for category, cat_results in benchmarks.items():
                if isinstance(cat_results, dict) and not category.endswith("_error"):
                    for bench_name, res in cat_results.items():
                        if isinstance(res, dict):
                            score = (
                                res.get("accuracy")
                                or res.get("token_f1")
                                or res.get("avg_action_score")
                            )
                            if score is not None:
                                metrics[f"benchmark/{bench_name}"] = score

        self._run.log(metrics)

    # ---- Benchmark logging ----

    def log_benchmark(self, bench_name: str, results: dict, cycle: int = 0):
        """Log external benchmark evaluation results."""
        if not self.is_active:
            return

        metrics = {"cycle": cycle}
        if isinstance(results, dict):
            for key, val in results.items():
                if isinstance(val, (int, float)):
                    metrics[f"benchmark/{bench_name}/{key}"] = val
                elif isinstance(val, dict):
                    for k2, v2 in val.items():
                        if isinstance(v2, (int, float)):
                            metrics[f"benchmark/{bench_name}/{k2}"] = v2

        self._run.log(metrics)

    # ---- Reward strategy logging ----

    def log_reward_strategy_selection(
        self,
        task_id: str,
        selected_strategy: str,
        task_characteristics: Optional[dict] = None,
    ):
        """Log which reward strategy was selected for a task."""
        if not self.is_active:
            return

        import wandb
        table_data = {
            "task_id": task_id,
            "strategy": selected_strategy,
        }
        if task_characteristics:
            table_data.update(task_characteristics)

        self._run.log({
            "strategy/selected": selected_strategy,
        })

    # ---- Artifact logging ----

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[dict] = None,
    ):
        """Log a file/directory as a W&B artifact."""
        if not self.is_active:
            return

        import wandb
        try:
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
            if Path(path).is_dir():
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            self._run.log_artifact(artifact)
            logger.info(f"[W&B] Artifact logged: {name} ({artifact_type})")
        except Exception as e:
            logger.warning(f"[W&B] Failed to log artifact: {e}")

    # ---- Summary ----

    def set_summary(self, key: str, value: Any):
        """Set a run summary value."""
        if not self.is_active:
            return
        self._run.summary[key] = value

    # ---- Finish ----

    def finish(self, quiet: bool = False):
        """Finish the W&B run."""
        if self.is_active:
            try:
                self._run.finish(quiet=quiet)
                if not quiet:
                    logger.info("[W&B] Run finished")
            except Exception as e:
                logger.warning(f"[W&B] Error finishing run: {e}")
            self._run = None


# ============================================================
# Helper: Get or create a global gym-level W&B run
# ============================================================

_gym_logger: Optional[GymWandbLogger] = None


def get_gym_logger() -> GymWandbLogger:
    """Get the global gym-level W&B logger (creates if needed)."""
    global _gym_logger
    if _gym_logger is None:
        _gym_logger = GymWandbLogger(run=None, enabled=False)
    return _gym_logger


def set_gym_logger(wb_logger: GymWandbLogger):
    """Set the global gym-level W&B logger."""
    global _gym_logger
    _gym_logger = wb_logger
