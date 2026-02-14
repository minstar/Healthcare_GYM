"""AutonomousGym -- The Shared Space Where Agents Self-Improve.

Like a real gym:
- The GYM provides equipment (domains, tasks, GPUs)
- The SCHEDULER manages facility resources (GPU allocation)
- AGENTS come and go, choosing their own workouts
- The LOGBOOK records everything for cross-agent learning

The Gym does NOT control what agents do. It only:
1. Manages GPU resources (no GPU sits idle)
2. Provides safety guardrails (stop dangerous regressions)
3. Maintains the SharedLogbook
4. Spawns and monitors agent workers

Architecture:
    AutonomousGym
    |-- GymScheduler        (GPU allocation, queue management)
    |-- SharedLogbook       (cross-agent records)
    |-- SafetyGuardrail     (prevent dangerous regressions)
    |-- AgentWorker pool    (async agent execution)

Usage:
    gym = AutonomousGym(config)
    gym.register_agent(agent_config_1)
    gym.register_agent(agent_config_2)
    gym.open()  # Start the gym -- agents train continuously

    # Or run from CLI:
    python -m bioagents.gym.autonomous_gym --config configs/autonomous_gym.yaml
"""

import json
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class AutonomousGymConfig:
    """Configuration for the Autonomous GYM."""

    # GPU resources
    num_gpus: int = 8
    gpu_ids: list = field(default_factory=lambda: list(range(8)))

    # Domains available in the gym (text-only; VL domains excluded)
    available_domains: list = field(default_factory=lambda: [
        "clinical_diagnosis", "drug_interaction", "ehr_management",
        "medical_qa", "triage_emergency", "cross_domain",
        "psychiatry", "obstetrics",
    ])

    # Safety guardrails
    safety_score_floor: float = 0.30   # Stop agent if safety drops below this
    max_consecutive_failures: int = 5   # Force cooldown after N failures
    cooldown_seconds: int = 60          # Cooldown duration

    # Scheduling
    max_queue_size: int = 50
    idle_check_interval: float = 5.0    # Seconds between idle GPU checks
    max_cycles_per_agent: int = 0       # 0 = unlimited

    # Logging
    log_dir: str = "logs/autonomous_gym"
    logbook_dir: str = "logs/shared_logbook"

    # Continuous mode
    continuous: bool = True


# ============================================================
# 2. GPU Scheduler
# ============================================================

class GPUState(Enum):
    FREE = "free"
    BUSY = "busy"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass
class GPUSlot:
    """Represents a single GPU slot in the gym."""
    gpu_id: int
    state: GPUState = GPUState.FREE
    current_agent: str = ""
    current_domain: str = ""
    current_phase: str = ""     # "eval" or "train"
    lease_id: str = ""          # Groups multi-GPU leases together
    started_at: str = ""
    total_workouts: int = 0
    total_busy_ms: float = 0.0


@dataclass
class GPULease:
    """A lease representing N GPUs assigned to one agent.

    Think of it like a gym equipment reservation:
    - 1 bench press for warm-up (eval = 1 GPU)
    - Full squat rack section for heavy training (train = 4 GPUs)
    - Other members wait until equipment is free
    """
    lease_id: str
    agent_id: str
    gpu_ids: list           # e.g., [0] for eval, [0,1,2,3] for training
    phase: str              # "eval" or "train"
    created_at: str = ""


@dataclass
class QueueEntry:
    """An agent waiting in the GPU queue."""
    agent_id: str
    gpus_requested: int     # How many GPUs it needs
    phase: str              # "eval" or "train"
    priority: float = 0.0   # Higher = more priority
    queued_at: str = ""


class GymScheduler:
    """GPU resource manager with flexible multi-GPU allocation and queue.

    Key design principles:
    - An agent can request N GPUs (1 for eval, 2-8 for training)
    - If not enough GPUs are free, the agent waits in a priority queue
    - When GPUs become free, queued agents are served in priority order
    - GPU groups are allocated contiguously when possible (NVLink topology)
    - Each allocation is tracked as a "lease" (group of GPUs for one job)

    Example flow:
        Agent A: "I need 1 GPU for eval"     -> gets GPU 0
        Agent B: "I need 4 GPUs for training" -> gets GPUs 1,2,3,4
        Agent C: "I need 4 GPUs for training" -> QUEUED (only 3 free)
        Agent A finishes eval                  -> releases GPU 0
        Agent C: still waiting (needs 4, only 4 free now) -> gets GPUs 0,5,6,7
    """

    def __init__(self, config: AutonomousGymConfig):
        self.config = config
        self._slots: dict[int, GPUSlot] = {}
        self._leases: dict[str, GPULease] = {}  # lease_id -> lease
        self._queue: list[QueueEntry] = []
        self._lock = threading.Lock()
        self._queue_event = threading.Event()  # Signal when GPUs freed
        self._lease_counter = 0

        # Initialize GPU slots
        for gpu_id in config.gpu_ids:
            self._slots[gpu_id] = GPUSlot(gpu_id=gpu_id)

        logger.info(
            f"[GymScheduler] Initialized with {len(self._slots)} GPUs "
            f"(multi-GPU allocation enabled)"
        )

    def _next_lease_id(self) -> str:
        """Generate a unique lease ID."""
        self._lease_counter += 1
        return f"lease_{self._lease_counter:04d}"

    def request_gpus(
        self, agent_id: str, num_gpus: int = 1, phase: str = "eval"
    ) -> Optional[GPULease]:
        """Request N GPUs for an agent.

        Args:
            agent_id: Requesting agent
            num_gpus: Number of GPUs needed
            phase: "eval" (inference) or "train" (SFT/GRPO)

        Returns:
            GPULease if allocation succeeded, None if queued/failed
        """
        with self._lock:
            free_ids = self._get_free_gpu_ids()

            if len(free_ids) >= num_gpus:
                # Allocate! Pick contiguous GPUs when possible
                allocated = self._pick_best_gpus(free_ids, num_gpus)
                lease = self._create_lease(agent_id, allocated, phase)
                logger.info(
                    f"[GymScheduler] {agent_id} ({phase}): "
                    f"allocated GPUs {allocated} (lease: {lease.lease_id})"
                )
                return lease
            else:
                # Not enough GPUs -- don't queue here, let caller decide
                logger.debug(
                    f"[GymScheduler] {agent_id} wants {num_gpus} GPUs "
                    f"but only {len(free_ids)} free"
                )
                return None

    def request_gpu(self, agent_id: str) -> Optional[int]:
        """Request a single GPU (backward-compatible).

        Returns GPU ID if available, None if all busy.
        """
        lease = self.request_gpus(agent_id, num_gpus=1, phase="eval")
        if lease:
            return lease.gpu_ids[0]
        return None

    def enqueue(
        self, agent_id: str, num_gpus: int, phase: str, priority: float = 0.0
    ):
        """Put an agent in the wait queue for GPUs.

        Args:
            agent_id: Agent that needs GPUs
            num_gpus: How many GPUs it needs
            phase: "eval" or "train"
            priority: Higher number = served first
        """
        with self._lock:
            # Don't double-queue
            if any(e.agent_id == agent_id for e in self._queue):
                return

            entry = QueueEntry(
                agent_id=agent_id,
                gpus_requested=num_gpus,
                phase=phase,
                priority=priority,
                queued_at=datetime.now().isoformat(),
            )
            self._queue.append(entry)
            # Sort by priority (highest first)
            self._queue.sort(key=lambda e: -e.priority)

            logger.info(
                f"[GymScheduler] {agent_id} QUEUED: needs {num_gpus} GPUs "
                f"for {phase} (priority={priority:.1f}, "
                f"queue_depth={len(self._queue)})"
            )

    def dequeue(self, agent_id: str):
        """Remove an agent from the wait queue."""
        with self._lock:
            self._queue = [e for e in self._queue if e.agent_id != agent_id]

    def process_queue(self) -> list[tuple[QueueEntry, GPULease]]:
        """Try to fulfill queued requests with newly freed GPUs.

        Called periodically by the main loop after GPUs are released.

        Returns:
            List of (entry, lease) for fulfilled requests
        """
        fulfilled = []
        with self._lock:
            remaining = []
            for entry in self._queue:
                free_ids = self._get_free_gpu_ids()
                if len(free_ids) >= entry.gpus_requested:
                    allocated = self._pick_best_gpus(
                        free_ids, entry.gpus_requested
                    )
                    lease = self._create_lease(
                        entry.agent_id, allocated, entry.phase
                    )
                    fulfilled.append((entry, lease))
                    logger.info(
                        f"[GymScheduler] DEQUEUED {entry.agent_id}: "
                        f"GPUs {allocated} ({entry.phase})"
                    )
                else:
                    remaining.append(entry)
            self._queue = remaining
        return fulfilled

    def release_lease(self, lease_id: str):
        """Release all GPUs in a lease."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return

            for gpu_id in lease.gpu_ids:
                if gpu_id in self._slots:
                    slot = self._slots[gpu_id]
                    if slot.started_at:
                        started = datetime.fromisoformat(slot.started_at)
                        elapsed_ms = (
                            datetime.now() - started
                        ).total_seconds() * 1000
                        slot.total_busy_ms += elapsed_ms
                    slot.total_workouts += 1
                    slot.state = GPUState.FREE
                    slot.current_agent = ""
                    slot.current_domain = ""
                    slot.current_phase = ""
                    slot.lease_id = ""
                    slot.started_at = ""

            logger.info(
                f"[GymScheduler] Lease {lease_id} released: "
                f"GPUs {lease.gpu_ids} (was: {lease.agent_id} / {lease.phase})"
            )

        # Signal that GPUs are available (wake queue processing)
        self._queue_event.set()

    def release_gpu(self, gpu_id: int, agent_id: str = ""):
        """Release a single GPU (backward-compatible)."""
        with self._lock:
            # Find the lease that contains this GPU
            for lease_id, lease in list(self._leases.items()):
                if gpu_id in lease.gpu_ids:
                    break
            else:
                # No lease found, just free the slot directly
                if gpu_id in self._slots:
                    slot = self._slots[gpu_id]
                    if slot.started_at:
                        started = datetime.fromisoformat(slot.started_at)
                        elapsed_ms = (
                            datetime.now() - started
                        ).total_seconds() * 1000
                        slot.total_busy_ms += elapsed_ms
                    slot.total_workouts += 1
                    slot.state = GPUState.FREE
                    slot.current_agent = ""
                    slot.current_domain = ""
                    slot.current_phase = ""
                    slot.lease_id = ""
                    slot.started_at = ""
                    logger.info(
                        f"[GymScheduler] GPU {gpu_id} released "
                        f"(was: {agent_id or 'unknown'})"
                    )
                self._queue_event.set()
                return

        # Release the whole lease
        self.release_lease(lease_id)

    def set_cooldown(self, gpu_id: int, duration_seconds: float):
        """Put a GPU on cooldown (e.g., after repeated failures)."""
        with self._lock:
            if gpu_id in self._slots:
                self._slots[gpu_id].state = GPUState.COOLDOWN
                self._slots[gpu_id].current_agent = ""
                self._slots[gpu_id].current_phase = ""

        def _release_cooldown():
            time.sleep(duration_seconds)
            with self._lock:
                if (
                    gpu_id in self._slots
                    and self._slots[gpu_id].state == GPUState.COOLDOWN
                ):
                    self._slots[gpu_id].state = GPUState.FREE
                    logger.info(
                        f"[GymScheduler] GPU {gpu_id} cooldown ended"
                    )
            self._queue_event.set()

        t = threading.Thread(target=_release_cooldown, daemon=True)
        t.start()

    # ------ Internal helpers ------

    def _get_free_gpu_ids(self) -> list[int]:
        """Get sorted list of free GPU IDs (caller must hold lock)."""
        return sorted(
            gid for gid, s in self._slots.items()
            if s.state == GPUState.FREE
        )

    def _pick_best_gpus(self, free_ids: list[int], n: int) -> list[int]:
        """Pick N GPUs, preferring contiguous blocks for NVLink.

        Strategy: find the longest contiguous run that fits N,
        otherwise just take the first N free.
        """
        if n == 1:
            return [free_ids[0]]

        # Try to find contiguous block
        best_start = -1
        best_len = 0
        start = 0
        for i in range(1, len(free_ids)):
            if free_ids[i] == free_ids[i - 1] + 1:
                if i - start + 1 > best_len:
                    best_start = start
                    best_len = i - start + 1
            else:
                start = i

        # Check last run
        if len(free_ids) - start > best_len:
            best_start = start
            best_len = len(free_ids) - start

        if best_len >= n:
            return free_ids[best_start:best_start + n]

        # No contiguous block, just take first N
        return free_ids[:n]

    def _create_lease(
        self, agent_id: str, gpu_ids: list[int], phase: str
    ) -> GPULease:
        """Create a lease and mark GPUs as busy (caller must hold lock)."""
        lease_id = self._next_lease_id()
        now = datetime.now().isoformat()

        for gid in gpu_ids:
            slot = self._slots[gid]
            slot.state = GPUState.BUSY
            slot.current_agent = agent_id
            slot.current_phase = phase
            slot.lease_id = lease_id
            slot.started_at = now

        lease = GPULease(
            lease_id=lease_id,
            agent_id=agent_id,
            gpu_ids=gpu_ids,
            phase=phase,
            created_at=now,
        )
        self._leases[lease_id] = lease
        return lease

    # ------ Status methods ------

    def get_utilization(self) -> dict:
        """Get current GPU utilization stats."""
        with self._lock:
            states = {}
            for gpu_id, slot in self._slots.items():
                states[gpu_id] = {
                    "state": slot.state.value,
                    "agent": slot.current_agent,
                    "domain": slot.current_domain,
                    "phase": slot.current_phase,
                    "lease": slot.lease_id,
                    "total_workouts": slot.total_workouts,
                }

            free = sum(
                1 for s in self._slots.values()
                if s.state == GPUState.FREE
            )
            busy = sum(
                1 for s in self._slots.values()
                if s.state == GPUState.BUSY
            )
            total = len(self._slots)

            return {
                "total_gpus": total,
                "free": free,
                "busy": busy,
                "utilization": busy / total if total > 0 else 0,
                "active_leases": len(self._leases),
                "queue_depth": len(self._queue),
                "slots": states,
            }

    def get_free_gpu_count(self) -> int:
        """Get number of free GPUs."""
        with self._lock:
            return sum(
                1 for s in self._slots.values()
                if s.state == GPUState.FREE
            )

    def get_queue_depth(self) -> int:
        """Get number of agents waiting in queue."""
        with self._lock:
            return len(self._queue)

    def update_domain(self, gpu_id: int, domain: str):
        """Update the current domain for a GPU slot."""
        with self._lock:
            if gpu_id in self._slots:
                self._slots[gpu_id].current_domain = domain


# ============================================================
# 3. Safety Guardrail
# ============================================================

class SafetyGuardrail:
    """Minimal safety guardrails for the autonomous gym.

    The guardrail intervenes ONLY when:
    1. An agent's safety score drops dangerously low
    2. An agent has too many consecutive failures
    3. A critical safety regression is detected

    It does NOT control training direction -- that's the agent's job.
    """

    def __init__(self, config: AutonomousGymConfig):
        self.config = config
        self._failure_counts: dict = {}
        self._safety_scores: dict = {}

    def check_agent(self, agent_id: str, workout_result: dict) -> dict:
        """Check if an agent should be paused or warned.

        Returns:
            {
                "action": "continue" | "warn" | "cooldown" | "stop",
                "reason": str,
            }
        """
        errors = workout_result.get("errors", [])
        score = workout_result.get("pre_score", 0.0)

        # Track consecutive failures
        if score < 0.3:
            self._failure_counts[agent_id] = (
                self._failure_counts.get(agent_id, 0) + 1
            )
        else:
            self._failure_counts[agent_id] = 0

        # Track safety-specific issues
        safety_violations = sum(
            1 for e in errors if "safety" in str(e).lower()
        )
        if safety_violations:
            prev = self._safety_scores.get(agent_id, 1.0)
            self._safety_scores[agent_id] = max(0, prev - 0.1 * safety_violations)

        # Check guardrails
        if self._safety_scores.get(agent_id, 1.0) < self.config.safety_score_floor:
            return {
                "action": "stop",
                "reason": (
                    f"Safety score {self._safety_scores[agent_id]:.2f} "
                    f"below floor {self.config.safety_score_floor}"
                ),
            }

        failure_count = self._failure_counts.get(agent_id, 0)
        if failure_count >= self.config.max_consecutive_failures:
            self._failure_counts[agent_id] = 0
            return {
                "action": "cooldown",
                "reason": (
                    f"{failure_count} consecutive failures, "
                    f"cooling down for {self.config.cooldown_seconds}s"
                ),
            }

        if failure_count >= 3:
            return {
                "action": "warn",
                "reason": (
                    f"{failure_count} consecutive failures, "
                    f"agent may be struggling"
                ),
            }

        return {"action": "continue", "reason": ""}


# ============================================================
# 4. Agent Worker
# ============================================================

class AgentWorker(threading.Thread):
    """A worker that runs an autonomous agent across one or more GPUs.

    Supports phase-based multi-GPU allocation:
    - Eval phase:  uses `gpus_for_eval` GPUs (default 1)
    - Train phase: uses `gpus_for_train` GPUs (default 1, can be 2-8)

    When training needs more GPUs than eval, the worker:
    1. Runs eval on the initial GPU lease
    2. Requests additional GPUs for training (waits in queue if busy)
    3. Runs training across the expanded GPU set
    4. Releases all GPUs when done

    Other agents wait in the queue and get served when GPUs free up.
    """

    def __init__(
        self,
        agent,
        agent_config,
        lease: "GPULease",
        scheduler: "GymScheduler",
        guardrail: SafetyGuardrail,
        logbook_dir: str,
        result_callback=None,
    ):
        super().__init__(daemon=True)
        self.agent = agent
        self.agent_config = agent_config
        self.lease = lease                   # Initial GPU lease (for eval)
        self.scheduler = scheduler
        self.guardrail = guardrail
        self.logbook_dir = logbook_dir
        self.result_callback = result_callback
        self._result = None

        # GPU requirements from config
        self.gpus_for_eval = getattr(agent_config, "gpus_for_eval", 1)
        self.gpus_for_train = getattr(agent_config, "gpus_for_train", 1)

    @property
    def gpu_ids(self) -> list[int]:
        return self.lease.gpu_ids

    @property
    def gpu_ids_str(self) -> str:
        return ",".join(str(g) for g in self.lease.gpu_ids)

    def run(self):
        """Execute one agent cycle via subprocess for GPU isolation."""
        import subprocess as sp
        import sys

        agent_id = self.agent.agent_id
        train_lease = None

        try:
            # === Phase 1: EVAL (use initial lease) ===
            cuda_str = self.gpu_ids_str
            logger.info(
                f"[AgentWorker] {agent_id} EVAL on "
                f"GPU(s) [{cuda_str}] (lease: {self.lease.lease_id})"
            )

            script = self._build_cycle_script(phase="eval")
            result = self._run_subprocess(script, cuda_str, agent_id, timeout=3600)

            # Check if training is needed and requires more GPUs
            needs_training = result.get("workout", {}).get("success", False)
            needs_more_gpus = self.gpus_for_train > self.gpus_for_eval

            if needs_training and needs_more_gpus:
                # === Phase 2: UPGRADE GPU lease for training ===
                extra_needed = self.gpus_for_train - self.gpus_for_eval
                logger.info(
                    f"[AgentWorker] {agent_id} needs {extra_needed} more "
                    f"GPUs for training (total: {self.gpus_for_train})"
                )

                # Try to get more GPUs, wait if necessary
                train_lease = self._wait_for_gpus(
                    agent_id, self.gpus_for_train, timeout=1800
                )

                if train_lease:
                    # Release eval lease, use training lease
                    self.scheduler.release_lease(self.lease.lease_id)
                    self.lease = train_lease
                    cuda_str = self.gpu_ids_str

                    logger.info(
                        f"[AgentWorker] {agent_id} TRAIN on "
                        f"GPU(s) [{cuda_str}] "
                        f"(lease: {train_lease.lease_id})"
                    )

                    # Run training subprocess
                    train_script = self._build_cycle_script(phase="train")
                    train_result = self._run_subprocess(
                        train_script, cuda_str, agent_id, timeout=7200
                    )

                    # Merge results
                    if train_result.get("workout"):
                        result["workout"]["training_result"] = train_result["workout"]
                else:
                    logger.warning(
                        f"[AgentWorker] {agent_id} couldn't get "
                        f"{self.gpus_for_train} GPUs for training, "
                        f"training on {self.gpus_for_eval} GPU(s) instead"
                    )

            # Safety check
            workout = result.get("workout", {})
            check = self.guardrail.check_agent(agent_id, workout)
            result["safety_check"] = check
            if check["action"] in ("cooldown", "stop"):
                logger.warning(
                    f"[SafetyGuardrail] {agent_id}: "
                    f"{check['action']} -- {check['reason']}"
                )

            self._result = result
            if self.result_callback:
                self.result_callback(agent_id, result)

        except sp.TimeoutExpired:
            logger.error(f"[AgentWorker] {agent_id} timed out")
        except Exception as e:
            logger.error(f"[AgentWorker] {agent_id} crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always release whatever lease we have
            self.scheduler.release_lease(self.lease.lease_id)
            if train_lease and train_lease.lease_id != self.lease.lease_id:
                self.scheduler.release_lease(train_lease.lease_id)

    def _wait_for_gpus(
        self, agent_id: str, num_gpus: int, timeout: int = 1800
    ) -> Optional["GPULease"]:
        """Wait for N GPUs to become available.

        Enqueues with high priority (training > eval) and polls.
        Returns lease when fulfilled, None on timeout.
        """
        # First release current eval GPUs so others can use them
        # while we wait for training GPUs
        self.scheduler.release_lease(self.lease.lease_id)

        # Enqueue with higher priority (training waits have priority)
        self.scheduler.enqueue(
            agent_id, num_gpus, phase="train", priority=10.0
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if queue was fulfilled
            lease = self.scheduler.request_gpus(
                agent_id, num_gpus, phase="train"
            )
            if lease:
                self.scheduler.dequeue(agent_id)
                return lease

            # Wait for a signal that GPUs freed up
            self.scheduler._queue_event.wait(timeout=10)
            self.scheduler._queue_event.clear()

        # Timeout - remove from queue, re-acquire minimal GPUs
        self.scheduler.dequeue(agent_id)
        fallback = self.scheduler.request_gpus(
            agent_id, self.gpus_for_eval, phase="train"
        )
        if fallback:
            self.lease = fallback
        return None

    def _run_subprocess(
        self, script: str, cuda_devices: str,
        agent_id: str, timeout: int = 3600
    ) -> dict:
        """Run a subprocess with given CUDA_VISIBLE_DEVICES."""
        import subprocess as sp
        import sys

        script_path = Path(f"/tmp/gym_worker_{agent_id}_{cuda_devices.replace(',','_')}.py")
        with open(script_path, "w") as f:
            f.write(script)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        proc = sp.run(
            [sys.executable, "-u", str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        result = {"agent_id": agent_id, "gpu_ids": cuda_devices}
        if proc.returncode == 0:
            for line in reversed(proc.stdout.strip().split("\n")):
                if line.startswith("{") and "pre_score" in line:
                    try:
                        result["workout"] = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass

            lines = proc.stdout.strip().split("\n")
            for line in lines[-20:]:
                if line.strip():
                    logger.info(f"  [{agent_id}] {line}")
        else:
            logger.error(
                f"[AgentWorker] {agent_id} exited with "
                f"code {proc.returncode}"
            )
            stderr_tail = proc.stderr[-500:] if proc.stderr else ""
            if stderr_tail:
                logger.error(f"  stderr: {stderr_tail}")
            result["workout"] = {
                "errors": [f"exit_code_{proc.returncode}"],
                "pre_score": 0.0,
            }

        return result

    def _build_cycle_script(self, phase: str = "eval") -> str:
        """Build a Python script to run one agent cycle."""
        cfg = self.agent_config
        return f'''
import sys, json, os
sys.path.insert(0, ".")
# CUDA_VISIBLE_DEVICES is already set in subprocess env by the parent

from bioagents.gym.shared_logbook import SharedLogbook
from bioagents.gym.autonomous_agent import AutonomousAgent, AutonomousAgentConfig

# Step 1: Build config
config = AutonomousAgentConfig(
    agent_id="{cfg.agent_id}",
    model_path="{cfg.model_path}",
    base_model_path="{cfg.base_model_path}",
    backend="{cfg.backend}",
    available_domains={cfg.available_domains},
    curiosity_weight={cfg.curiosity_weight},
    weakness_weight={cfg.weakness_weight},
    peer_learning_weight={cfg.peer_learning_weight},
    diversity_weight={cfg.diversity_weight},
    mastery_push_weight={cfg.mastery_push_weight},
    safety_weight={cfg.safety_weight},
    max_turns={cfg.max_turns},
    eval_tasks_per_domain={cfg.eval_tasks_per_domain},
    training_epochs={cfg.training_epochs},
    learning_rate={cfg.learning_rate},
    quality_threshold={cfg.quality_threshold},
    mastery_threshold={cfg.mastery_threshold},
    gpus_for_eval={cfg.gpus_for_eval},
    gpus_for_train={cfg.gpus_for_train},
    inference_batch_size={cfg.inference_batch_size},
    inference_max_new_tokens={cfg.inference_max_new_tokens},
    inference_max_length={cfg.inference_max_length},
    train_batch_size={cfg.train_batch_size},
    train_max_length={cfg.train_max_length},
    gradient_accumulation_steps={cfg.gradient_accumulation_steps},
    lora_r={cfg.lora_r},
    gpu_memory_utilization={cfg.gpu_memory_utilization},
    benchmark_every_n_cycles={cfg.benchmark_every_n_cycles},
    benchmark_max_samples={cfg.benchmark_max_samples},
    output_dir="{cfg.output_dir}",
    log_dir="{cfg.log_dir}",
)

# Step 2: Profile the model (auto-detect capabilities + repair)
from bioagents.gym.model_profile import ModelProfiler
profile = ModelProfiler.profile_and_repair(
    model_path=config.model_path,
    base_model_path=config.base_model_path,
    available_domains=config.available_domains,
)

# Step 2b: Auto-tune params if not already set by parent
optimal = profile.compute_optimal_params(
    gpu_memory_gb=80.0,
    num_gpus={cfg.gpus_for_eval},
)
config.apply_optimal_params(optimal)

print(profile.summary())

if not profile.is_valid:
    print(json.dumps({{"errors": profile.validation_errors, "pre_score": 0.0}}, default=str))
    sys.exit(1)

# Auto-filter domains to compatible ones
config.available_domains = profile.get_compatible_domains(config.available_domains)
config._model_profile = profile

# Step 3: Run one cycle
logbook = SharedLogbook("{self.logbook_dir}")
agent = AutonomousAgent(config, logbook)
result = agent.run_one_cycle(gpu_id=-1)  # GPU binding handled by CUDA_VISIBLE_DEVICES env var

# Output result as JSON for parent to parse
print(json.dumps(result.get("workout", {{}}), default=str))
'''


# ============================================================
# 5. AutonomousGym (Main Orchestrator)
# ============================================================

class AutonomousGym:
    """The Autonomous GYM -- a shared space where agents self-improve.

    The gym provides:
    - GPU resources (via GymScheduler)
    - Safety guardrails (minimal intervention)
    - Shared logbook (cross-agent learning)
    - Monitoring and dashboards

    The gym does NOT:
    - Tell agents what to train on
    - Control training strategies
    - Set curriculum (agents decide themselves)

    Usage:
        gym = AutonomousGym(config)
        gym.register_agent(agent_config_1)
        gym.register_agent(agent_config_2)
        gym.open()   # Blocks until stopped
    """

    def __init__(self, config: AutonomousGymConfig):
        self.config = config

        # Core components
        from bioagents.gym.shared_logbook import SharedLogbook
        self.logbook = SharedLogbook(config.logbook_dir)
        self.scheduler = GymScheduler(config)
        self.guardrail = SafetyGuardrail(config)

        # Agent registry
        self._agents: dict = {}
        self._agent_configs: dict = {}
        self._active_workers: list = []
        self._results: list = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("  Autonomous Healthcare AI GYM")
        logger.info("=" * 70)
        logger.info(f"  GPUs: {config.num_gpus}")
        logger.info(f"  Domains: {len(config.available_domains)}")
        logger.info(
            f"  Mode: "
            f"{'CONTINUOUS' if config.continuous else 'FIXED'}"
        )

    def register_agent(self, agent_config) -> str:
        """Register an agent with the gym.

        Before accepting an agent, the gym:
        1. Profiles the model (auto-detects capabilities)
        2. Repairs missing files (e.g., processor configs for VL models)
        3. Filters available domains to compatible ones
        4. Validates the model can actually be loaded
        5. Stores the profile for use during training

        Args:
            agent_config: AutonomousAgentConfig

        Returns:
            agent_id
        """
        from bioagents.gym.autonomous_agent import AutonomousAgent
        from bioagents.gym.model_profile import ModelProfiler

        # Step 1: Profile the model
        profile = ModelProfiler.profile_and_repair(
            model_path=agent_config.model_path,
            base_model_path=agent_config.base_model_path,
            available_domains=self.config.available_domains,
        )

        logger.info(f"\n{profile.summary()}")

        if not profile.is_valid:
            logger.error(
                f"[GYM] REJECTED agent {agent_config.agent_id}: "
                f"{profile.validation_errors}"
            )
            return ""

        # Step 2: Auto-filter domains to compatible ones
        compatible_domains = profile.get_compatible_domains(
            agent_config.available_domains
        )
        if not compatible_domains:
            logger.error(
                f"[GYM] REJECTED agent {agent_config.agent_id}: "
                f"no compatible domains for {profile.architecture}"
            )
            return ""

        # Update agent config with filtered domains
        agent_config.available_domains = compatible_domains
        agent_config._model_profile = profile

        # Step 3: Auto-tune batch sizes & sequence lengths based on GPU
        gpu_mem_gb = self._detect_gpu_memory_gb()
        num_gpus_eval = getattr(agent_config, "gpus_for_eval", 1)
        optimal = profile.compute_optimal_params(
            gpu_memory_gb=gpu_mem_gb,
            num_gpus=num_gpus_eval,
        )
        agent_config.apply_optimal_params(optimal)

        # Step 4: Register
        agent = AutonomousAgent(agent_config, self.logbook)
        self._agents[agent_config.agent_id] = agent
        self._agent_configs[agent_config.agent_id] = agent_config

        modalities = ", ".join(profile.modalities)
        op = profile.optimal_params
        logger.info(
            f"[GYM] Registered agent: {agent_config.agent_id}\n"
            f"  Model:      {profile.model_name} ({profile.architecture})\n"
            f"  Modalities: {modalities}\n"
            f"  Domains:    {len(compatible_domains)}/{len(self.config.available_domains)} "
            f"({', '.join(compatible_domains)})\n"
            f"  Memory:     ~{profile.estimated_memory_gb:.0f} GB\n"
            f"  Class:      {profile.model_class}\n"
            f"  --- Auto-Tuned Parameters ---\n"
            f"  Inf batch:  {op.get('inference_batch_size')}  |  "
            f"Inf ctx: {op.get('inference_max_length')}  |  "
            f"max_new: {op.get('inference_max_new_tokens')}\n"
            f"  Train batch:{op.get('train_batch_size')}  |  "
            f"Train ctx:{op.get('train_max_length')}  |  "
            f"grad_accum: {op.get('gradient_accumulation_steps')}\n"
            f"  LoRA r:     {op.get('lora_r')}  |  "
            f"GPU util: {op.get('gpu_memory_utilization')}  |  "
            f"Free VRAM: {op.get('_free_after_model_gb')} GB"
        )
        return agent_config.agent_id

    @staticmethod
    def _detect_gpu_memory_gb() -> float:
        """Detect per-GPU memory in GB.  Defaults to 80 (A100-80G)."""
        try:
            import torch
            if torch.cuda.is_available():
                # Use GPU 0 as reference (assumes homogeneous cluster)
                total_bytes = torch.cuda.get_device_properties(0).total_mem
                return total_bytes / (1024 ** 3)
        except Exception:
            pass
        # Fallback: try nvidia-smi
        try:
            import subprocess as sp
            out = sp.check_output(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            mib = int(out.strip().split("\n")[0])
            return mib / 1024
        except Exception:
            pass
        return 80.0  # default A100-80G

    def open(self):
        """Open the gym -- agents start training continuously.

        This is the main loop that:
        1. Checks for free GPUs
        2. Assigns waiting agents to free GPUs
        3. Monitors active workers
        4. Collects results
        5. Prints dashboards periodically
        """
        logger.info("\n[GYM] DOORS OPEN! Agents are welcome.")
        logger.info(
            f"[GYM] {len(self._agents)} agents registered"
        )

        # Handle graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)

        def _signal_handler(signum, frame):
            logger.info("\n[GYM] Closing time! Finishing active workouts...")
            self._stop_event.set()
            signal.signal(signal.SIGINT, original_sigint)

        signal.signal(signal.SIGINT, _signal_handler)

        cycle = 0
        dashboard_interval = 10  # Print dashboard every N cycles

        try:
            while not self._stop_event.is_set():
                cycle += 1

                # Clean up finished workers
                self._cleanup_workers()

                # Assign agents to free GPUs
                self._assign_agents()

                # Periodic dashboard
                if cycle % dashboard_interval == 0:
                    self._print_status()

                # Wait before next check
                self._stop_event.wait(
                    timeout=self.config.idle_check_interval
                )

        except Exception as e:
            logger.error(f"[GYM] Error: {e}")
            import traceback
            traceback.print_exc()

        # Wait for active workers to finish
        logger.info(
            "[GYM] Waiting for active workouts to finish..."
        )
        for worker in self._active_workers:
            worker.join(timeout=300)  # 5 min timeout

        # Final summary
        self._print_final_summary()
        logger.info("[GYM] Doors closed.")

    def _assign_agents(self):
        """Assign waiting agents to free GPUs.

        Multi-GPU aware: each agent requests gpus_for_eval GPUs.
        If not enough GPUs are free, the agent waits.
        """
        # First: process the queue (fulfill waiting training requests)
        fulfilled = self.scheduler.process_queue()
        for entry, lease in fulfilled:
            agent_id = entry.agent_id
            if agent_id in self._agents:
                # A queued training request was fulfilled.
                # The AgentWorker._wait_for_gpus handles this via polling,
                # so we just need to signal.
                self.scheduler._queue_event.set()

        free_gpus = self.scheduler.get_free_gpu_count()
        if free_gpus == 0:
            return

        # Round-robin through agents
        for agent_id, agent in self._agents.items():
            if free_gpus <= 0:
                break

            # Skip agents that are already running
            if any(
                w.agent.agent_id == agent_id and w.is_alive()
                for w in self._active_workers
            ):
                continue

            # Skip agents on cooldown/stopped by guardrail
            check = self.guardrail.check_agent(agent_id, {})
            if check["action"] == "stop":
                continue

            # Determine how many GPUs this agent needs for eval
            cfg = self._agent_configs[agent_id]
            gpus_needed = getattr(cfg, "gpus_for_eval", 1)

            if free_gpus < gpus_needed:
                # Not enough for this agent, try next (smaller) one
                continue

            # Request GPU lease
            lease = self.scheduler.request_gpus(
                agent_id, num_gpus=gpus_needed, phase="eval"
            )
            if lease is None:
                continue

            # Start worker
            worker = AgentWorker(
                agent=agent,
                agent_config=cfg,
                lease=lease,
                scheduler=self.scheduler,
                guardrail=self.guardrail,
                logbook_dir=self.config.logbook_dir,
                result_callback=self._on_result,
            )
            worker.start()
            self._active_workers.append(worker)
            free_gpus -= gpus_needed

            gpu_str = ",".join(str(g) for g in lease.gpu_ids)
            logger.info(
                f"[GYM] {agent_id} -> GPU [{gpu_str}] "
                f"(eval, {gpus_needed} GPU(s))"
            )

    def _cleanup_workers(self):
        """Remove finished workers from the active list."""
        self._active_workers = [
            w for w in self._active_workers if w.is_alive()
        ]

    def _on_result(self, agent_id: str, result: dict):
        """Callback when an agent finishes a cycle."""
        with self._lock:
            self._results.append({
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "result": result,
            })

        # Handle safety guardrail actions
        safety = result.get("safety_check", {})
        if safety.get("action") == "cooldown":
            # Find the agent's last GPU and cool it down
            logger.warning(
                f"[GYM] Cooling down {agent_id}: "
                f"{safety['reason']}"
            )

    def _print_status(self):
        """Print current gym status."""
        util = self.scheduler.get_utilization()

        print()
        print("-" * 70)
        print(
            f"  GYM Status | "
            f"GPUs: {util['busy']}/{util['total_gpus']} busy "
            f"({util['utilization']:.0%}) | "
            f"Workers: {len(self._active_workers)} active | "
            f"Queue: {util.get('queue_depth', 0)} waiting"
        )
        print("-" * 70)

        for gpu_id, slot_info in util["slots"].items():
            state = slot_info["state"]
            agent = slot_info["agent"] or "-"
            domain = slot_info["domain"] or "-"
            phase = slot_info.get("phase", "-") or "-"
            lease = slot_info.get("lease", "") or ""
            lease_tag = f" [{lease}]" if lease else ""
            print(
                f"  GPU {gpu_id}: [{state:>8}] "
                f"{phase:<6} agent={agent:<25} domain={domain}{lease_tag}"
            )

        # Queue status
        if util.get("queue_depth", 0) > 0:
            print(f"  QUEUE: {util['queue_depth']} agent(s) waiting for GPUs")

        # Show auto-tuned params summary (compact, one line per agent)
        shown_agents = set()
        for w in self._active_workers:
            aid = getattr(w, "agent_id", "")
            if aid and aid not in shown_agents:
                shown_agents.add(aid)
                cfg = self._agent_configs.get(aid)
                if cfg:
                    tb = getattr(cfg, "train_batch_size", "?")
                    ib = getattr(cfg, "inference_batch_size", "?")
                    tc = getattr(cfg, "train_max_length", "?")
                    ic = getattr(cfg, "inference_max_length", "?")
                    ga = getattr(cfg, "gradient_accumulation_steps", "?")
                    lr = getattr(cfg, "lora_r", "?")
                    print(
                        f"    {aid[:22]:<22} "
                        f"inf_b={ib:<3} inf_ctx={ic:<5} "
                        f"trn_b={tb:<3} trn_ctx={tc:<5} "
                        f"ga={ga:<2} lora_r={lr}"
                    )

        # Check herding
        herding = self.logbook.detect_herding()
        if herding.get("herding_detected"):
            print(
                f"  WARNING: {herding['recommendation']}"
            )

        print("-" * 70)

    def _print_final_summary(self):
        """Print final gym summary."""
        print()
        print("=" * 70)
        print("  Autonomous GYM -- Session Summary")
        print("=" * 70)

        # Utilization
        util = self.scheduler.get_utilization()
        total_workouts = sum(
            s["total_workouts"] for s in util["slots"].values()
        )
        print(f"  Total workouts completed: {total_workouts}")

        # Per-agent summary
        print()
        for agent_id, agent in self._agents.items():
            status = agent.get_status()
            profile = self.logbook.get_agent_profile(agent_id)
            if profile:
                print(
                    f"  {agent_id}: "
                    f"cycles={status['total_cycles']} "
                    f"avg={profile.avg_score:.1%} "
                    f"domains={profile.total_domains_visited}"
                )
            else:
                print(
                    f"  {agent_id}: "
                    f"cycles={status['total_cycles']} (no profile)"
                )

        # Logbook dashboard
        self.logbook.print_gym_dashboard()

        # Save session log
        session_path = (
            self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(session_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "num_gpus": self.config.num_gpus,
                    "domains": self.config.available_domains,
                },
                "utilization": util,
                "total_results": len(self._results),
                "gym_summary": self.logbook.get_gym_summary(),
            }, f, indent=2, ensure_ascii=False)

        print(f"\n  Session log saved to {session_path}")
        print("=" * 70)

    def close(self):
        """Gracefully close the gym."""
        self._stop_event.set()

    # -- Convenience Methods --

    def get_gym_status(self) -> dict:
        """Get full gym status."""
        return {
            "utilization": self.scheduler.get_utilization(),
            "agents": {
                aid: a.get_status() for aid, a in self._agents.items()
            },
            "logbook": self.logbook.get_gym_summary(),
            "herding": self.logbook.detect_herding(),
        }


# ============================================================
# 6. CLI Entry Point
# ============================================================

def run_autonomous_gym(config_path: Optional[str] = None, **kwargs):
    """Entry point for the Autonomous GYM."""
    from bioagents.gym.autonomous_agent import AutonomousAgentConfig

    if config_path:
        import yaml
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)

        gym_config = AutonomousGymConfig(**cfg_dict.get("gym", {}))
        gym = AutonomousGym(gym_config)

        # Register agents from config
        for agent_cfg in cfg_dict.get("agents", []):
            agent_config = AutonomousAgentConfig(
                available_domains=gym_config.available_domains,
                **agent_cfg,
            )
            gym.register_agent(agent_config)

    else:
        gym_config = AutonomousGymConfig(**kwargs)
        gym = AutonomousGym(gym_config)

    gym.open()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Healthcare AI GYM"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config"
    )
    args = parser.parse_args()

    run_autonomous_gym(config_path=args.config)
