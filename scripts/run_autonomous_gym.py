#!/usr/bin/env python3
"""Autonomous GYM Launcher -- Designed for 30+ day continuous operation.

Features:
- Auto-restart on crash (with exponential backoff)
- Disk space monitoring (pause if < 50GB free)
- Periodic checkpoint cleanup (keep last N per agent)
- Heartbeat logging for monitoring
- Graceful SIGTERM/SIGINT handling
- GPU health monitoring

Usage:
    # Standard launch (runs forever):
    python scripts/run_autonomous_gym.py --config configs/autonomous_gym.yaml

    # With nohup for terminal disconnect survival:
    nohup python -u scripts/run_autonomous_gym.py --config configs/autonomous_gym.yaml \
        > logs/autonomous_gym/launcher.log 2>&1 &

    # With screen:
    screen -dmS gym python -u scripts/run_autonomous_gym.py --config configs/autonomous_gym.yaml
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Configuration
# ============================================================

MAX_RESTARTS = 999          # Max restarts before giving up entirely
INITIAL_BACKOFF_S = 30      # Initial wait after crash
MAX_BACKOFF_S = 600         # Max wait between restarts (10 min)
HEARTBEAT_INTERVAL_S = 300  # Log heartbeat every 5 min
DISK_CHECK_INTERVAL_S = 3600  # Check disk every hour
MIN_DISK_GB = 50            # Pause if disk < 50GB
MAX_CHECKPOINTS_PER_AGENT = 5  # Keep last N checkpoints per agent
CHECKPOINT_CLEANUP_INTERVAL_S = 7200  # Clean old checkpoints every 2 hours


def get_disk_free_gb(path: str = "/data") -> float:
    """Get free disk space in GB."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024 ** 3)
    except Exception:
        return 999.0  # Assume plenty if can't check


def get_gpu_status() -> dict:
    """Get GPU utilization and health."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus[int(parts[0])] = {
                    "util": int(parts[1]),
                    "mem_used": int(parts[2]),
                    "mem_total": int(parts[3]),
                    "temp": int(parts[4]),
                }
        return gpus
    except Exception:
        return {}


def cleanup_old_checkpoints(base_dir: str, max_keep: int = 5):
    """Remove old checkpoints, keeping only the most recent N per agent."""
    base = Path(base_dir)
    if not base.exists():
        return

    for agent_dir in base.iterdir():
        if not agent_dir.is_dir():
            continue

        # Find workout directories sorted by modification time
        workout_dirs = sorted(
            [d for d in agent_dir.iterdir() if d.is_dir() and d.name.startswith("workout_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        # Remove old ones
        for old_dir in workout_dirs[max_keep:]:
            try:
                shutil.rmtree(old_dir)
                log(f"Cleaned up old checkpoint: {old_dir}")
            except Exception as e:
                log(f"Failed to clean {old_dir}: {e}")


def log(msg: str):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [LAUNCHER] {msg}", flush=True)


def run_gym(config_path: str):
    """Run the Autonomous GYM with crash recovery."""
    import yaml

    log("=" * 70)
    log("  Autonomous Healthcare AI GYM -- 30-Day Continuous Launcher")
    log("=" * 70)
    log(f"  Config: {config_path}")
    log(f"  Start time: {datetime.now().isoformat()}")
    log(f"  Target end: {(datetime.now() + timedelta(days=30)).isoformat()}")
    log(f"  PID: {os.getpid()}")

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gym_cfg = cfg.get("gym", {})
    agents_cfg = cfg.get("agents", [])

    log(f"  GPUs: {gym_cfg.get('num_gpus', 8)}")
    log(f"  Agents: {len(agents_cfg)}")
    for a in agents_cfg:
        log(f"    - {a['agent_id']}: {a['model_path']}")
    log("=" * 70)

    # Ensure log directories
    log_dir = Path(gym_cfg.get("log_dir", "logs/autonomous_gym"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Track restarts
    restart_count = 0
    backoff_s = INITIAL_BACKOFF_S
    start_time = time.time()
    last_heartbeat = time.time()
    last_disk_check = time.time()
    last_checkpoint_cleanup = time.time()

    # Graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        log(f"Received signal {signum}, requesting graceful shutdown...")
        shutdown_requested = True

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    while not shutdown_requested and restart_count < MAX_RESTARTS:
        try:
            # Pre-flight checks
            disk_gb = get_disk_free_gb("/data")
            log(f"Disk free: {disk_gb:.1f} GB")
            if disk_gb < MIN_DISK_GB:
                log(f"LOW DISK! {disk_gb:.1f} GB < {MIN_DISK_GB} GB. Cleaning checkpoints...")
                for a in agents_cfg:
                    cleanup_old_checkpoints(
                        a.get("output_dir", "checkpoints/autonomous"),
                        max_keep=2,
                    )
                disk_gb = get_disk_free_gb("/data")
                if disk_gb < MIN_DISK_GB / 2:
                    log(f"CRITICAL: Still low disk ({disk_gb:.1f} GB). Waiting 30min...")
                    time.sleep(1800)
                    continue

            gpu_status = get_gpu_status()
            if gpu_status:
                gpu_summary = {}
                for k, v in gpu_status.items():
                    temp = v["temp"]
                    mem_used = v["mem_used"]
                    mem_total = v["mem_total"]
                    gpu_summary[k] = f"{temp}C {mem_used}/{mem_total}MB"
                log(f"GPU status: {json.dumps(gpu_summary)}")

            # Launch the gym
            if restart_count > 0:
                log(f"Restart #{restart_count} (backoff was {backoff_s}s)")
            else:
                log("Starting Autonomous GYM...")

            # Import and run
            from bioagents.gym.autonomous_gym import AutonomousGym, AutonomousGymConfig
            from bioagents.gym.autonomous_agent import AutonomousAgentConfig

            gym_config = AutonomousGymConfig(**gym_cfg)
            gym = AutonomousGym(gym_config)

            # Register all agents
            # Inherit benchmark settings from top-level config
            bench_cfg = cfg.get("benchmark", {})
            benchmark_every = bench_cfg.get("every_n_cycles", 3)
            benchmark_samples = bench_cfg.get("max_samples", 0)

            for agent_cfg in agents_cfg:
                # Inject benchmark config from gym level
                if "benchmark_every_n_cycles" not in agent_cfg:
                    agent_cfg["benchmark_every_n_cycles"] = benchmark_every
                if "benchmark_max_samples" not in agent_cfg:
                    agent_cfg["benchmark_max_samples"] = benchmark_samples

                agent_config = AutonomousAgentConfig(
                    available_domains=gym_config.available_domains,
                    **agent_cfg,
                )
                gym.register_agent(agent_config)

            # Run! This blocks until gym.close() or error
            gym.open()

            # If we get here, gym closed gracefully
            if shutdown_requested:
                log("Gym closed by shutdown request.")
                break
            else:
                log("Gym closed unexpectedly but gracefully.")
                backoff_s = INITIAL_BACKOFF_S  # Reset backoff

        except KeyboardInterrupt:
            log("Keyboard interrupt received.")
            break

        except Exception as e:
            restart_count += 1
            log(f"GYM CRASHED: {e}")
            log(traceback.format_exc())

            # Save crash report
            crash_path = log_dir / f"crash_{restart_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(crash_path, "w") as f:
                json.dump({
                    "restart_count": restart_count,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": time.time() - start_time,
                    "disk_free_gb": get_disk_free_gb("/data"),
                    "gpu_status": get_gpu_status(),
                }, f, indent=2)

            log(f"Crash report saved to {crash_path}")
            log(f"Waiting {backoff_s}s before restart...")
            time.sleep(backoff_s)

            # Exponential backoff
            backoff_s = min(backoff_s * 2, MAX_BACKOFF_S)

        # Periodic maintenance between cycles
        now = time.time()

        # Heartbeat
        if now - last_heartbeat > HEARTBEAT_INTERVAL_S:
            uptime = timedelta(seconds=int(now - start_time))
            log(f"HEARTBEAT: uptime={uptime}, restarts={restart_count}, disk={get_disk_free_gb('/data'):.1f}GB")
            last_heartbeat = now

        # Checkpoint cleanup
        if now - last_checkpoint_cleanup > CHECKPOINT_CLEANUP_INTERVAL_S:
            log("Running checkpoint cleanup...")
            for a in agents_cfg:
                cleanup_old_checkpoints(
                    a.get("output_dir", "checkpoints/autonomous"),
                    max_keep=MAX_CHECKPOINTS_PER_AGENT,
                )
            last_checkpoint_cleanup = now

    # Final summary
    total_uptime = timedelta(seconds=int(time.time() - start_time))
    log("=" * 70)
    log("  Autonomous GYM -- Session Complete")
    log(f"  Total uptime: {total_uptime}")
    log(f"  Total restarts: {restart_count}")
    log(f"  Reason: {'Shutdown requested' if shutdown_requested else f'Max restarts ({MAX_RESTARTS})'}")
    log("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous GYM Launcher")
    parser.add_argument(
        "--config",
        default="configs/autonomous_gym.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))
    run_gym(args.config)
