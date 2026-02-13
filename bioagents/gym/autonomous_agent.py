"""AutonomousAgent -- Self-Aware Agent for the Autonomous GYM.

Each agent has intrinsic motivation to self-improve:
1. REFLECT: Analyze own recent records to identify weaknesses
2. CHOOSE: Decide which station/domain to train at next
3. TRAIN: Execute a workout (evaluate + learn)
4. RECORD: Log results to the SharedLogbook
5. REPEAT

The key difference from GymCoach-driven training:
- GymCoach tells the model what to do (top-down)
- AutonomousAgent decides for itself (bottom-up, intrinsic motivation)

Architecture:
    AutonomousAgent
    |-- SelfAwareness      (reflect on own performance)
    |-- StrategySelector    (choose what to do next)
    |-- WorkoutExecutor     (execute the chosen workout)
    |-- SharedLogbook       (read/write shared records)

Usage:
    agent = AutonomousAgent(
        agent_id="qwen3_8b_v1",
        model_path="checkpoints/qwen3_8b_sft",
        logbook=shared_logbook,
        available_domains=["clinical_diagnosis", "drug_interaction", ...],
    )

    # The agent runs autonomously
    while True:
        agent.run_one_cycle()
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class AgentState(Enum):
    """Current state of an autonomous agent."""
    IDLE = "idle"
    REFLECTING = "reflecting"
    CHOOSING = "choosing"
    TRAINING = "training"
    RECORDING = "recording"
    WAITING_FOR_GPU = "waiting_for_gpu"


class Motivation(Enum):
    """What motivates the agent's next action."""
    CURIOSITY = "curiosity"           # Explore a new/untried domain
    WEAKNESS_FIXING = "weakness"      # Fix a known weakness
    PEER_LEARNING = "peer_learning"   # Learn from a peer's strength
    DIVERSITY = "diversity"           # Anti-herding, explore least-visited
    MASTERY_PUSH = "mastery_push"     # Push a near-conquered domain over the line
    SAFETY_FOCUS = "safety"           # Address safety concerns


@dataclass
class AgentDecision:
    """A decision made by the autonomous agent."""
    domain: str
    motivation: str
    confidence: float = 0.5
    reasoning: str = ""
    priority_score: float = 0.0
    training_method: str = "sft"
    estimated_difficulty: str = "moderate"
    timestamp: str = ""


@dataclass
class AutonomousAgentConfig:
    """Configuration for an autonomous agent."""
    agent_id: str
    model_path: str

    # Base model path (for merged VL checkpoints that need processor files)
    base_model_path: str = ""

    # Available domains to train on (auto-filtered by ModelProfile if available)
    available_domains: list = field(default_factory=lambda: [
        "clinical_diagnosis", "drug_interaction", "ehr_management",
        "medical_qa", "triage_emergency", "cross_domain",
        "psychiatry", "obstetrics",
    ])

    # Strategy weights (how much each motivation factor matters)
    curiosity_weight: float = 0.15      # Explore new domains
    weakness_weight: float = 0.35       # Fix weaknesses
    peer_learning_weight: float = 0.20  # Learn from peers
    diversity_weight: float = 0.15      # Anti-herding
    mastery_push_weight: float = 0.10   # Push near-conquered domains
    safety_weight: float = 0.05         # Safety priority boost

    # Training config
    backend: str = "transformers"
    max_turns: int = 15
    eval_tasks_per_domain: int = 20
    training_epochs: int = 2
    learning_rate: float = 2e-5
    quality_threshold: float = 0.5

    # Reflection config
    reflection_window: int = 20         # How many past workouts to reflect on
    mastery_threshold: float = 0.90

    # GPU requirements (multi-GPU support)
    gpus_for_eval: int = 1              # GPUs needed for evaluation
    gpus_for_train: int = 1             # GPUs needed for training (SFT/GRPO)

    # --- GPU-Aware Batch / Sequence Tuning ---
    # These can be set manually in YAML, or auto-populated by
    # ModelProfile.compute_optimal_params() during GYM registration.
    # "auto" or 0 means: let the model profile decide at runtime.
    inference_batch_size: int = 0       # 0 = auto-tune
    inference_max_new_tokens: int = 0   # 0 = auto-tune
    inference_max_length: int = 0       # 0 = auto-tune
    train_batch_size: int = 0           # 0 = auto-tune
    train_max_length: int = 0           # 0 = auto-tune
    gradient_accumulation_steps: int = 0  # 0 = auto-tune
    lora_r: int = 0                     # 0 = auto-tune
    gpu_memory_utilization: float = 0.0  # 0.0 = auto-tune

    # Paths
    output_dir: str = "checkpoints/autonomous"
    log_dir: str = "logs/autonomous"

    # Model profile (auto-populated by GYM on registration)
    _model_profile: object = field(default=None, repr=False)

    def apply_optimal_params(self, optimal: dict):
        """Apply auto-tuned parameters, respecting manual overrides.

        If a field is set to its zero/default value (meaning "auto"),
        the auto-tuned value is used.  If the user has set a specific
        value in the YAML, that takes priority.
        """
        mapping = {
            "inference_batch_size": "inference_batch_size",
            "inference_max_new_tokens": "inference_max_new_tokens",
            "inference_max_length": "inference_max_length",
            "train_batch_size": "train_batch_size",
            "train_max_length": "train_max_length",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "lora_r": "lora_r",
            "gpu_memory_utilization": "gpu_memory_utilization",
        }
        for opt_key, cfg_key in mapping.items():
            current = getattr(self, cfg_key)
            auto_val = optimal.get(opt_key)
            if auto_val is not None:
                # Only override if current value is zero/"auto"
                if isinstance(current, float):
                    if current <= 0.0:
                        setattr(self, cfg_key, auto_val)
                elif isinstance(current, int):
                    if current <= 0:
                        setattr(self, cfg_key, auto_val)
        logger.info(
            f"[{self.agent_id}] Tuned params: "
            f"inf_batch={self.inference_batch_size}, "
            f"inf_ctx={self.inference_max_length}, "
            f"train_batch={self.train_batch_size}, "
            f"train_ctx={self.train_max_length}, "
            f"grad_accum={self.gradient_accumulation_steps}, "
            f"lora_r={self.lora_r}, "
            f"gpu_util={self.gpu_memory_utilization}"
        )


# ============================================================
# 1. Self-Awareness Module
# ============================================================

class SelfAwareness:
    """Agent's ability to understand its own capabilities and limitations.

    Analyzes the agent's workout history from the SharedLogbook to:
    1. Identify strengths and weaknesses by domain
    2. Detect learning plateaus
    3. Find recurring error patterns
    4. Estimate confidence per domain
    """

    def __init__(self, agent_id: str, logbook):
        self.agent_id = agent_id
        self.logbook = logbook

    def reflect(self) -> dict:
        """Perform self-reflection by analyzing own records.

        Returns:
            {
                "strengths": [domains where I excel],
                "weaknesses": [domains where I struggle],
                "plateaus": [domains where I'm stuck],
                "untried": [domains I haven't tried],
                "error_patterns": {error_type: count},
                "overall_score": float,
                "confidence_map": {domain: confidence},
                "improvement_rate": float,
            }
        """
        profile = self.logbook.get_agent_profile(self.agent_id)
        if not profile:
            return {
                "strengths": [],
                "weaknesses": [],
                "plateaus": [],
                "untried": [],
                "error_patterns": {},
                "overall_score": 0.0,
                "confidence_map": {},
                "improvement_rate": 0.0,
                "is_new": True,
            }

        # Get recent workout history
        recent_workouts = self.logbook.get_agent_workout_history(
            self.agent_id, last_n=50
        )

        # Detect plateaus
        plateaus = self._detect_plateaus(recent_workouts)

        # Compute confidence map
        confidence_map = self._compute_confidence(profile, recent_workouts)

        # Compute improvement rate
        improvement_rate = self._compute_improvement_rate(recent_workouts)

        return {
            "strengths": profile.strengths,
            "weaknesses": profile.weaknesses,
            "plateaus": plateaus,
            "untried": self._find_untried_domains(profile),
            "error_patterns": dict(profile.recurring_errors),
            "overall_score": profile.avg_score,
            "confidence_map": confidence_map,
            "improvement_rate": improvement_rate,
            "domain_scores": dict(profile.domain_scores),
            "is_new": False,
        }

    def _detect_plateaus(self, workouts: list, window: int = 5) -> list:
        """Detect domains where scores haven't improved recently."""
        domain_history = {}
        for w in workouts:
            domain = w.get("domain", "")
            if domain not in domain_history:
                domain_history[domain] = []
            domain_history[domain].append(w.get("action_score", 0))

        plateaus = []
        for domain, scores in domain_history.items():
            if len(scores) >= window:
                recent = scores[-window:]
                score_range = max(recent) - min(recent)
                if score_range < 0.03 and recent[-1] < 0.90:
                    plateaus.append(domain)

        return plateaus

    def _compute_confidence(self, profile, workouts: list) -> dict:
        """Compute per-domain confidence based on consistency and recency."""
        confidence = {}
        for domain, score in profile.domain_scores.items():
            domain_workouts = [
                w for w in workouts if w.get("domain") == domain
            ]
            if not domain_workouts:
                confidence[domain] = 0.3
                continue

            # Confidence = f(score, consistency, recency)
            scores = [w.get("action_score", 0) for w in domain_workouts[-5:]]
            if len(scores) >= 2:
                variance = sum((s - score) ** 2 for s in scores) / len(scores)
                consistency = max(0, 1.0 - variance * 10)
            else:
                consistency = 0.5

            confidence[domain] = min(1.0, score * 0.5 + consistency * 0.5)

        return confidence

    def _compute_improvement_rate(self, workouts: list) -> float:
        """Compute overall improvement rate over recent workouts."""
        if len(workouts) < 4:
            return 0.0

        earlier = workouts[:len(workouts) // 2]
        later = workouts[len(workouts) // 2:]

        earlier_avg = sum(w.get("action_score", 0) for w in earlier) / len(earlier)
        later_avg = sum(w.get("action_score", 0) for w in later) / len(later)

        return later_avg - earlier_avg

    def _find_untried_domains(self, profile) -> list:
        """Find domains the agent hasn't attempted yet."""
        # We'll check against the logbook's known domains
        all_agents = self.logbook.get_registered_agents()
        all_domains = set()
        for aid in all_agents:
            other_profile = self.logbook.get_agent_profile(aid)
            if other_profile:
                all_domains.update(other_profile.domain_scores.keys())

        tried = set(profile.domain_scores.keys())
        return sorted(all_domains - tried)


# ============================================================
# 2. Strategy Selector
# ============================================================

class StrategySelector:
    """Decides what the agent should do next based on self-reflection.

    Scoring function for each candidate domain:
        score = (
            weakness_weight * weakness_score
            + curiosity_weight * curiosity_score
            + peer_learning_weight * peer_score
            + diversity_weight * diversity_score
            + mastery_push_weight * mastery_push_score
            + safety_weight * safety_score
        )
    """

    def __init__(self, config: AutonomousAgentConfig, logbook):
        self.config = config
        self.logbook = logbook

    def choose_next_action(self, reflection: dict) -> AgentDecision:
        """Choose the next domain and training strategy.

        Args:
            reflection: Output from SelfAwareness.reflect()

        Returns:
            AgentDecision with domain, motivation, and strategy
        """
        candidates = []

        for domain in self.config.available_domains:
            scores = self._score_domain(domain, reflection)
            total = sum(scores.values())
            motivation = max(scores, key=scores.get)

            candidates.append({
                "domain": domain,
                "total_score": total,
                "motivation": motivation,
                "scores": scores,
            })

        # Sort by total score
        candidates.sort(key=lambda c: c["total_score"], reverse=True)

        if not candidates:
            # Fallback: random domain
            domain = random.choice(self.config.available_domains)
            return AgentDecision(
                domain=domain,
                motivation=Motivation.CURIOSITY.value,
                confidence=0.3,
                reasoning="No data available, exploring randomly.",
                timestamp=datetime.now().isoformat(),
            )

        best = candidates[0]

        # Add some stochasticity (epsilon-greedy style)
        if random.random() < 0.1 and len(candidates) > 1:
            # 10% chance of exploring a non-optimal domain
            best = random.choice(candidates[1:min(4, len(candidates))])
            best["motivation"] = Motivation.CURIOSITY.value

        # Determine training method
        training_method = self._select_training_method(
            best["domain"], reflection
        )

        # Estimate difficulty
        difficulty = self._estimate_difficulty(best["domain"], reflection)

        decision = AgentDecision(
            domain=best["domain"],
            motivation=best["motivation"],
            confidence=best["total_score"],
            reasoning=self._generate_reasoning(best, reflection),
            priority_score=best["total_score"],
            training_method=training_method,
            estimated_difficulty=difficulty,
            timestamp=datetime.now().isoformat(),
        )

        return decision

    def _score_domain(self, domain: str, reflection: dict) -> dict:
        """Score a domain across all motivation factors."""
        scores = {}
        domain_scores = reflection.get("domain_scores", {})
        my_score = domain_scores.get(domain, 0.0)

        # 1. Weakness score: lower my score = higher motivation
        if domain in reflection.get("weaknesses", []):
            weakness = 1.0
        elif my_score < 0.5:
            weakness = 0.8
        elif my_score < 0.7:
            weakness = 0.5
        else:
            weakness = 0.1
        scores[Motivation.WEAKNESS_FIXING.value] = (
            weakness * self.config.weakness_weight
        )

        # 2. Curiosity score: untried domains get high curiosity
        if domain in reflection.get("untried", []):
            curiosity = 1.0
        elif domain_scores.get(domain) is None:
            curiosity = 0.9
        else:
            curiosity = 0.1
        scores[Motivation.CURIOSITY.value] = (
            curiosity * self.config.curiosity_weight
        )

        # 3. Peer learning score: big gap between me and best peer
        suggestions = self.logbook.get_improvement_suggestions(
            self.config.agent_id
        )
        peer_gap = 0.0
        for s in suggestions:
            if s.get("domain") == domain and s.get("type") == "learn_from_peer":
                peer_gap = s.get("gap", 0)
                break
        scores[Motivation.PEER_LEARNING.value] = (
            min(1.0, peer_gap * 3) * self.config.peer_learning_weight
        )

        # 4. Diversity score: visit less-visited domains
        diversity_rec = self.logbook.get_diversity_recommendation(
            self.config.agent_id
        )
        diversity_candidates = diversity_rec.get("candidates", [])
        diversity_score = 0.0
        for c in diversity_candidates:
            if c.get("domain") == domain:
                diversity_score = c.get("priority", 0.5)
                break
        scores[Motivation.DIVERSITY.value] = (
            diversity_score * self.config.diversity_weight
        )

        # 5. Mastery push: domains close to being conquered
        if 0.80 <= my_score < self.config.mastery_threshold:
            mastery_push = 1.0 - (self.config.mastery_threshold - my_score) * 5
        else:
            mastery_push = 0.0
        scores[Motivation.MASTERY_PUSH.value] = (
            max(0, mastery_push) * self.config.mastery_push_weight
        )

        # 6. Safety boost: domains with safety violations get priority
        safety_errors = reflection.get("error_patterns", {})
        safety_count = safety_errors.get("safety_violation", 0)
        safety_boost = min(1.0, safety_count * 0.3) if safety_count else 0.0
        scores[Motivation.SAFETY_FOCUS.value] = (
            safety_boost * self.config.safety_weight
        )

        # Plateau penalty: if plateaued, reduce this domain's score
        if domain in reflection.get("plateaus", []):
            for key in scores:
                scores[key] *= 0.5  # Halve all scores for plateaued domains

        return scores

    def _select_training_method(self, domain: str, reflection: dict) -> str:
        """Select training method based on current mastery."""
        my_score = reflection.get("domain_scores", {}).get(domain, 0.0)

        if my_score < 0.3:
            return "sft"  # Need foundation
        elif my_score < 0.7:
            return "sft"  # Still building skills
        elif domain in reflection.get("plateaus", []):
            return "grpo"  # Plateau -> try RL
        else:
            return "grpo"  # Refinement stage

    def _estimate_difficulty(self, domain: str, reflection: dict) -> str:
        """Estimate task difficulty for the chosen domain."""
        my_score = reflection.get("domain_scores", {}).get(domain, 0.0)

        if my_score < 0.3:
            return "easy"
        elif my_score < 0.6:
            return "moderate"
        elif my_score < 0.85:
            return "hard"
        else:
            return "expert"

    def _generate_reasoning(self, best: dict, reflection: dict) -> str:
        """Generate human-readable reasoning for the decision."""
        domain = best["domain"]
        motivation = best["motivation"]
        my_score = reflection.get("domain_scores", {}).get(domain, 0.0)

        parts = [f"Chose {domain}"]

        if motivation == Motivation.WEAKNESS_FIXING.value:
            parts.append(f"because my score is low ({my_score:.1%})")
        elif motivation == Motivation.CURIOSITY.value:
            parts.append("to explore a new domain")
        elif motivation == Motivation.PEER_LEARNING.value:
            parts.append("because peers score much higher here")
        elif motivation == Motivation.DIVERSITY.value:
            parts.append("for domain diversity (anti-herding)")
        elif motivation == Motivation.MASTERY_PUSH.value:
            parts.append(f"to push toward mastery ({my_score:.1%} -> 90%)")
        elif motivation == Motivation.SAFETY_FOCUS.value:
            parts.append("due to safety violation history")

        if reflection.get("plateaus") and domain in reflection["plateaus"]:
            parts.append("[WARNING: domain is plateaued]")

        return " | ".join(parts)


# ============================================================
# 3. Workout Executor
# ============================================================

class WorkoutExecutor:
    """Executes a workout (evaluation + training) for the agent."""

    def __init__(self, config: AutonomousAgentConfig):
        self.config = config

    def execute_workout(
        self, decision: AgentDecision, gpu_id: int = -1
    ) -> dict:
        """Execute a complete workout cycle.

        Steps:
        1. Bind to assigned GPU via CUDA_VISIBLE_DEVICES
        2. Evaluate current performance on the chosen domain
        3. Analyze failures
        4. Generate targeted training data
        5. Train
        6. Re-evaluate

        Returns:
            dict with workout results
        """
        import os
        start_time = time.time()
        domain = decision.domain

        # GPU isolation: bind this agent to its assigned GPU
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(
                f"[{self.config.agent_id}] Bound to GPU {gpu_id}"
            )

        results = {
            "domain": domain,
            "agent_id": self.config.agent_id,
            "motivation": decision.motivation,
            "gpu_id": gpu_id,
            "pre_score": 0.0,
            "post_score": 0.0,
            "improvement": 0.0,
            "errors": [],
            "training_method": decision.training_method,
            "tasks_completed": 0,
            "duration_ms": 0.0,
            "success": False,
        }

        try:
            # Step 1: Evaluate
            logger.info(
                f"[{self.config.agent_id}] Evaluating on {domain}..."
            )
            eval_result = self._evaluate(domain)
            results["pre_score"] = eval_result.get("action_score", 0.0)
            results["tasks_completed"] = eval_result.get("num_tasks", 0)
            results["errors"] = eval_result.get("error_types", [])

            # Step 2: Analyze and generate
            logger.info(
                f"[{self.config.agent_id}] Pre-score: "
                f"{results['pre_score']:.1%}, generating training data..."
            )
            training_data = self._generate_training_data(
                domain, eval_result, decision
            )

            # Step 3: Train
            if training_data:
                logger.info(
                    f"[{self.config.agent_id}] Training "
                    f"({decision.training_method})..."
                )
                new_model = self._train(
                    domain, training_data, decision.training_method
                )
                if new_model:
                    self.config.model_path = new_model

            # Step 4: Re-evaluate (optional, expensive)
            results["post_score"] = results["pre_score"]  # Placeholder
            results["improvement"] = (
                results["post_score"] - results["pre_score"]
            )
            results["success"] = True

        except Exception as e:
            logger.error(
                f"[{self.config.agent_id}] Workout failed: {e}"
            )
            import traceback
            traceback.print_exc()
            results["errors"].append(f"workout_crash: {str(e)}")

        # Free GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        results["duration_ms"] = (time.time() - start_time) * 1000
        return results

    def _evaluate(self, domain: str) -> dict:
        """Evaluate current model on a domain."""
        try:
            from bioagents.evaluation.agent_runner import AgentRunner, RunConfig

            # Use auto-tuned inference params if available
            max_new_tokens = self.config.inference_max_new_tokens or 1024
            gpu_mem_util = self.config.gpu_memory_utilization or 0.85

            run_config = RunConfig(
                model_name_or_path=self.config.model_path,
                backend=self.config.backend,
                domain=domain,
                max_turns=self.config.max_turns,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
                gpu_memory_utilization=gpu_mem_util,
                log_dir=str(
                    Path(self.config.log_dir) / self.config.agent_id / "eval"
                ),
            )

            runner = AgentRunner(run_config)
            runner.load_model()
            task_results = runner.run_all_tasks()

            # Compute stats
            scores = [r.action_score for r in task_results]
            error_types = []
            for r in task_results:
                if r.action_score < 0.8:
                    # Quick error categorization
                    if r.total_turns <= 2:
                        error_types.append("premature_stop")
                    elif r.total_turns >= 12:
                        error_types.append("over_investigation")
                    else:
                        error_types.append("reasoning_error")

            del runner
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

            return {
                "action_score": (
                    sum(scores) / len(scores) if scores else 0.0
                ),
                "num_tasks": len(task_results),
                "num_passed": sum(1 for s in scores if s >= 0.8),
                "error_types": error_types,
                "task_results": task_results,
            }

        except Exception as e:
            logger.warning(
                f"[{self.config.agent_id}] Evaluation failed: {e}"
            )
            return {
                "action_score": 0.0,
                "num_tasks": 0,
                "error_types": [f"eval_crash: {str(e)}"],
            }

    def _generate_training_data(
        self, domain: str, eval_result: dict, decision: AgentDecision
    ) -> Optional[list]:
        """Generate targeted training data based on evaluation results."""
        try:
            from bioagents.gym.gym_coach import TargetedDataGenerator

            generator = TargetedDataGenerator()
            error_types = eval_result.get("error_types", [])

            if not error_types:
                return None

            # Convert to priorities format
            from collections import Counter
            error_counts = Counter(error_types)
            priorities = [
                {
                    "domain": domain,
                    "failure_type": err_type,
                    "count": count,
                    "avg_severity": 3,
                    "mastery": "intermediate",
                    "priority_score": count * 3,
                }
                for err_type, count in error_counts.items()
            ]

            tasks = generator.generate_targeted_tasks(
                priorities, max_tasks=50, tasks_per_weakness=10
            )
            return tasks

        except Exception as e:
            logger.warning(
                f"[{self.config.agent_id}] Data generation failed: {e}"
            )
            return None

    def _train(
        self, domain: str, training_data: list, method: str
    ) -> Optional[str]:
        """Train the model on generated data."""
        try:
            from bioagents.gym.self_play import SelfPlayConfig, SelfPlayLoop

            iteration_id = int(time.time()) % 100000
            output_dir = str(
                Path(self.config.output_dir)
                / self.config.agent_id
                / f"workout_{iteration_id}"
            )

            # Use auto-tuned params if available, fallback to defaults
            train_batch = self.config.train_batch_size or 2
            grad_accum = self.config.gradient_accumulation_steps or 8
            lora_r = self.config.lora_r or 16
            train_max_len = self.config.train_max_length or 4096

            sp_config = SelfPlayConfig(
                model_name_or_path=self.config.model_path,
                backend=self.config.backend,
                domains=[domain],
                tasks_per_domain=self.config.eval_tasks_per_domain,
                max_iterations=1,
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.training_epochs,
                quality_threshold=self.config.quality_threshold,
                min_trajectories_for_training=10,
                batch_size=train_batch,
                gradient_accumulation_steps=grad_accum,
                lora_r=lora_r,
                max_length=train_max_len,
                output_dir=output_dir,
                log_dir=str(
                    Path(self.config.log_dir) / self.config.agent_id
                ),
                trajectory_dir=str(
                    Path(output_dir) / "trajectories"
                ),
            )

            loop = SelfPlayLoop(sp_config)
            loop.run()

            merged_path = str(
                Path(output_dir) / "iter_1_sft" / "merged"
            )
            if Path(merged_path).exists():
                return merged_path

        except Exception as e:
            logger.warning(
                f"[{self.config.agent_id}] Training failed: {e}"
            )

        return None


# ============================================================
# 4. AutonomousAgent (Main Class)
# ============================================================

class AutonomousAgent:
    """A self-aware, autonomously learning agent for the Healthcare AI GYM.

    The agent runs its own learning loop:
    1. REFLECT on recent performance
    2. CHOOSE which domain to work on
    3. EXECUTE a workout (evaluate -> train)
    4. RECORD results to SharedLogbook
    5. REPEAT

    Usage:
        agent = AutonomousAgent(config, logbook)
        agent.run_one_cycle()       # Single cycle
        agent.run_continuous()      # Never-ending loop
    """

    def __init__(self, config: AutonomousAgentConfig, logbook):
        self.config = config
        self.logbook = logbook

        # Internal modules
        self.awareness = SelfAwareness(config.agent_id, logbook)
        self.strategy = StrategySelector(config, logbook)
        self.executor = WorkoutExecutor(config)

        # State
        self.state = AgentState.IDLE
        self.current_gpu: int = -1
        self.total_cycles: int = 0
        self.decision_history: list = []
        self._stop_requested = False

        # Paths
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[AutonomousAgent:{config.agent_id}] Initialized"
        )
        logger.info(
            f"  Model: {config.model_path}"
        )
        logger.info(
            f"  Domains: {config.available_domains}"
        )

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    def stop(self):
        """Request graceful stop."""
        self._stop_requested = True

    def run_one_cycle(self, gpu_id: int = -1) -> dict:
        """Run a single autonomous learning cycle.

        Args:
            gpu_id: GPU to use (-1 for auto)

        Returns:
            dict with cycle results
        """
        cycle_start = time.time()
        self.current_gpu = gpu_id
        self.total_cycles += 1

        cycle_result = {
            "agent_id": self.config.agent_id,
            "cycle": self.total_cycles,
            "gpu_id": gpu_id,
        }

        logger.info(
            f"\n{'='*60}"
        )
        logger.info(
            f"  [{self.config.agent_id}] Cycle #{self.total_cycles} "
            f"(GPU {gpu_id})"
        )
        logger.info(
            f"{'='*60}"
        )

        # Phase 1: REFLECT
        self.state = AgentState.REFLECTING
        logger.info(
            f"  [1/4] REFLECTING on recent performance..."
        )
        reflection = self.awareness.reflect()

        if reflection.get("is_new"):
            logger.info("    First time in the gym! Exploring...")
        else:
            logger.info(
                f"    Overall: {reflection['overall_score']:.1%} | "
                f"Strengths: {reflection['strengths'][:2]} | "
                f"Weaknesses: {reflection['weaknesses'][:2]}"
            )
            if reflection["plateaus"]:
                logger.warning(
                    f"    Plateaued: {reflection['plateaus']}"
                )
            logger.info(
                f"    Improvement rate: "
                f"{reflection['improvement_rate']:+.2%}"
            )

        cycle_result["reflection"] = {
            "overall_score": reflection.get("overall_score", 0),
            "strengths": reflection.get("strengths", []),
            "weaknesses": reflection.get("weaknesses", []),
            "plateaus": reflection.get("plateaus", []),
        }

        # Phase 2: CHOOSE
        self.state = AgentState.CHOOSING
        logger.info(
            f"  [2/4] CHOOSING next domain..."
        )
        decision = self.strategy.choose_next_action(reflection)
        self.decision_history.append(decision)

        logger.info(
            f"    Decision: {decision.domain} "
            f"(motivation={decision.motivation})"
        )
        logger.info(
            f"    Reasoning: {decision.reasoning}"
        )
        logger.info(
            f"    Method: {decision.training_method} | "
            f"Difficulty: {decision.estimated_difficulty}"
        )

        cycle_result["decision"] = {
            "domain": decision.domain,
            "motivation": decision.motivation,
            "training_method": decision.training_method,
            "reasoning": decision.reasoning,
        }

        # Phase 3: TRAIN (workout)
        self.state = AgentState.TRAINING
        logger.info(
            f"  [3/4] EXECUTING workout on {decision.domain}..."
        )
        workout_result = self.executor.execute_workout(decision, gpu_id)

        logger.info(
            f"    Score: {workout_result['pre_score']:.1%} | "
            f"Tasks: {workout_result['tasks_completed']} | "
            f"Errors: {len(workout_result['errors'])}"
        )

        cycle_result["workout"] = workout_result

        # Phase 4: RECORD
        self.state = AgentState.RECORDING
        logger.info(
            f"  [4/4] RECORDING to shared logbook..."
        )
        from bioagents.gym.shared_logbook import WorkoutEntry
        entry = WorkoutEntry(
            agent_id=self.config.agent_id,
            domain=decision.domain,
            task_id=f"cycle_{self.total_cycles}",
            action_score=workout_result["pre_score"],
            reward_score=workout_result.get("post_score", 0),
            errors=workout_result["errors"],
            tools_used=[],
            self_reflection=decision.reasoning,
            confidence=decision.confidence,
            iteration=self.total_cycles,
            model_path=self.config.model_path,
            training_method=decision.training_method,
            gpu_id=gpu_id,
            duration_ms=workout_result["duration_ms"],
        )
        self.logbook.record_workout(entry)

        # Done
        self.state = AgentState.IDLE
        cycle_ms = (time.time() - cycle_start) * 1000
        cycle_result["duration_ms"] = cycle_ms

        logger.info(
            f"  Cycle complete in {cycle_ms/1000:.1f}s"
        )

        # Check leaderboard position
        leaderboard = self.logbook.get_leaderboard(decision.domain)
        my_rank = next(
            (
                i + 1 for i, e in enumerate(leaderboard)
                if e.agent_id == self.config.agent_id
            ),
            len(leaderboard),
        )
        logger.info(
            f"  Leaderboard rank in {decision.domain}: "
            f"#{my_rank}/{len(leaderboard)}"
        )

        return cycle_result

    def run_continuous(self, gpu_id: int = -1, max_cycles: int = 0):
        """Run the agent continuously until stopped.

        Args:
            gpu_id: GPU to use
            max_cycles: Max cycles (0 = infinite)
        """
        logger.info(
            f"[{self.config.agent_id}] Starting continuous mode "
            f"(GPU {gpu_id})"
        )

        cycle = 0
        try:
            while not self._stop_requested:
                cycle += 1
                if max_cycles > 0 and cycle > max_cycles:
                    break

                result = self.run_one_cycle(gpu_id)

                # Brief pause between cycles
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info(
                f"[{self.config.agent_id}] Interrupted, "
                f"completed {cycle} cycles"
            )
        except Exception as e:
            logger.error(
                f"[{self.config.agent_id}] Error: {e}"
            )
            import traceback
            traceback.print_exc()

        logger.info(
            f"[{self.config.agent_id}] Stopped after "
            f"{cycle} cycles"
        )

    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            "agent_id": self.config.agent_id,
            "state": self.state.value,
            "model_path": self.config.model_path,
            "gpu": self.current_gpu,
            "total_cycles": self.total_cycles,
            "last_decision": (
                {
                    "domain": self.decision_history[-1].domain,
                    "motivation": self.decision_history[-1].motivation,
                }
                if self.decision_history
                else None
            ),
        }
