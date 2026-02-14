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
    training_method: str = "grpo"
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
    benchmark_every_n_cycles: int = 3  # Run external benchmarks every N cycles (0=never)
    benchmark_max_samples: int = 0     # 0 = full benchmark, >0 = sample N questions
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
        """Select training method — always GRPO (pure RL, no SFT).

        Rationale: Pre-trained 7-8B models already have medical knowledge.
        Rather than memorizing answers via SFT, the model should learn
        through trial-and-error in the GYM environment using GRPO rewards:
        - 5D composite reward (accuracy, format, process, safety, coherence)
        - External benchmark signals (TextQA, VQA, MedLFQA, EHR)
        This produces better agent behavior than SFT fine-tuning.
        """
        return "grpo"

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
            # Step 1: Evaluate on internal GYM tasks
            logger.info(
                f"[{self.config.agent_id}] Evaluating on {domain}..."
            )
            eval_result = self._evaluate(domain)
            results["pre_score"] = eval_result.get("action_score", 0.0)
            results["tasks_completed"] = eval_result.get("num_tasks", 0)
            results["errors"] = eval_result.get("error_types", [])

            # Step 2: RL Training via Multi-Turn GRPO
            # No SFT — the model learns directly from environment
            # interaction rewards (5D: accuracy, format, process, safety, coherence)
            logger.info(
                f"[{self.config.agent_id}] Pre-score: "
                f"{results['pre_score']:.1%}, starting GRPO training..."
            )
            new_model = self._train_grpo(domain, eval_result, decision)
            if new_model:
                self.config.model_path = new_model
                logger.info(
                    f"[{self.config.agent_id}] Model updated: {new_model}"
                )

            # Step 3: Re-evaluate after GRPO
            if new_model:
                logger.info(
                    f"[{self.config.agent_id}] Re-evaluating after GRPO..."
                )
                post_result = self._evaluate(domain)
                results["post_score"] = post_result.get("action_score", 0.0)
            else:
                results["post_score"] = results["pre_score"]

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

    def evaluate_external_benchmarks(self, cycle_num: int) -> dict:
        """Run external benchmark evaluation across ALL categories.

        Categories evaluated:
        1. Text QA (8 benchmarks) — MedQA, MedMCQA, MMLU x6
        2. Vision QA (6 datasets, VL models only) — VQA-RAD, SLAKE, PathVQA, etc.
        3. MedLFQA (5 datasets) — Long-form medical QA (KQA, LiveQA, MedicationQA, etc.)
        4. EHR Benchmarks (2 databases) — MIMIC-III (50 tasks) + eICU (50 tasks)

        This is the comprehensive evaluation that produces paper-ready
        numbers. It's expensive, so it only runs every
        ``benchmark_every_n_cycles`` cycles.

        Returns:
            dict with benchmark results per category, or empty dict if skipped.
        """
        interval = self.config.benchmark_every_n_cycles
        if interval <= 0 or (cycle_num % interval != 0):
            return {}

        logger.info(
            f"[{self.config.agent_id}] Running FULL EXTERNAL BENCHMARK "
            f"evaluation (cycle #{cycle_num})..."
        )

        results = {}
        benchmark_output_dir = str(
            Path(self.config.log_dir)
            / self.config.agent_id
            / "benchmarks"
            / f"cycle_{cycle_num}"
        )

        # ── 1. Text QA Benchmarks (MedQA, MedMCQA, MMLU x6) ─────────────
        try:
            from bioagents.evaluation.benchmark_eval import (
                BenchmarkEvaluator,
                BenchmarkConfig,
            )
            from bioagents.evaluation.agent_runner import repair_model_config
            repair_model_config(self.config.model_path)

            text_benchmarks = [
                "medqa", "medmcqa",
                "mmlu_clinical", "mmlu_professional",
                "mmlu_anatomy", "mmlu_genetics",
                "mmlu_biology", "mmlu_college_med",
            ]

            bench_config = BenchmarkConfig(
                model_name_or_path=self.config.model_path,
                model_name=self.config.agent_id,
                benchmarks=text_benchmarks,
                max_samples=self.config.benchmark_max_samples,
                batch_size=self.config.inference_batch_size or 8,
                output_dir=benchmark_output_dir,
                temperature=0.0,
                max_new_tokens=256,
            )

            evaluator = BenchmarkEvaluator(bench_config)
            text_results = evaluator.evaluate_all()
            results["text_qa"] = text_results

            # Free memory
            del evaluator
            import torch
            torch.cuda.empty_cache()

            logger.info(
                f"[{self.config.agent_id}] Text QA benchmarks complete:"
            )
            for bench, res in text_results.items():
                if isinstance(res, dict) and "accuracy" in res:
                    logger.info(
                        f"    {bench}: {res['accuracy']:.3f} "
                        f"({res.get('correct', '?')}/{res.get('total', '?')})"
                    )

        except Exception as e:
            logger.warning(
                f"[{self.config.agent_id}] Text benchmark eval failed: {e}"
            )
            import traceback
            traceback.print_exc()
            results["text_qa_error"] = str(e)

        # ── 2. VQA Benchmarks (VL models only) ───────────────────────────
        is_vl = getattr(
            self.config, "_model_profile", None
        )
        if is_vl and getattr(is_vl, "is_vl_model", False):
            try:
                from bioagents.evaluation.vqa_benchmark_eval import (
                    VQABenchmarkEvaluator,
                    VQABenchmarkConfig,
                )

                vqa_config = VQABenchmarkConfig(
                    model_name_or_path=self.config.model_path,
                    model_name=self.config.agent_id,
                    benchmarks=[
                        "vqa_rad", "slake", "pathvqa",
                        "pmc_vqa", "vqa_med_2021", "quilt_vqa",
                    ],
                    max_samples=self.config.benchmark_max_samples or 200,
                    output_dir=benchmark_output_dir,
                )

                vqa_evaluator = VQABenchmarkEvaluator(vqa_config)
                vqa_results = vqa_evaluator.evaluate_all()
                results["vqa"] = vqa_results

                del vqa_evaluator
                import torch
                torch.cuda.empty_cache()

                logger.info(
                    f"[{self.config.agent_id}] VQA benchmarks complete:"
                )
                for bench, res in vqa_results.items():
                    if isinstance(res, dict) and "accuracy" in res:
                        logger.info(
                            f"    {bench}: {res['accuracy']:.3f}"
                        )

            except Exception as e:
                logger.warning(
                    f"[{self.config.agent_id}] VQA benchmark eval failed: {e}"
                )
                results["vqa_error"] = str(e)

        # ── 3. MedLFQA Long-form QA Benchmarks ───────────────────────────
        try:
            from bioagents.evaluation.benchmark_eval import BenchmarkConfig
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            # MedLFQA uses the same evaluation logic as run_full_benchmark_suite
            # but we integrate it here for the GYM cycle
            medlfqa_datasets = {
                "kqa_golden": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
                "live_qa": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
                "medication_qa": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
                "healthsearch_qa": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
                "kqa_silver": "evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl",
            }

            project_root = Path(__file__).parent.parent.parent
            available_datasets = {
                k: v for k, v in medlfqa_datasets.items()
                if (project_root / v).exists()
            }

            if available_datasets:
                logger.info(
                    f"[{self.config.agent_id}] Running MedLFQA benchmarks "
                    f"({len(available_datasets)} datasets)..."
                )

                # Load model once for all MedLFQA datasets
                model_config = AutoConfig.from_pretrained(
                    self.config.model_path, trust_remote_code=True
                )
                model_type = getattr(model_config, "model_type", "")
                is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

                load_kwargs = dict(
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                )

                if is_qwen_vl:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.config.model_path, **load_kwargs
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_path, **load_kwargs
                    )
                model.eval()

                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path, trust_remote_code=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                import json as _json
                medlfqa_results = {}
                max_samples = self.config.benchmark_max_samples or 0

                for ds_key, ds_path in available_datasets.items():
                    data = []
                    full_path = project_root / ds_path
                    with open(full_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append(_json.loads(line))
                    if max_samples > 0:
                        data = data[:max_samples]

                    if not data:
                        continue

                    correct_count = 0
                    total_count = 0
                    rouge_l_sum = 0.0

                    for item in data:
                        question = item.get("Question", "")
                        reference = item.get("Free_form_answer", "")
                        if not question:
                            continue

                        messages = [
                            {"role": "system", "content": "You are a medical expert. Provide detailed, accurate answers."},
                            {"role": "user", "content": f"Question: {question}\n\nProvide a comprehensive answer."},
                        ]
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = tokenizer(
                            text, return_tensors="pt", truncation=True, max_length=4096
                        ).to(model.device)

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs, max_new_tokens=512, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                            )

                        generated = outputs[0][inputs["input_ids"].shape[-1]:]
                        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

                        # Compute token-level F1 as a proxy for accuracy
                        pred_tokens = set(response.lower().split())
                        ref_tokens = set(reference.lower().split())
                        if pred_tokens and ref_tokens:
                            common = pred_tokens & ref_tokens
                            if common:
                                precision = len(common) / len(pred_tokens)
                                recall = len(common) / len(ref_tokens)
                                f1 = 2 * precision * recall / (precision + recall)
                            else:
                                f1 = 0.0
                        else:
                            f1 = 0.0

                        rouge_l_sum += f1
                        total_count += 1

                    if total_count > 0:
                        medlfqa_results[ds_key] = {
                            "token_f1": rouge_l_sum / total_count,
                            "total": total_count,
                        }

                results["medlfqa"] = medlfqa_results

                del model
                torch.cuda.empty_cache()

                logger.info(
                    f"[{self.config.agent_id}] MedLFQA benchmarks complete:"
                )
                for ds, res in medlfqa_results.items():
                    logger.info(
                        f"    {ds}: token_f1={res['token_f1']:.3f} "
                        f"(n={res['total']})"
                    )
            else:
                logger.info(
                    f"[{self.config.agent_id}] MedLFQA: no datasets found, skipping"
                )

        except Exception as e:
            logger.warning(
                f"[{self.config.agent_id}] MedLFQA benchmark eval failed: {e}"
            )
            import traceback
            traceback.print_exc()
            results["medlfqa_error"] = str(e)

        # ── 4. EHR Benchmarks (MIMIC-III + eICU) ─────────────────────────
        try:
            from bioagents.evaluation.ehr_benchmark_eval import (
                EHRBenchmarkEvaluator,
                EHRBenchmarkConfig,
            )

            ehr_config = EHRBenchmarkConfig(
                model_name_or_path=self.config.model_path,
                model_name=self.config.agent_id,
                backend=self.config.backend,
                benchmarks=["mimic_iii", "eicu"],
                max_samples=self.config.benchmark_max_samples,
                max_turns=self.config.max_turns,
                output_dir=str(Path(benchmark_output_dir) / "ehr"),
            )

            ehr_evaluator = EHRBenchmarkEvaluator(ehr_config)
            ehr_results = ehr_evaluator.evaluate_all()
            results["ehr"] = ehr_results

            del ehr_evaluator
            import torch
            torch.cuda.empty_cache()

            logger.info(
                f"[{self.config.agent_id}] EHR benchmarks complete:"
            )
            for db_key, res in ehr_results.items():
                if isinstance(res, dict) and "avg_action_score" in res:
                    logger.info(
                        f"    {db_key}: action_score={res['avg_action_score']:.3f} "
                        f"({res.get('completed', '?')}/{res.get('total_tasks', '?')} tasks)"
                    )

        except Exception as e:
            logger.warning(
                f"[{self.config.agent_id}] EHR benchmark eval failed: {e}"
            )
            import traceback
            traceback.print_exc()
            results["ehr_error"] = str(e)

        # ── Summary ───────────────────────────────────────────────────────
        categories_done = [k for k in results if not k.endswith("_error")]
        logger.info(
            f"[{self.config.agent_id}] EXTERNAL BENCHMARK complete: "
            f"{len(categories_done)} categories evaluated "
            f"({', '.join(categories_done)})"
        )

        return results

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

    def _train_grpo(
        self,
        domain: str,
        eval_result: dict,
        decision: AgentDecision,
    ) -> Optional[str]:
        """Train the model via Multi-Turn GRPO (pure RL, no SFT).

        This is the core training method of Healthcare AI GYM.
        The model learns directly from environment interaction:

        1. Sample tasks from the chosen domain
        2. Run G rollouts per task through the GYM environment
        3. Score each trajectory with 5D composite reward
           (accuracy + format + process + safety + coherence)
        4. Compute group-relative advantages (GRPO)
        5. Update policy to maximize advantage-weighted returns

        If benchmark results are available from recent cycles,
        they are used to boost reward weights for weak areas.

        Returns:
            Path to merged model checkpoint, or None if training failed.
        """
        try:
            from bioagents.training.grpo_trainer import (
                BioAgentGRPOConfig,
                MultiTurnGRPOConfig,
                train_multiturn,
            )
            from bioagents.evaluation.agent_runner import repair_model_config

            # Repair config if needed (Transformers version compat)
            repair_model_config(self.config.model_path)

            iteration_id = int(time.time()) % 100000
            output_dir = str(
                Path(self.config.output_dir)
                / self.config.agent_id
                / f"grpo_{iteration_id}"
            )

            # Use auto-tuned params if available, fallback to defaults
            train_batch = self.config.train_batch_size or 2
            grad_accum = self.config.gradient_accumulation_steps or 4
            lora_r = self.config.lora_r or 16
            train_max_len = self.config.train_max_length or 4096
            num_rollouts = 4  # G: rollouts per task for group-relative

            # Find domain tasks file
            tasks_path = str(
                Path("data/domains") / domain / "tasks.json"
            )
            if not Path(tasks_path).exists():
                logger.warning(
                    f"[{self.config.agent_id}] No tasks file for "
                    f"{domain}: {tasks_path}"
                )
                return None

            # --- Analyze benchmark weaknesses to adjust reward weights ---
            reward_functions = self._get_reward_weights_from_benchmarks(
                domain, eval_result
            )

            logger.info(
                f"[{self.config.agent_id}] GRPO config: "
                f"domain={domain}, batch={train_batch}, "
                f"grad_accum={grad_accum}, lora_r={lora_r}, "
                f"rollouts={num_rollouts}, "
                f"rewards={reward_functions}"
            )

            grpo_config = BioAgentGRPOConfig(
                model_name_or_path=self.config.model_path,
                torch_dtype="bfloat16",
                attn_implementation="sdpa",
                peft_enabled=True,
                peft_r=lora_r,
                peft_lora_alpha=lora_r * 2,
                peft_lora_dropout=0.05,
                domain=domain,
                tasks_path=tasks_path,
                output_dir=output_dir,
                num_train_epochs=self.config.training_epochs,
                per_device_train_batch_size=train_batch,
                gradient_accumulation_steps=grad_accum,
                learning_rate=self.config.learning_rate,
                bf16=True,
                num_generations=num_rollouts,
                beta=0.04,
                temperature=0.7,
                max_turns=self.config.max_turns,
                use_gym_env=True,
                reward_functions=reward_functions,
                use_wandb=False,
                log_dir=str(
                    Path(self.config.log_dir) / self.config.agent_id
                ),
                seed=42 + iteration_id % 1000,
            )

            # Run multi-turn GRPO training
            trajectories = train_multiturn(grpo_config)

            logger.info(
                f"[{self.config.agent_id}] GRPO complete: "
                f"{len(trajectories)} trajectories collected"
            )

            # Check for merged model or best checkpoint
            final_path = str(Path(output_dir) / "final")
            best_path = str(Path(output_dir) / "best")

            # Merge LoRA if needed
            for candidate in [best_path, final_path]:
                if Path(candidate).exists():
                    merged_path = candidate + "_merged"
                    self._merge_lora_checkpoint(candidate, merged_path)
                    if Path(merged_path).exists():
                        return merged_path

            # If no best/final, check if the output_dir has adapter
            if Path(output_dir).exists():
                return output_dir

        except Exception as e:
            logger.warning(
                f"[{self.config.agent_id}] GRPO training failed: {e}"
            )
            import traceback
            traceback.print_exc()

        return None

    def _get_reward_weights_from_benchmarks(
        self, domain: str, eval_result: dict
    ) -> list:
        """Compute adaptive reward weights based on benchmark + GYM signals.

        The idea: if the model scores poorly on knowledge benchmarks
        (MedQA, MMLU), boost the accuracy reward weight. If it makes
        format errors or safety violations in GYM tasks, boost those
        respective reward weights.

        Returns:
            list of reward function dicts with adaptive weights
        """
        # Default 5D reward weights
        accuracy_w = 0.30
        format_w = 0.15
        process_w = 0.25
        safety_w = 0.20
        coherence_w = 0.10

        # --- Adapt from GYM evaluation errors ---
        error_types = eval_result.get("error_types", [])
        if error_types:
            from collections import Counter
            error_counts = Counter(error_types)
            total_errors = sum(error_counts.values())

            # If many reasoning errors, boost process reward
            reasoning_ratio = error_counts.get("reasoning_error", 0) / max(total_errors, 1)
            if reasoning_ratio > 0.5:
                process_w += 0.10
                accuracy_w -= 0.05
                format_w -= 0.05

            # If premature stops, boost process (need longer reasoning)
            premature_ratio = error_counts.get("premature_stop", 0) / max(total_errors, 1)
            if premature_ratio > 0.3:
                process_w += 0.05
                coherence_w += 0.05
                accuracy_w -= 0.05
                format_w -= 0.05

        # --- Adapt from benchmark scores (if available in logbook) ---
        try:
            # Check if there are recent benchmark scores in agent history
            from bioagents.gym.shared_logbook import SharedLogbook
            # Access logbook through the config's log_dir
            logbook_path = Path(self.config.log_dir) / "shared_logbook.json"
            if logbook_path.exists():
                import json
                with open(logbook_path, "r") as f:
                    logbook_data = json.load(f)

                # Find latest benchmark entries for this agent
                benchmark_entries = [
                    e for e in logbook_data.get("entries", [])
                    if e.get("agent_id") == self.config.agent_id
                    and e.get("domain", "").startswith("benchmark_")
                ]

                if benchmark_entries:
                    # Average score across all benchmarks (text_qa, vqa, medlfqa, ehr)
                    bench_scores = [
                        e.get("action_score", 0) for e in benchmark_entries[-20:]
                    ]
                    avg_benchmark = sum(bench_scores) / max(len(bench_scores), 1)

                    if avg_benchmark < 0.4:
                        # Very poor knowledge → heavily boost accuracy
                        accuracy_w += 0.15
                        process_w -= 0.05
                        format_w -= 0.05
                        coherence_w -= 0.05
                        logger.info(
                            f"[{self.config.agent_id}] Benchmark avg={avg_benchmark:.2f} "
                            f"(low) → boosting accuracy reward"
                        )
                    elif avg_benchmark < 0.6:
                        # Moderate → slightly boost accuracy
                        accuracy_w += 0.05
                        format_w -= 0.025
                        coherence_w -= 0.025

                    # EHR-specific: low EHR action_score → boost process reward
                    # (EHR tasks require multi-step tool-use reasoning)
                    ehr_entries = [
                        e for e in benchmark_entries[-20:]
                        if "mimic" in e.get("domain", "") or "eicu" in e.get("domain", "")
                    ]
                    if ehr_entries:
                        avg_ehr = sum(e.get("action_score", 0) for e in ehr_entries) / len(ehr_entries)
                        if avg_ehr < 0.5:
                            process_w += 0.05
                            accuracy_w -= 0.025
                            format_w -= 0.025
                            logger.info(
                                f"[{self.config.agent_id}] EHR avg={avg_ehr:.2f} "
                                f"(low) → boosting process reward for tool-use"
                            )
        except Exception:
            pass  # Logbook not available, use defaults

        # Normalize to sum = 1.0
        total = accuracy_w + format_w + process_w + safety_w + coherence_w
        accuracy_w /= total
        format_w /= total
        process_w /= total
        safety_w /= total
        coherence_w /= total

        return [
            {"name": "accuracy", "weight": round(accuracy_w, 3)},
            {"name": "format", "weight": round(format_w, 3)},
            {"name": "process", "weight": round(process_w, 3)},
            {"name": "safety", "weight": round(safety_w, 3)},
            {"name": "coherence", "weight": round(coherence_w, 3)},
        ]

    def _merge_lora_checkpoint(self, adapter_path: str, output_path: str):
        """Merge LoRA adapter weights into the base model."""
        if Path(output_path).exists() and (Path(output_path) / "config.json").exists():
            logger.info(f"  Merged model already exists: {output_path}")
            return

        try:
            import subprocess
            merge_script = f"""
import torch, json, sys
sys.path.insert(0, ".")
from pathlib import Path
from peft import PeftModel

adapter_path = "{adapter_path}"
output_path = "{output_path}"

if not Path(adapter_path).exists():
    print(f"Adapter not found: {{adapter_path}}")
    exit(1)

# Find base model from adapter config
adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
if not adapter_cfg_path.exists():
    print(f"No adapter_config.json in {{adapter_path}}, skipping merge")
    exit(0)

adapter_cfg = json.load(open(adapter_cfg_path))
base_path = adapter_cfg.get("base_model_name_or_path", "")

from bioagents.evaluation.agent_runner import repair_model_config
repair_model_config(base_path)

from bioagents.gym.model_profile import ModelProfiler
profile = ModelProfiler.profile(base_path)
model = profile.load_model(device_map="cpu")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(output_path)

tokenizer = profile.load_tokenizer()
tokenizer.save_pretrained(output_path)

# Copy processor files from base if VL model
if profile.requires_processor:
    import shutil
    for fname in ["preprocessor_config.json", "chat_template.json",
                   "special_tokens_map.json", "added_tokens.json"]:
        src = Path(base_path) / fname
        dst = Path(output_path) / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

print(f"Merged to: {{output_path}}")
"""
            script_path = f"/tmp/merge_grpo_{adapter_path.replace('/', '_')[-40:]}.py"
            with open(script_path, "w") as f:
                f.write(merge_script)

            result = subprocess.run(
                ["python", script_path],
                timeout=600,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(f"LoRA merge failed: {result.stderr[:500]}")
            else:
                logger.info(f"LoRA merge complete: {output_path}")

        except Exception as e:
            logger.warning(f"LoRA merge error: {e}")


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
            f"  [1/5] REFLECTING on recent performance..."
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

        # Phase 2: EXTERNAL BENCHMARK (periodic, BEFORE training)
        # This must happen before GRPO training so that benchmark
        # results can inform the adaptive reward weights.
        benchmark_results = self.executor.evaluate_external_benchmarks(
            self.total_cycles
        )
        if benchmark_results:
            cycle_result["benchmarks"] = benchmark_results
            logger.info(
                f"  [2/5] EXTERNAL BENCHMARKS evaluated "
                f"({len(benchmark_results)} categories)"
            )

            # Record benchmark entries to logbook (before training)
            for category, cat_results in benchmark_results.items():
                if isinstance(cat_results, dict) and "error" not in category:
                    for bench_name, res in cat_results.items():
                        if not isinstance(res, dict):
                            continue
                        # Extract score: text_qa/vqa use "accuracy",
                        # medlfqa uses "token_f1", ehr uses "avg_action_score"
                        score = (
                            res.get("accuracy")
                            or res.get("token_f1")
                            or res.get("avg_action_score")
                        )
                        if score is not None:
                            from bioagents.gym.shared_logbook import WorkoutEntry
                            bench_entry = WorkoutEntry(
                                agent_id=self.config.agent_id,
                                domain=f"benchmark_{bench_name}",
                                task_id=f"benchmark_cycle_{self.total_cycles}",
                                action_score=score,
                                reward_score=score,
                                errors=[],
                                tools_used=[],
                                self_reflection=f"External benchmark ({category}): {bench_name}",
                                confidence=1.0,
                                iteration=self.total_cycles,
                                model_path=self.config.model_path,
                                training_method="benchmark",
                                gpu_id=gpu_id,
                                duration_ms=0,
                            )
                            self.logbook.record_workout(bench_entry)
        else:
            logger.info(
                f"  [2/5] BENCHMARK: skipped (not a benchmark cycle)"
            )

        # Phase 3: CHOOSE
        self.state = AgentState.CHOOSING
        logger.info(
            f"  [3/5] CHOOSING next domain..."
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

        # Phase 4: GRPO TRAIN (workout)
        # The model learns through multi-turn environment interaction
        # with adaptive 5D rewards (informed by Phase 2 benchmarks)
        self.state = AgentState.TRAINING
        logger.info(
            f"  [4/5] EXECUTING GRPO workout on {decision.domain}..."
        )
        workout_result = self.executor.execute_workout(decision, gpu_id)

        logger.info(
            f"    Score: {workout_result['pre_score']:.1%} → "
            f"{workout_result.get('post_score', 0):.1%} | "
            f"Tasks: {workout_result['tasks_completed']} | "
            f"Δ: {workout_result.get('improvement', 0):+.1%}"
        )

        cycle_result["workout"] = workout_result

        # Phase 5: RECORD
        self.state = AgentState.RECORDING
        logger.info(
            f"  [5/5] RECORDING to shared logbook..."
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
