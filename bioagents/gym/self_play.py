"""Self-Play Training Loop for BIOAgents Healthcare AI GYM.

Core contribution: Agent-driven iterative improvement loop.
1. Collect trajectories by running agent in GYM environments
2. Judge trajectory quality with LLM-as-Judge
3. Filter high-quality trajectories → SFT data
4. Train on filtered trajectories (SFT or GRPO)
5. Evaluate on held-out benchmarks
6. Repeat

This implements the key idea from the project README:
"스스로 기록한 방식을 보고 어떠한 trajectory가 있어야 realistic한지 판단한다.
 realistic한 scenario 기반으로 스스로 학습(RL - GRPO)을 시키며 exploration & exploitation을 진행한다."
"""

import json
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training loop."""

    # Model
    model_name_or_path: str = "checkpoints/sft_p2_aggressive_lingshu/merged"
    base_model_for_lora: Optional[str] = None
    backend: str = "transformers"  # "transformers" or "vllm"

    # Trajectory collection
    domains: list[str] = field(
        default_factory=lambda: [
            "clinical_diagnosis",
            "medical_qa",
            "visual_diagnosis",
            "drug_interaction",
            "ehr_management",
        ]
    )
    tasks_per_domain: int = 20  # tasks to collect per domain per iteration
    max_turns: int = 15
    temperature: float = 0.7  # higher for exploration
    num_trajectories_per_task: int = 2  # multiple rollouts for diversity

    # Quality filtering
    quality_threshold: float = 0.6  # min composite score to keep trajectory
    use_llm_judge: bool = True  # use LLM-as-Judge for NL_ASSERTION
    judge_model: str = "self"  # "self" = same model, or path to judge model

    # Training
    training_method: str = "grpo"  # "grpo" (primary) or "sft" (legacy)
    lora_r: int = 16
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 1e-5
    num_train_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 4096  # Training sequence length (auto-tuned or manual)

    # Iteration control
    max_iterations: int = 5
    eval_every: int = 1  # evaluate every N iterations
    min_trajectories_for_training: int = 50

    # Paths
    output_dir: str = "checkpoints/self_play"
    log_dir: str = "logs/self_play"
    trajectory_dir: str = "datasets/self_play_trajectories"


@dataclass
class TrajectoryRecord:
    """A single collected trajectory with quality scores."""

    domain: str
    task_id: str
    messages: list[dict]  # chat messages format
    tool_calls: list[dict]
    total_turns: int
    action_score: float
    reward: float

    # Quality scores from judge
    judge_scores: dict = field(default_factory=dict)
    composite_quality: float = 0.0

    # Metadata
    model_name: str = ""
    iteration: int = 0
    timestamp: str = ""

    def to_sft_example(self) -> dict:
        """Convert to SFT training format."""
        return {
            "messages": self.messages,
            "metadata": {
                "domain": self.domain,
                "task_id": self.task_id,
                "source": "self_play",
                "iteration": self.iteration,
                "action_score": self.action_score,
                "reward": self.reward,
                "composite_quality": self.composite_quality,
                "judge_scores": self.judge_scores,
            },
        }


class TrajectoryJudge:
    """LLM-as-Judge for trajectory quality assessment.

    Evaluates trajectories on multiple dimensions:
    1. Clinical Correctness: Are the medical decisions sound?
    2. Tool Usage Efficiency: Are tools used effectively?
    3. Reasoning Quality: Is the reasoning structured and thorough?
    4. Completeness: Does the trajectory reach a proper conclusion?
    """

    JUDGE_SYSTEM_PROMPT = """You are an expert medical AI evaluator. Your task is to evaluate the quality of a medical agent's trajectory (sequence of actions and reasoning) in a clinical scenario.

Rate the trajectory on each dimension from 0.0 to 1.0:

1. **Clinical Correctness** (0-1): Are the medical decisions, diagnoses, and recommendations clinically sound? Are there any dangerous or incorrect actions?
2. **Tool Usage Efficiency** (0-1): Does the agent use tools effectively? Does it gather necessary information without unnecessary redundancy?
3. **Reasoning Quality** (0-1): Is the reasoning structured, thorough, and well-justified? Does it reference evidence and clinical guidelines?
4. **Completeness** (0-1): Does the trajectory reach a proper clinical conclusion? Are all necessary steps covered?

Respond in JSON format:
{"clinical_correctness": 0.X, "tool_efficiency": 0.X, "reasoning_quality": 0.X, "completeness": 0.X, "explanation": "brief justification"}"""

    def __init__(self, model=None, tokenizer=None, use_heuristic_fallback: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.use_heuristic_fallback = use_heuristic_fallback

    def judge_trajectory(self, trajectory: TrajectoryRecord) -> dict:
        """Judge a trajectory's quality."""
        if self.model is not None and self.tokenizer is not None:
            return self._judge_with_llm(trajectory)
        elif self.use_heuristic_fallback:
            return self._judge_with_heuristics(trajectory)
        else:
            return {
                "clinical_correctness": 0.5,
                "tool_efficiency": 0.5,
                "reasoning_quality": 0.5,
                "completeness": 0.5,
            }

    def _judge_with_llm(self, trajectory: TrajectoryRecord) -> dict:
        """Use LLM to judge trajectory quality."""
        import torch

        # Format trajectory for judge
        traj_text = self._format_trajectory_for_judge(trajectory)

        messages = [
            {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Domain: {trajectory.domain}\nTask: {trajectory.task_id}\n\n{traj_text}",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=300, temperature=0.1, do_sample=True
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )

        # Parse JSON response
        try:
            # Find JSON in response
            import re

            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                scores = json.loads(json_match.group())
                return {
                    "clinical_correctness": float(
                        scores.get("clinical_correctness", 0.5)
                    ),
                    "tool_efficiency": float(scores.get("tool_efficiency", 0.5)),
                    "reasoning_quality": float(scores.get("reasoning_quality", 0.5)),
                    "completeness": float(scores.get("completeness", 0.5)),
                    "explanation": scores.get("explanation", ""),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback to heuristics if LLM response is invalid
        return self._judge_with_heuristics(trajectory)

    def _judge_with_heuristics(self, trajectory: TrajectoryRecord) -> dict:
        """Heuristic-based trajectory quality assessment."""
        scores = {}

        # 1. Clinical Correctness — proxy: action_score from environment
        scores["clinical_correctness"] = trajectory.action_score

        # 2. Tool Usage Efficiency
        tool_names = [tc.get("tool_name", "") for tc in trajectory.tool_calls]
        unique_tools = set(tool_names)

        # Penalize too many repeated calls
        if len(tool_names) > 0:
            diversity_ratio = len(unique_tools) / len(tool_names)
        else:
            diversity_ratio = 0.0

        # Penalize too few or too many turns
        turn_penalty = 1.0
        if trajectory.total_turns <= 1:
            turn_penalty = 0.3  # too few
        elif trajectory.total_turns > 12:
            turn_penalty = max(0.3, 1.0 - (trajectory.total_turns - 12) * 0.05)

        scores["tool_efficiency"] = min(1.0, diversity_ratio * 0.6 + turn_penalty * 0.4)

        # 3. Reasoning Quality — check for think tool, medical terms, structure
        all_text = " ".join(
            m.get("content", "") for m in trajectory.messages if m.get("role") == "assistant"
        )

        has_think = any(tc.get("tool_name") == "think" for tc in trajectory.tool_calls)
        medical_terms = sum(
            1
            for term in [
                "diagnosis",
                "treatment",
                "patient",
                "symptoms",
                "lab",
                "vital",
                "medication",
                "clinical",
                "recommend",
                "evidence",
                "guideline",
                "assessment",
                "differential",
            ]
            if term.lower() in all_text.lower()
        )
        reasoning_score = min(1.0, (0.3 if has_think else 0.0) + medical_terms * 0.05 + 0.2)
        scores["reasoning_quality"] = reasoning_score

        # 4. Completeness — check if trajectory has a final answer/conclusion
        has_submit = any(
            tc.get("tool_name") in ("submit_answer", "answer_visual_question")
            for tc in trajectory.tool_calls
        )
        has_conclusion = any(
            keyword in all_text.lower()
            for keyword in ["conclusion", "recommend", "assessment", "final", "summary"]
        )
        scores["completeness"] = 0.5 + (0.3 if has_submit else 0.0) + (0.2 if has_conclusion else 0.0)

        return scores

    def _format_trajectory_for_judge(self, trajectory: TrajectoryRecord) -> str:
        """Format a trajectory into readable text for the judge."""
        parts = []
        for i, msg in enumerate(trajectory.messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"[{role.upper()}] {content}")
        return "\n\n".join(parts)


class SelfPlayLoop:
    """Orchestrates the self-play training loop.

    Flow:
    1. Initialize model
    2. For each iteration:
       a. Collect trajectories across all domains
       b. Judge trajectory quality
       c. Filter high-quality trajectories
       d. Train on filtered data
       e. Evaluate on benchmarks
       f. Log results
    """

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.iteration = 0
        self.all_trajectories: list[TrajectoryRecord] = []
        self.iteration_stats: list[dict] = []

        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.trajectory_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        """Run the complete self-play training loop."""
        logger.info(f"Starting Self-Play Training Loop")
        logger.info(f"  Model: {self.config.model_name_or_path}")
        logger.info(f"  Domains: {self.config.domains}")
        logger.info(f"  Max iterations: {self.config.max_iterations}")

        current_model_path = self.config.model_name_or_path

        for iteration in range(1, self.config.max_iterations + 1):
            self.iteration = iteration
            logger.info(f"\n{'='*70}")
            logger.info(f"  SELF-PLAY ITERATION {iteration}/{self.config.max_iterations}")
            logger.info(f"  Model: {current_model_path}")
            logger.info(f"{'='*70}")

            # Step 1: Collect trajectories
            logger.info(f"[Step 1] Collecting trajectories...")
            trajectories = self._collect_trajectories(current_model_path)
            logger.info(f"  Collected {len(trajectories)} trajectories")

            # Step 2: Judge quality
            logger.info(f"[Step 2] Judging trajectory quality...")
            self._judge_trajectories(trajectories, current_model_path)

            # Step 3: Filter high-quality
            logger.info(f"[Step 3] Filtering trajectories...")
            filtered = self._filter_trajectories(trajectories)
            logger.info(
                f"  Kept {len(filtered)}/{len(trajectories)} "
                f"({100*len(filtered)/max(len(trajectories),1):.0f}%)"
            )

            self.all_trajectories.extend(filtered)

            # Step 4: Train
            if len(filtered) >= self.config.min_trajectories_for_training:
                logger.info(f"[Step 4] Training on {len(filtered)} trajectories...")
                new_model_path = self._train(filtered, iteration)
                if new_model_path:
                    current_model_path = new_model_path
                    logger.info(f"  New model: {new_model_path}")
            else:
                logger.warning(
                    f"  Not enough trajectories ({len(filtered)} < {self.config.min_trajectories_for_training}). "
                    f"Accumulating (total: {len(self.all_trajectories)})..."
                )
                # Try with accumulated trajectories
                if len(self.all_trajectories) >= self.config.min_trajectories_for_training:
                    logger.info(f"[Step 4] Training on accumulated {len(self.all_trajectories)} trajectories...")
                    new_model_path = self._train(self.all_trajectories, iteration)
                    if new_model_path:
                        current_model_path = new_model_path

            # Step 5: Evaluate
            if iteration % self.config.eval_every == 0:
                logger.info(f"[Step 5] Evaluating...")
                eval_results = self._evaluate(current_model_path, iteration)
                self.iteration_stats.append(
                    {
                        "iteration": iteration,
                        "model_path": current_model_path,
                        "num_trajectories": len(trajectories),
                        "num_filtered": len(filtered),
                        "eval_results": eval_results,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Save progress
            self._save_iteration_log(iteration, trajectories, filtered)

        # Final summary
        self._print_summary()

    def _collect_trajectories(self, model_path: str) -> list[TrajectoryRecord]:
        """Collect trajectories by running the agent in GYM environments."""
        from bioagents.evaluation.agent_runner import AgentRunner, RunConfig

        trajectories = []

        for domain in self.config.domains:
            logger.info(f"  Collecting from {domain}...")

            run_config = RunConfig(
                model_name_or_path=model_path,
                backend=self.config.backend,
                domain=domain,
                max_turns=self.config.max_turns,
                temperature=self.config.temperature,
                log_dir=str(Path(self.config.log_dir) / f"iter_{self.iteration}"),
            )

            try:
                runner = AgentRunner(run_config)
                runner.load_model()
                task_results = runner.run_all_tasks()

                for result in task_results:
                    # Build messages from trajectory
                    messages = self._result_to_messages(result, domain)
                    if not messages or len(messages) < 2:
                        continue  # Skip empty/trivial trajectories

                    traj = TrajectoryRecord(
                        domain=domain,
                        task_id=result.task_id,
                        messages=messages,
                        tool_calls=[
                            {
                                "tool_name": turn.parsed_tool_call.get("name", "") if turn.parsed_tool_call else "",
                                "arguments": turn.parsed_tool_call.get("arguments", {}) if turn.parsed_tool_call else {},
                            }
                            for turn in result.turns
                            if turn.parsed_tool_call
                        ],
                        total_turns=result.total_turns,
                        action_score=result.action_score,
                        reward=result.final_reward,
                        model_name=str(model_path),
                        iteration=self.iteration,
                        timestamp=datetime.now().isoformat(),
                    )
                    trajectories.append(traj)

                # Free model between domains
                del runner
                import torch
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"  Error collecting from {domain}: {e}")
                import traceback
                traceback.print_exc()

        return trajectories

    def _result_to_messages(self, result, domain: str) -> list[dict]:
        """Convert a TaskResult into chat messages format for SFT."""
        messages = []

        # System message
        system_msg = (
            f"You are a medical AI assistant specializing in {domain.replace('_', ' ')}. "
            f"Use the available tools to analyze the case and provide your clinical assessment."
        )
        messages.append({"role": "system", "content": system_msg})

        for turn in result.turns:
            # Agent message (with tool call)
            if turn.parsed_tool_call:
                tool_json = json.dumps(turn.parsed_tool_call, ensure_ascii=False)
                messages.append({"role": "assistant", "content": tool_json})
                # Tool response
                if turn.tool_response:
                    messages.append({"role": "user", "content": f"Tool result: {turn.tool_response}"})
            elif turn.raw_output:
                messages.append({"role": "assistant", "content": turn.raw_output})

        return messages

    def _judge_trajectories(
        self, trajectories: list[TrajectoryRecord], model_path: str
    ):
        """Judge trajectory quality using LLM-as-Judge or heuristics."""
        if self.config.use_llm_judge and self.config.judge_model == "self":
            # Use the same model as judge (self-evaluation)
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model_type = getattr(config, "model_type", "")

                if model_type in ("qwen2_5_vl", "qwen2_vl"):
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
                    )
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                judge = TrajectoryJudge(model=model, tokenizer=tokenizer)

                for traj in trajectories:
                    scores = judge.judge_trajectory(traj)
                    traj.judge_scores = scores
                    traj.composite_quality = sum(scores.values()) / max(len(scores), 1)

                del model, tokenizer
                torch.cuda.empty_cache()
                return

            except Exception as e:
                logger.warning(f"LLM judge failed, falling back to heuristics: {e}")

        # Heuristic fallback
        judge = TrajectoryJudge(use_heuristic_fallback=True)
        for traj in trajectories:
            scores = judge.judge_trajectory(traj)
            traj.judge_scores = scores
            # Weighted composite: clinical_correctness is most important
            weights = {
                "clinical_correctness": 0.35,
                "tool_efficiency": 0.2,
                "reasoning_quality": 0.25,
                "completeness": 0.2,
            }
            traj.composite_quality = sum(
                scores.get(k, 0.5) * w for k, w in weights.items()
            )

    def _filter_trajectories(
        self, trajectories: list[TrajectoryRecord]
    ) -> list[TrajectoryRecord]:
        """Filter trajectories based on quality threshold."""
        filtered = [
            t
            for t in trajectories
            if t.composite_quality >= self.config.quality_threshold
        ]

        # Also ensure diversity: max N per (domain, task_id)
        seen = {}
        diverse_filtered = []
        max_per_task = self.config.num_trajectories_per_task
        for t in sorted(filtered, key=lambda x: x.composite_quality, reverse=True):
            key = (t.domain, t.task_id)
            if seen.get(key, 0) < max_per_task:
                diverse_filtered.append(t)
                seen[key] = seen.get(key, 0) + 1

        return diverse_filtered

    def _train(self, trajectories: list[TrajectoryRecord], iteration: int) -> Optional[str]:
        """Train on filtered trajectories."""
        # Save trajectories as SFT data
        sft_path = Path(self.config.trajectory_dir) / f"iter_{iteration}_sft.jsonl"
        with open(sft_path, "w") as f:
            for t in trajectories:
                f.write(json.dumps(t.to_sft_example(), ensure_ascii=False) + "\n")

        logger.info(f"  Saved {len(trajectories)} SFT examples to {sft_path}")

        if self.config.training_method == "sft":
            return self._train_sft(sft_path, iteration)
        elif self.config.training_method == "grpo":
            return self._train_grpo(trajectories, iteration)
        return None

    def _train_sft(self, sft_path: Path, iteration: int) -> Optional[str]:
        """Run SFT training on collected trajectories."""
        import yaml

        output_dir = str(Path(self.config.output_dir) / f"iter_{iteration}_sft")

        # Generate config
        sft_config = {
            "model": {
                "name_or_path": self.config.model_name_or_path
                if iteration == 1
                else str(Path(self.config.output_dir) / f"iter_{iteration-1}_sft" / "merged"),
                "torch_dtype": "bfloat16",
                "attn_implementation": "sdpa",
            },
            "peft": {
                "enabled": True,
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_r * 2,
                "lora_dropout": 0.05,
                "target_modules": self.config.lora_target_modules,
            },
            "dataset": {
                "sft_path": str(sft_path),
                "min_reward": 0.0,
                "max_samples": len(open(sft_path).readlines()),
                "max_length": self.config.max_length,
                "train_ratio": 0.9,
            },
            "training": {
                "output_dir": output_dir,
                "num_train_epochs": self.config.num_train_epochs,
                "per_device_train_batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "learning_rate": self.config.learning_rate,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "bf16": True,
                "logging_steps": 5,
                "save_steps": 999999,
                "save_total_limit": 1,
                "seed": 42 + iteration,
            },
            "logging": {
                "project": "bioagents-self-play",
                "run_name": f"self_play_iter_{iteration}",
                "use_wandb": False,
                "log_dir": self.config.log_dir,
            },
        }

        config_path = Path(self.config.output_dir) / f"iter_{iteration}_sft_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sft_config, f, default_flow_style=False)

        # Run training
        import subprocess

        cmd = [
            "python",
            "-m",
            "bioagents.training.sft_trainer",
            "--config",
            str(config_path),
        ]

        logger.info(f"  Running SFT: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            # Merge LoRA
            merged_path = str(Path(output_dir) / "merged")
            self._merge_lora(output_dir + "/final", merged_path)
            return merged_path
        else:
            logger.error(f"  SFT training failed: {result.stderr[-500:]}")
            return None

    def _train_grpo(
        self, trajectories: list[TrajectoryRecord], iteration: int
    ) -> Optional[str]:
        """Run GRPO training using multi-turn environment rollouts.

        Instead of learning from curated SFT examples, the model
        interacts with the GYM environment, collects diverse
        trajectories, and learns from group-relative advantages.
        """
        from bioagents.training.grpo_trainer import (
            BioAgentGRPOConfig,
            train_multiturn,
        )
        from bioagents.evaluation.agent_runner import repair_model_config

        # Use most recent model checkpoint
        model_path = (
            self.config.model_name_or_path
            if iteration == 1
            else str(
                Path(self.config.output_dir) / f"iter_{iteration-1}_grpo" / "merged"
            )
        )
        # Fallback to base if previous checkpoint doesn't exist
        if not Path(model_path).exists():
            model_path = self.config.model_name_or_path

        repair_model_config(model_path)

        output_dir = str(
            Path(self.config.output_dir) / f"iter_{iteration}_grpo"
        )

        # Find tasks for the primary domain
        domain = self.config.domains[0] if self.config.domains else "medical_qa"
        tasks_path = str(Path("data/domains") / domain / "tasks.json")
        if not Path(tasks_path).exists():
            logger.warning(f"Tasks not found: {tasks_path}, falling back to SFT")
            sft_path = Path(self.config.trajectory_dir) / f"iter_{iteration}_sft.jsonl"
            return self._train_sft(sft_path, iteration)

        grpo_config = BioAgentGRPOConfig(
            model_name_or_path=model_path,
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
            peft_enabled=True,
            peft_r=self.config.lora_r,
            peft_lora_alpha=self.config.lora_r * 2,
            peft_lora_dropout=0.05,
            domain=domain,
            tasks_path=tasks_path,
            output_dir=output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            bf16=True,
            num_generations=4,
            beta=0.04,
            temperature=0.7,
            max_turns=self.config.max_turns,
            use_gym_env=True,
            use_wandb=False,
            log_dir=self.config.log_dir,
            seed=42 + iteration,
        )

        try:
            collected = train_multiturn(grpo_config)
            logger.info(
                f"GRPO iter {iteration}: {len(collected)} trajectories"
            )
        except Exception as e:
            logger.warning(f"GRPO training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Merge LoRA into base
        best_path = str(Path(output_dir) / "best")
        final_path = str(Path(output_dir) / "final")

        for candidate in [best_path, final_path]:
            if Path(candidate).exists():
                merged_path = str(Path(output_dir) / "merged")
                self._merge_lora(candidate, merged_path)
                if Path(merged_path).exists():
                    return merged_path

        return None

    def _merge_lora(self, adapter_path: str, output_path: str):
        """Merge LoRA adapter into base model using ModelProfile for loading."""
        if Path(output_path).exists() and (Path(output_path) / "config.json").exists():
            logger.info(f"  Merged model already exists: {output_path}")
            return

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
adapter_cfg = json.load(open(Path(adapter_path) / "adapter_config.json"))
base_path = adapter_cfg.get("base_model_name_or_path", "")

# Use ModelProfile for correct model class detection
from bioagents.gym.model_profile import ModelProfiler
profile = ModelProfiler.profile(base_path)
print(f"Base model profile: {{profile.model_name}} ({{profile.model_class}})")

model = profile.load_model(device_map="cpu")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(output_path)

tokenizer = profile.load_tokenizer()
tokenizer.save_pretrained(output_path)

# Copy processor files from base if this is a VL model
if profile.requires_processor:
    import shutil
    for fname in ["preprocessor_config.json", "chat_template.json",
                   "special_tokens_map.json", "added_tokens.json",
                   "merges.txt", "vocab.json"]:
        src = Path(base_path) / fname
        dst = Path(output_path) / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {{fname}} from base model")

print(f"Merged to: {{output_path}}")
"""
        script_path = "/tmp/merge_self_play.py"
        with open(script_path, "w") as f:
            f.write(merge_script)

        subprocess.run(["python", script_path], timeout=600)

    def _evaluate(self, model_path: str, iteration: int) -> dict:
        """Evaluate current model on all domains."""
        from bioagents.evaluation.agent_runner import AgentRunner, RunConfig

        results = {}
        for domain in self.config.domains:
            try:
                run_config = RunConfig(
                    model_name_or_path=model_path,
                    backend=self.config.backend,
                    domain=domain,
                    max_turns=self.config.max_turns,
                    temperature=0.1,  # low temp for eval
                    log_dir=str(
                        Path(self.config.log_dir) / f"iter_{iteration}" / "eval"
                    ),
                )
                runner = AgentRunner(run_config)
                runner.load_model()
                task_results = runner.run_all_tasks()

                avg_action = sum(r.action_score for r in task_results) / max(
                    len(task_results), 1
                )
                avg_reward = sum(r.final_reward for r in task_results) / max(
                    len(task_results), 1
                )

                results[domain] = {
                    "action_score": round(avg_action, 3),
                    "reward": round(avg_reward, 3),
                    "tasks": len(task_results),
                }

                del runner
                import torch
                torch.cuda.empty_cache()

            except Exception as e:
                results[domain] = {"error": str(e)}

        return results

    def _save_iteration_log(
        self,
        iteration: int,
        all_trajectories: list[TrajectoryRecord],
        filtered: list[TrajectoryRecord],
    ):
        """Save iteration log and trajectory data."""
        log_path = Path(self.config.log_dir) / f"iter_{iteration}_summary.json"

        # Compute stats
        quality_scores = [t.composite_quality for t in all_trajectories]
        filtered_scores = [t.composite_quality for t in filtered]

        domain_stats = {}
        for t in all_trajectories:
            if t.domain not in domain_stats:
                domain_stats[t.domain] = {"total": 0, "filtered": 0, "scores": []}
            domain_stats[t.domain]["total"] += 1
            domain_stats[t.domain]["scores"].append(t.composite_quality)

        for t in filtered:
            if t.domain in domain_stats:
                domain_stats[t.domain]["filtered"] += 1

        summary = {
            "iteration": iteration,
            "total_trajectories": len(all_trajectories),
            "filtered_trajectories": len(filtered),
            "filter_rate": len(filtered) / max(len(all_trajectories), 1),
            "avg_quality_all": sum(quality_scores) / max(len(quality_scores), 1),
            "avg_quality_filtered": sum(filtered_scores)
            / max(len(filtered_scores), 1),
            "domain_stats": {
                d: {
                    **s,
                    "avg_quality": sum(s["scores"]) / max(len(s["scores"]), 1),
                }
                for d, s in domain_stats.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"  Iteration log saved to {log_path}")

    def _print_summary(self):
        """Print final summary of all iterations."""
        logger.info(f"\n{'='*70}")
        logger.info(f"  SELF-PLAY TRAINING SUMMARY")
        logger.info(f"{'='*70}")

        for stat in self.iteration_stats:
            logger.info(f"\n  Iteration {stat['iteration']}:")
            logger.info(
                f"    Trajectories: {stat['num_trajectories']} total, {stat['num_filtered']} filtered"
            )
            if "eval_results" in stat:
                for domain, res in stat["eval_results"].items():
                    if "action_score" in res:
                        logger.info(
                            f"    {domain}: action={res['action_score']}, reward={res['reward']}"
                        )

        # Save full stats
        stats_path = Path(self.config.log_dir) / "self_play_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.iteration_stats, f, indent=2)

        logger.info(f"\n  Full stats saved to {stats_path}")


def run_self_play(config_path: Optional[str] = None, **kwargs):
    """Entry point for self-play training."""
    if config_path:
        import yaml

        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = SelfPlayConfig(**cfg_dict)
    else:
        config = SelfPlayConfig(**kwargs)

    loop = SelfPlayLoop(config)
    loop.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-Play Training Loop")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--model", default=None, help="Model path override")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--domains", nargs="+", default=None, help="Domains to use")
    parser.add_argument("--quality-threshold", type=float, default=0.5, help="Min quality to keep")
    parser.add_argument("--min-trajectories", type=int, default=30, help="Min trajectories for training")

    args = parser.parse_args()

    kwargs = {}
    if args.model:
        kwargs["model_name_or_path"] = args.model
    if args.iterations:
        kwargs["max_iterations"] = args.iterations
    if args.domains:
        kwargs["domains"] = args.domains
    if args.quality_threshold:
        kwargs["quality_threshold"] = args.quality_threshold
    if args.min_trajectories:
        kwargs["min_trajectories_for_training"] = args.min_trajectories

    run_self_play(config_path=args.config, **kwargs)
