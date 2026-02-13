"""BIOAgents SFT Trainer — Supervised Fine-Tuning with TRL.

Trains medical agents using expert demonstrations:
1. Trajectory-based SFT: Learn from successful agent trajectories
2. Direct QA SFT: Learn from medical QA with ideal tool-use sequences
3. Instruction SFT: Learn from medical instruction data

Usage:
    python bioagents/training/sft_trainer.py --config configs/sft_medical_qa.yaml
    accelerate launch bioagents/training/sft_trainer.py --config configs/sft_medical_qa.yaml
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


@dataclass
class BioAgentSFTConfig:
    """SFT training configuration."""

    # Model
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # PEFT
    peft_enabled: bool = True
    peft_r: int = 16
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    peft_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Dataset
    trajectory_dir: str = ""
    qa_tasks_path: str = "data/domains/medical_qa/tasks.json"
    instruction_path: str = ""
    sft_path: str = ""  # Pre-generated SFT JSONL file (takes precedence)
    min_reward: float = 0.5
    max_samples: int = 5000
    max_length: int = 4096
    train_ratio: float = 0.9

    # Training
    output_dir: str = "checkpoints/sft_medical_qa"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42

    # Logging
    wandb_project: str = "bioagents-sft"
    run_name: str = "sft_medical_qa"
    use_wandb: bool = True
    log_dir: str = "logs/runs"

    @classmethod
    def from_yaml(cls, path: str) -> "BioAgentSFTConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        kwargs = {}
        if "model" in raw:
            kwargs["model_name_or_path"] = raw["model"].get("name_or_path", cls.model_name_or_path)
            kwargs["torch_dtype"] = raw["model"].get("torch_dtype", cls.torch_dtype)
            kwargs["attn_implementation"] = raw["model"].get("attn_implementation", cls.attn_implementation)
        if "peft" in raw:
            kwargs["peft_enabled"] = raw["peft"].get("enabled", cls.peft_enabled)
            kwargs["peft_r"] = raw["peft"].get("r", cls.peft_r)
            kwargs["peft_lora_alpha"] = raw["peft"].get("lora_alpha", cls.peft_lora_alpha)
            kwargs["peft_lora_dropout"] = raw["peft"].get("lora_dropout", cls.peft_lora_dropout)
            kwargs["peft_target_modules"] = raw["peft"].get("target_modules", [])
        if "dataset" in raw:
            d = raw["dataset"]
            for key in [
                "trajectory_dir", "qa_tasks_path", "instruction_path",
                "sft_path", "min_reward", "max_samples", "max_length", "train_ratio",
            ]:
                if key in d:
                    kwargs[key] = d[key]
        if "training" in raw:
            t = raw["training"]
            for key in [
                "output_dir", "num_train_epochs", "per_device_train_batch_size",
                "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
                "warmup_ratio", "weight_decay", "max_grad_norm", "bf16",
                "logging_steps", "save_steps", "eval_steps", "save_total_limit", "seed",
            ]:
                if key in t:
                    kwargs[key] = t[key]
        if "logging" in raw:
            kwargs["wandb_project"] = raw["logging"].get("project", cls.wandb_project)
            kwargs["run_name"] = raw["logging"].get("run_name", cls.run_name)
            kwargs["use_wandb"] = raw["logging"].get("use_wandb", cls.use_wandb)
            kwargs["log_dir"] = raw["logging"].get("log_dir", cls.log_dir)

        return cls(**kwargs)


def build_sft_dataset(config: BioAgentSFTConfig):
    """Build SFT dataset from multiple sources.

    Returns:
        Tuple of (train_dataset, eval_dataset) as HuggingFace Datasets
    """
    from datasets import Dataset

    from bioagents.data_pipeline.sft_generator import (
        trajectory_to_sft,
        qa_tasks_to_sft,
        instruction_to_sft,
    )

    all_examples = []

    # 0. Load from pre-generated SFT file (highest priority)
    if config.sft_path:
        sft_file = Path(config.sft_path)
        if sft_file.exists():
            with open(sft_file, "r", encoding="utf-8") as f:
                if str(sft_file).endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            all_examples.append(json.loads(line))
                else:
                    all_examples = json.load(f)
            logger.info(f"  → {len(all_examples)} examples from pre-generated SFT file: {sft_file}")
        else:
            logger.warning(f"  SFT file not found: {sft_file}")

    # 1. Load from trajectory directory (if no sft_path or additional data needed)
    if config.trajectory_dir:
        traj_dir = Path(config.trajectory_dir)
        if traj_dir.exists():
            traj_files = list(traj_dir.glob("*.json"))
            logger.info(f"Found {len(traj_files)} trajectory files")
            for tf in traj_files:
                examples = trajectory_to_sft(
                    str(tf), min_reward=config.min_reward,
                )
                all_examples.extend(examples)
            logger.info(f"  → {len(all_examples)} examples from trajectories")

    # 2. Load from QA tasks (synthetic expert demonstrations)
    if config.qa_tasks_path:
        qa_path = Path(config.qa_tasks_path)
        if qa_path.exists():
            with open(qa_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            qa_examples = qa_tasks_to_sft(tasks, include_reasoning=True)
            all_examples.extend(qa_examples)
            logger.info(f"  → {len(qa_examples)} examples from QA tasks")

    # 3. Load from instruction data
    if config.instruction_path:
        inst_path = Path(config.instruction_path)
        if inst_path.exists():
            with open(inst_path, "r", encoding="utf-8") as f:
                if str(inst_path).endswith(".jsonl"):
                    instructions = [json.loads(line) for line in f if line.strip()]
                else:
                    instructions = json.load(f)
            inst_examples = instruction_to_sft(instructions)
            all_examples.extend(inst_examples)
            logger.info(f"  → {len(inst_examples)} examples from instructions")

    if not all_examples:
        raise ValueError("No SFT examples found from any source!")

    # Limit total samples
    if len(all_examples) > config.max_samples:
        import random
        random.seed(config.seed)
        all_examples = random.sample(all_examples, config.max_samples)

    logger.info(f"Total SFT examples: {len(all_examples)}")

    # Convert to HuggingFace Dataset
    # TRL SFTTrainer expects "messages" as a list of message dicts
    records = []
    for ex in all_examples:
        records.append({
            "messages": ex["messages"],  # Keep as list of dicts for TRL
        })

    dataset = Dataset.from_list(records)

    # Split
    split_idx = int(len(dataset) * config.train_ratio)
    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset))) if split_idx < len(dataset) else None

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")
    return train_dataset, eval_dataset


def train(config: BioAgentSFTConfig):
    """Run SFT training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    logger.info("=" * 60)
    logger.info("BIOAgents SFT Trainer")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Output: {config.output_dir}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    load_kwargs = dict(
        torch_dtype=dtype_map.get(config.torch_dtype, torch.bfloat16),
        trust_remote_code=True,
    )

    # Detect model type to choose correct Auto class
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    architectures = getattr(model_config, "architectures", [])
    auto_map = getattr(model_config, "auto_map", {})
    uses_custom_auto = "AutoModelForCausalLM" in auto_map

    is_vl_model = any(
        "vl" in a.lower() or "vision" in a.lower()
        for a in (architectures or [])
    ) or "vl" in model_type.lower()
    is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl") or ("qwen2" in model_type.lower() and "vl" in model_type.lower())

    logger.info(f"Model type: {model_type}, VL: {is_vl_model}, Qwen-VL: {is_qwen_vl}")

    if is_qwen_vl:
        # Qwen2.5-VL models (including Lingshu-7B) need specific class
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path, **load_kwargs,
        )
    elif uses_custom_auto:
        # Custom model with auto_map
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, **load_kwargs,
        )
    else:
        # Standard CausalLM
        if config.attn_implementation and config.attn_implementation != "sdpa":
            load_kwargs["attn_implementation"] = config.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, **load_kwargs,
        )

    # --- Gradient Checkpointing (saves ~40% VRAM) ---
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    logger.info("Gradient checkpointing enabled")

    # --- PEFT ---
    peft_config = None
    if config.peft_enabled:
        from peft import LoraConfig, TaskType

        peft_config = LoraConfig(
            r=config.peft_r,
            lora_alpha=config.peft_lora_alpha,
            lora_dropout=config.peft_lora_dropout,
            target_modules=config.peft_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    # --- Dataset ---
    train_dataset, eval_dataset = build_sft_dataset(config)

    # --- SFT Config (optimized for A100 80GB) ---
    os.makedirs(config.output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        max_length=config.max_length,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.run_name,
        logging_dir=config.log_dir,
        # --- Performance optimizations ---
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",  # Fused optimizer (faster on A100)
        torch_compile=False,  # Disable for compatibility
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # --- Train ---
    logger.info("Starting SFT training...")
    logger.info(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Optimizations: gradient_checkpointing, fused_adam, pin_memory, 4 workers")
    trainer.train()

    # Save
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))
    logger.info(f"✅ SFT training complete! Model saved to {config.output_dir}/final")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="BIOAgents SFT Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to SFT config YAML")
    parser.add_argument("--dry_run", action="store_true", help="Build dataset without training")
    args = parser.parse_args()

    config = BioAgentSFTConfig.from_yaml(args.config)
    logger.info(f"Loaded config from {args.config}")

    if args.dry_run:
        train_ds, eval_ds = build_sft_dataset(config)
        sample = train_ds[0]["messages"]
        logger.info(f"Sample messages ({len(sample)} turns):")
        for msg in sample[:3]:
            logger.info(f"  [{msg['role']}]: {msg['content'][:100]}...")
        logger.info("✅ Dry run complete!")
        return

    train(config)


if __name__ == "__main__":
    main()
