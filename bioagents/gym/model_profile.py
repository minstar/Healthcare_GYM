"""Model Profile System -- Self-Declaration of Model Capabilities.

When a model "enters" the Autonomous GYM, it doesn't just bring weights.
It brings a complete profile of WHO it is and WHAT it can do.

Think of it as a gym membership card:
- What kind of exercises can you do? (modalities)
- What equipment do you need? (model_class, processor)
- How much space do you take? (memory)
- What's your training history? (base_model, fine_tune_history)

The ModelProfiler auto-detects all of this by inspecting the model
directory -- config.json, tokenizer_config.json, preprocessor_config.json, etc.

Usage:
    from bioagents.gym.model_profile import ModelProfiler

    # Auto-detect model capabilities
    profile = ModelProfiler.profile("/path/to/model")

    # Check what this model can do
    print(profile.modalities)        # ["text", "image"]
    print(profile.supported_domains) # ["clinical_diagnosis", "visual_diagnosis", ...]
    print(profile.model_class)       # "Qwen2_5_VLForConditionalGeneration"

    # Use loading instructions
    model = profile.load_model(device_map="auto")
    tokenizer = profile.load_tokenizer()
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("bioagents.gym.model_profile")


# ============================================================
# Domain <-> Modality Mapping
# ============================================================

# Domains that require image/video understanding
VISUAL_DOMAINS = {"visual_diagnosis", "radiology_report"}

# Domains that are text-only
TEXT_DOMAINS = {
    "clinical_diagnosis",
    "drug_interaction",
    "ehr_management",
    "medical_qa",
    "triage_emergency",
    "cross_domain",
    "psychiatry",
    "obstetrics",
}

ALL_DOMAINS = TEXT_DOMAINS | VISUAL_DOMAINS


# ============================================================
# Known Model Architectures Registry
# ============================================================

@dataclass
class ArchitectureSpec:
    """Specification for a known model architecture."""
    model_class: str                   # e.g., "AutoModelForCausalLM"
    model_import_path: str             # e.g., "transformers"
    tokenizer_class: str               # e.g., "AutoTokenizer"
    requires_processor: bool = False
    processor_class: str = ""          # e.g., "AutoProcessor"
    modalities: tuple = ("text",)
    supports_sft: bool = True
    supports_grpo: bool = True
    supports_lora: bool = True
    load_kwargs: dict = field(default_factory=dict)


# Registry of known architectures and their loading instructions
ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec] = {
    # === Standard CausalLM Models ===
    "Qwen3ForCausalLM": ArchitectureSpec(
        model_class="AutoModelForCausalLM",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        modalities=("text",),
    ),
    "Qwen2ForCausalLM": ArchitectureSpec(
        model_class="AutoModelForCausalLM",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        modalities=("text",),
    ),
    "Qwen2MoeForCausalLM": ArchitectureSpec(
        model_class="AutoModelForCausalLM",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        modalities=("text",),
    ),
    "LlamaForCausalLM": ArchitectureSpec(
        model_class="AutoModelForCausalLM",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        modalities=("text",),
    ),
    "MistralForCausalLM": ArchitectureSpec(
        model_class="AutoModelForCausalLM",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        modalities=("text",),
    ),
    "Gemma2ForCausalLM": ArchitectureSpec(
        model_class="AutoModelForCausalLM",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        modalities=("text",),
    ),

    # === Vision-Language Models ===
    "Qwen2_5_VLForConditionalGeneration": ArchitectureSpec(
        model_class="Qwen2_5_VLForConditionalGeneration",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        requires_processor=True,
        processor_class="AutoProcessor",
        modalities=("text", "image", "video"),
        load_kwargs={"use_cache": False},
    ),
    "Qwen2VLForConditionalGeneration": ArchitectureSpec(
        model_class="Qwen2VLForConditionalGeneration",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        requires_processor=True,
        processor_class="AutoProcessor",
        modalities=("text", "image", "video"),
        load_kwargs={"use_cache": False},
    ),
    "LlavaForConditionalGeneration": ArchitectureSpec(
        model_class="LlavaForConditionalGeneration",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        requires_processor=True,
        processor_class="AutoProcessor",
        modalities=("text", "image"),
    ),
    "InternVLChatModel": ArchitectureSpec(
        model_class="AutoModelForCausalLM",
        model_import_path="transformers",
        tokenizer_class="AutoTokenizer",
        requires_processor=True,
        processor_class="AutoProcessor",
        modalities=("text", "image"),
        load_kwargs={"trust_remote_code": True},
    ),
}

# Map model_type -> default architecture name (fallback)
MODEL_TYPE_TO_ARCH: dict[str, str] = {
    "qwen3": "Qwen3ForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
}


# ============================================================
# Model Profile
# ============================================================

@dataclass
class ModelProfile:
    """Self-declared profile of a model's capabilities.

    This is the model's "gym membership card" -- everything the GYM
    needs to know to properly load, evaluate, and train this model.
    """

    # --- Identity ---
    model_path: str
    model_name: str = ""         # Human-readable name
    model_type: str = "unknown"  # HF model_type (e.g., "qwen3", "qwen2_5_vl")
    architecture: str = "unknown"  # Full architecture class name
    base_model: str = ""         # Base model if this is a fine-tune

    # --- Capabilities ---
    modalities: list[str] = field(default_factory=lambda: ["text"])
    supported_domains: list[str] = field(default_factory=list)
    is_vl_model: bool = False

    # --- Loading Instructions ---
    model_class: str = "AutoModelForCausalLM"
    model_import_path: str = "transformers"
    tokenizer_class: str = "AutoTokenizer"
    requires_processor: bool = False
    processor_class: str = ""
    has_processor_config: bool = False
    load_kwargs: dict = field(default_factory=dict)

    # --- Resource Requirements ---
    num_parameters: int = 0
    estimated_memory_gb: float = 0.0
    num_layers: int = 0
    hidden_size: int = 0
    vocab_size: int = 0

    # --- Training Compatibility ---
    supports_sft: bool = True
    supports_grpo: bool = True
    supports_lora: bool = True
    torch_dtype: str = "bfloat16"

    # --- Optimal Runtime Parameters (auto-tuned per GPU) ---
    # These are computed by compute_optimal_params() based on
    # model size and available GPU memory.
    optimal_params: dict = field(default_factory=dict)

    # --- File Inventory ---
    has_config: bool = False
    has_tokenizer: bool = False
    has_chat_template: bool = False
    model_files: list[str] = field(default_factory=list)

    # --- Validation ---
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    def can_handle_domain(self, domain: str) -> bool:
        """Check if this model can handle a specific domain."""
        if domain in VISUAL_DOMAINS:
            return self.is_vl_model and "image" in self.modalities
        return True  # Text-only domains are always supported

    def get_compatible_domains(self, available_domains: list[str]) -> list[str]:
        """Filter a list of domains to only those this model can handle."""
        return [d for d in available_domains if self.can_handle_domain(d)]

    def load_model(self, **extra_kwargs):
        """Load the model using the profile's instructions.

        Returns:
            Loaded model (on GPU with eval mode)
        """
        import torch

        # Merge loading kwargs
        kw = {
            "torch_dtype": getattr(torch, self.torch_dtype, torch.bfloat16),
            "trust_remote_code": True,
            **self.load_kwargs,
            **extra_kwargs,
        }

        # Get the model class
        model_cls = self._resolve_class(self.model_class, self.model_import_path)

        logger.info(
            f"[ModelProfile] Loading {self.model_name} with "
            f"{self.model_class} (VL={self.is_vl_model})"
        )

        model = model_cls.from_pretrained(self.model_path, **kw)
        model.eval()
        return model

    def load_tokenizer(self):
        """Load the tokenizer using the profile's instructions."""
        tokenizer_cls = self._resolve_class(self.tokenizer_class, "transformers")
        return tokenizer_cls.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def load_processor(self):
        """Load the processor (for VL models).

        Falls back to loading from base model if processor config is missing.
        """
        if not self.requires_processor:
            return None

        processor_cls = self._resolve_class(
            self.processor_class or "AutoProcessor", "transformers"
        )

        # Try loading from model path first
        try:
            return processor_cls.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(
                f"[ModelProfile] Processor not found at {self.model_path}: {e}"
            )

        # Fallback: try base model path
        if self.base_model and Path(self.base_model).exists():
            try:
                logger.info(
                    f"[ModelProfile] Loading processor from base: {self.base_model}"
                )
                return processor_cls.from_pretrained(
                    self.base_model, trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"[ModelProfile] Base processor also failed: {e}")

        return None

    def _resolve_class(self, class_name: str, module_path: str):
        """Dynamically import and return a class."""
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    # ------------------------------------------------------------------
    # GPU-Aware Automatic Parameter Tuning
    # ------------------------------------------------------------------

    def compute_optimal_params(
        self,
        gpu_memory_gb: float = 80.0,
        num_gpus: int = 1,
        memory_safety_margin: float = 0.90,
    ) -> dict:
        """Compute optimal batch sizes and sequence lengths for this model.

        Estimates how to **maximally utilise** available GPU memory based on
        the model's own size and architecture.  The heuristics here are
        deliberately conservative at first and will be tightened as we
        accumulate runtime telemetry (OOM logs, actual VRAM usage, etc.).

        Parameters
        ----------
        gpu_memory_gb : float
            Total VRAM per GPU in GB (default 80 for A100-80G).
        num_gpus : int
            Number of GPUs allocated for this agent's current phase.
        memory_safety_margin : float
            Fraction of total VRAM we allow ourselves to use (0-1).
            Default 0.90 leaves a ~10 % buffer for CUDA overhead, etc.

        Returns
        -------
        dict  (also stored on ``self.optimal_params``)
            Keys:
            - inference_batch_size      : int
            - inference_max_new_tokens  : int
            - inference_max_length      : int
            - train_batch_size          : int
            - train_max_length          : int
            - gradient_accumulation_steps : int
            - effective_train_batch_size : int  (batch * grad_accum)
            - gpu_memory_utilization    : float (for vLLM / HF)
            - lora_r                    : int
            - dataloader_num_workers    : int
        """
        model_mem = self.estimated_memory_gb or self._estimate_model_memory()
        usable_gb = gpu_memory_gb * memory_safety_margin * num_gpus
        free_gb = max(usable_gb - model_mem, 1.0)

        # ---- Model-size tier classification ----
        param_b = (self.num_parameters or 0) / 1e9  # in billions
        if param_b <= 0:
            # fallback: estimate from memory (~2 bytes per param in bf16)
            param_b = model_mem / 2.0

        # ---- VL models need ~30 % extra memory for vision encoder + image tokens
        vl_overhead = 1.30 if self.is_vl_model else 1.0

        # ---- Compute per-sample memory budget (rough) ----
        # For bf16 with LoRA:
        #   Activation memory per token per layer ≈:
        #     QKV projections (3 * hidden * 2 bytes)
        #     + Attention scores (seq * heads * 2 bytes)
        #     + FFN intermediates (4 * hidden * 2 bytes)
        #     + input/output residuals
        #   With gradient checkpointing (~3x savings), multiply by ~10x overhead
        hidden = self.hidden_size or 4096
        layers = self.num_layers or 32

        # bytes per token per sample during training (accounts for activations,
        # gradients, and grad checkpointing)
        # Factor 10: empirically matched to actual OOM boundaries for LoRA on A100
        bytes_per_token = hidden * layers * 2 * 10  # 2 bytes bf16 * ~10x activation overhead
        bytes_per_token *= vl_overhead

        # ---- Inference parameters ----
        # KV cache per token ≈ 2 * hidden * layers * 2 bytes
        kv_bytes_per_token = 2 * hidden * layers * 2
        kv_bytes_per_token *= vl_overhead

        # Inference: we can fit more because no gradients / optimizer
        # Available for KV cache = free memory (beyond model)
        inference_available_gb = free_gb * 0.85  # leave some for temp tensors
        inference_available_bytes = inference_available_gb * (1024 ** 3)

        # Target inference context length
        if param_b <= 3:
            target_inf_ctx = 8192
        elif param_b <= 10:
            target_inf_ctx = 4096
        elif param_b <= 30:
            target_inf_ctx = 2048
        else:
            target_inf_ctx = 1024

        # Max inference batch that fits
        per_sample_inf_bytes = kv_bytes_per_token * target_inf_ctx
        inf_batch = max(1, int(inference_available_bytes / per_sample_inf_bytes))
        # Cap and round down to multiple of 4 (tensor core alignment)
        inf_batch = min(inf_batch, 64)
        inf_batch = max(4, (inf_batch // 4) * 4)

        # max_new_tokens: scale with available context
        if param_b <= 10:
            inf_max_new_tokens = 2048
        else:
            inf_max_new_tokens = 1024

        # ---- Training parameters (LoRA SFT) ----
        # Training needs ~2-3x more memory than inference due to:
        #   - LoRA adapters + optimizer states (~25 % of free)
        #   - Activation checkpointing helps but not modelled here
        lora_overhead_gb = model_mem * 0.15  # LoRA ~10-15 % of model
        train_available_gb = free_gb - lora_overhead_gb
        train_available_bytes = max(train_available_gb, 0.5) * (1024 ** 3)

        # Target training context length
        if param_b <= 3:
            target_train_ctx = 4096
        elif param_b <= 10:
            target_train_ctx = 2048
        elif param_b <= 30:
            target_train_ctx = 1024
        else:
            target_train_ctx = 512

        # VL models: vision tokens can be very long
        if self.is_vl_model:
            target_train_ctx = min(target_train_ctx, 2048)
            target_inf_ctx = min(target_inf_ctx, 4096)

        per_sample_train_bytes = bytes_per_token * target_train_ctx
        train_batch = max(1, int(train_available_bytes / per_sample_train_bytes))
        # Cap and round down to multiple of 4 (tensor core alignment)
        train_batch = min(train_batch, 32)
        train_batch = max(4, (train_batch // 4) * 4)

        # Gradient accumulation: target effective batch ~16-32
        target_effective_batch = 16
        if train_batch >= target_effective_batch:
            grad_accum = 1
        else:
            # Ceiling division to reach at least target_effective_batch
            grad_accum = max(1, -(-target_effective_batch // train_batch))
        effective_batch = train_batch * grad_accum

        # LoRA rank: bigger models get smaller rank to save memory
        if param_b <= 3:
            lora_r = 32
        elif param_b <= 10:
            lora_r = 16
        elif param_b <= 30:
            lora_r = 8
        else:
            lora_r = 4

        # GPU memory utilization for vLLM/HF
        gpu_mem_util = min(memory_safety_margin, 0.92)

        # Dataloader workers
        dl_workers = min(4, max(1, int(free_gb / 5)))

        params = {
            # ---- Inference ----
            "inference_batch_size": inf_batch,
            "inference_max_new_tokens": inf_max_new_tokens,
            "inference_max_length": target_inf_ctx,
            # ---- Training ----
            "train_batch_size": train_batch,
            "train_max_length": target_train_ctx,
            "gradient_accumulation_steps": grad_accum,
            "effective_train_batch_size": effective_batch,
            # ---- Resource hints ----
            "gpu_memory_utilization": round(gpu_mem_util, 2),
            "lora_r": lora_r,
            "dataloader_num_workers": dl_workers,
            # ---- Metadata ----
            "_model_memory_gb": round(model_mem, 1),
            "_usable_gpu_gb": round(usable_gb, 1),
            "_free_after_model_gb": round(free_gb, 1),
            "_param_billions": round(param_b, 2),
        }

        self.optimal_params = params
        logger.info(
            f"[ModelProfile] {self.model_name}: optimal params computed -- "
            f"inf_batch={inf_batch}, train_batch={train_batch}, "
            f"grad_accum={grad_accum}, train_ctx={target_train_ctx}, "
            f"inf_ctx={target_inf_ctx}, lora_r={lora_r}, "
            f"free_gb={free_gb:.1f}"
        )
        return params

    def _estimate_model_memory(self) -> float:
        """Fallback model memory estimation from num_parameters."""
        if self.num_parameters > 0:
            # bf16: 2 bytes per param
            return (self.num_parameters * 2) / (1024 ** 3)
        # Very rough fallback from layer count
        if self.num_layers > 0 and self.hidden_size > 0:
            # Approximate: params ≈ 12 * hidden^2 * layers
            approx_params = 12 * (self.hidden_size ** 2) * self.num_layers
            return (approx_params * 2) / (1024 ** 3)
        return 16.0  # Safe default for ~8B model

    def get_optimal_param(self, key: str, default=None):
        """Retrieve a single optimal parameter by key.

        Convenience method so callers don't need to know the dict structure.
        """
        return self.optimal_params.get(key, default)

    def summary(self) -> str:
        """Return a human-readable summary of this profile."""
        lines = [
            f"{'=' * 60}",
            f"  Model Profile: {self.model_name}",
            f"{'=' * 60}",
            f"  Path:          {self.model_path}",
            f"  Type:          {self.model_type}",
            f"  Architecture:  {self.architecture}",
            f"  Modalities:    {', '.join(self.modalities)}",
            f"  VL Model:      {self.is_vl_model}",
            f"  Model Class:   {self.model_class}",
            f"  Processor:     {'Required' if self.requires_processor else 'Not needed'}",
            f"  Memory:        ~{self.estimated_memory_gb:.1f} GB",
            f"  Layers:        {self.num_layers}",
            f"  Hidden Size:   {self.hidden_size}",
            f"  Vocab Size:    {self.vocab_size}",
            f"  Dtype:         {self.torch_dtype}",
            f"  SFT:           {'Yes' if self.supports_sft else 'No'}",
            f"  GRPO:          {'Yes' if self.supports_grpo else 'No'}",
            f"  LoRA:          {'Yes' if self.supports_lora else 'No'}",
            f"  Valid:         {'YES' if self.is_valid else 'NO'}",
        ]
        if self.supported_domains:
            lines.append(f"  Domains:       {', '.join(self.supported_domains)}")
        if self.validation_errors:
            lines.append(f"  ERRORS:")
            for err in self.validation_errors:
                lines.append(f"    - {err}")
        if self.validation_warnings:
            lines.append(f"  WARNINGS:")
            for warn in self.validation_warnings:
                lines.append(f"    - {warn}")
        if self.optimal_params:
            lines.append(f"  --- Optimal Parameters (auto-tuned) ---")
            op = self.optimal_params
            lines.append(
                f"  Inference:     batch={op.get('inference_batch_size')}, "
                f"ctx={op.get('inference_max_length')}, "
                f"max_new={op.get('inference_max_new_tokens')}"
            )
            lines.append(
                f"  Training:      batch={op.get('train_batch_size')}, "
                f"ctx={op.get('train_max_length')}, "
                f"grad_accum={op.get('gradient_accumulation_steps')}, "
                f"eff_batch={op.get('effective_train_batch_size')}"
            )
            lines.append(
                f"  Resources:     lora_r={op.get('lora_r')}, "
                f"gpu_util={op.get('gpu_memory_utilization')}, "
                f"workers={op.get('dataloader_num_workers')}"
            )
            lines.append(
                f"  Budget:        model={op.get('_model_memory_gb')}GB, "
                f"usable={op.get('_usable_gpu_gb')}GB, "
                f"free={op.get('_free_after_model_gb')}GB"
            )
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON/YAML persistence)."""
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "base_model": self.base_model,
            "modalities": self.modalities,
            "supported_domains": self.supported_domains,
            "is_vl_model": self.is_vl_model,
            "model_class": self.model_class,
            "tokenizer_class": self.tokenizer_class,
            "requires_processor": self.requires_processor,
            "processor_class": self.processor_class,
            "estimated_memory_gb": self.estimated_memory_gb,
            "num_parameters": self.num_parameters,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "torch_dtype": self.torch_dtype,
            "supports_sft": self.supports_sft,
            "supports_grpo": self.supports_grpo,
            "supports_lora": self.supports_lora,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "optimal_params": self.optimal_params,
        }


# ============================================================
# Model Profiler -- Auto-Detection
# ============================================================

class ModelProfiler:
    """Auto-detect model capabilities by inspecting model files.

    This is the "gym receptionist" that examines a model when it
    first walks through the door, checking its ID and capabilities.
    """

    @staticmethod
    def profile(
        model_path: str,
        available_domains: list[str] | None = None,
        base_model_hint: str = "",
    ) -> ModelProfile:
        """Create a ModelProfile by inspecting the model directory.

        Args:
            model_path: Path to model directory
            available_domains: List of domains available in the gym.
                Profile will filter to compatible ones.
            base_model_hint: Optional hint about the base model path
                (useful for merged checkpoints missing processor files)

        Returns:
            ModelProfile with auto-detected capabilities
        """
        model_path = str(Path(model_path).resolve())
        model_dir = Path(model_path)
        profile = ModelProfile(
            model_path=model_path,
            model_name=model_dir.name,
        )

        errors = []
        warnings = []

        # === Step 1: Check directory exists ===
        if not model_dir.exists():
            errors.append(f"Model path does not exist: {model_path}")
            profile.validation_errors = errors
            return profile

        # === Step 2: Inventory files ===
        profile.model_files = [f.name for f in model_dir.iterdir() if f.is_file()]
        profile.has_config = "config.json" in profile.model_files
        profile.has_tokenizer = (
            "tokenizer.json" in profile.model_files
            or "tokenizer_config.json" in profile.model_files
        )
        profile.has_processor_config = (
            "preprocessor_config.json" in profile.model_files
        )
        profile.has_chat_template = (
            "chat_template.json" in profile.model_files
            or "chat_template.jinja" in profile.model_files
        )

        if not profile.has_config:
            errors.append("Missing config.json")
            profile.validation_errors = errors
            return profile

        if not profile.has_tokenizer:
            errors.append("Missing tokenizer files")

        # === Step 3: Parse config.json ===
        with open(model_dir / "config.json") as f:
            config = json.load(f)

        profile.model_type = config.get("model_type", "unknown")
        architectures = config.get("architectures", [])
        profile.architecture = architectures[0] if architectures else "unknown"

        # Extract model dimensions
        # Handle both flat and nested (text_config) structures
        text_config = config.get("text_config", config)
        profile.hidden_size = text_config.get("hidden_size", 0)
        profile.num_layers = text_config.get("num_hidden_layers", 0)
        profile.vocab_size = config.get("vocab_size", text_config.get("vocab_size", 0))
        profile.torch_dtype = config.get("torch_dtype", "bfloat16")

        # Estimate parameter count and memory
        if profile.hidden_size > 0 and profile.num_layers > 0:
            intermediate = text_config.get("intermediate_size", profile.hidden_size * 4)
            # Rough estimate: params ≈ layers * (4*h^2 + 2*h*intermediate) + vocab*h
            h = profile.hidden_size
            params = (
                profile.num_layers * (4 * h * h + 2 * h * intermediate)
                + profile.vocab_size * h * 2
            )
            profile.num_parameters = int(params)
            # bfloat16 = 2 bytes per param
            profile.estimated_memory_gb = round(params * 2 / (1024 ** 3), 1)

        # === Step 4: Determine capabilities from architecture ===
        arch_spec = ARCHITECTURE_REGISTRY.get(profile.architecture)

        if arch_spec is None:
            # Try fallback via model_type
            fallback_arch = MODEL_TYPE_TO_ARCH.get(profile.model_type)
            if fallback_arch:
                arch_spec = ARCHITECTURE_REGISTRY.get(fallback_arch)
                warnings.append(
                    f"Unknown architecture '{profile.architecture}', "
                    f"using fallback from model_type '{profile.model_type}'"
                )
            else:
                # Truly unknown -- try generic AutoModel
                warnings.append(
                    f"Unknown architecture '{profile.architecture}' and "
                    f"model_type '{profile.model_type}'. Using AutoModelForCausalLM."
                )
                arch_spec = ArchitectureSpec(
                    model_class="AutoModelForCausalLM",
                    model_import_path="transformers",
                    tokenizer_class="AutoTokenizer",
                    modalities=("text",),
                )

        # Apply architecture spec to profile
        profile.model_class = arch_spec.model_class
        profile.model_import_path = arch_spec.model_import_path
        profile.tokenizer_class = arch_spec.tokenizer_class
        profile.requires_processor = arch_spec.requires_processor
        profile.processor_class = arch_spec.processor_class
        profile.modalities = list(arch_spec.modalities)
        profile.is_vl_model = len(arch_spec.modalities) > 1
        profile.supports_sft = arch_spec.supports_sft
        profile.supports_grpo = arch_spec.supports_grpo
        profile.supports_lora = arch_spec.supports_lora
        profile.load_kwargs = dict(arch_spec.load_kwargs)

        # === Step 5: Check for vision config (confirms VL capability) ===
        has_vision_config = "vision_config" in config
        if profile.is_vl_model and not has_vision_config:
            warnings.append(
                "Detected as VL model from architecture but no vision_config "
                "found in config.json. Visual capabilities may be limited."
            )

        # === Step 6: Handle processor availability for VL models ===
        if profile.requires_processor and not profile.has_processor_config:
            # VL model without processor config -- common in merged checkpoints
            warnings.append(
                "VL model is missing preprocessor_config.json. "
                "Will attempt to load processor from base model."
            )

            # Try to detect base model
            base_model = base_model_hint
            if not base_model:
                # Check for adapter_config.json (LoRA merges)
                adapter_config_path = model_dir / "adapter_config.json"
                if adapter_config_path.exists():
                    with open(adapter_config_path) as f:
                        adapter_cfg = json.load(f)
                    base_model = adapter_cfg.get("base_model_name_or_path", "")

                # Check common patterns in config
                if not base_model:
                    auto_map = config.get("auto_map", {})
                    for key, val in auto_map.items():
                        if "--" in str(val):
                            # format: "org/model--module.py"
                            candidate = str(val).split("--")[0]
                            if candidate:
                                base_model = candidate
                                break

            if base_model and Path(base_model).exists():
                profile.base_model = base_model
                # Copy processor config from base if it exists
                base_proc = Path(base_model) / "preprocessor_config.json"
                if base_proc.exists():
                    import shutil
                    dest = model_dir / "preprocessor_config.json"
                    shutil.copy2(base_proc, dest)
                    profile.has_processor_config = True
                    logger.info(
                        f"[ModelProfiler] Copied preprocessor_config from "
                        f"{base_model} to {model_path}"
                    )
            elif base_model:
                profile.base_model = base_model
                warnings.append(
                    f"Base model path '{base_model}' not found locally. "
                    f"Processor loading may fail."
                )

        # === Step 7: Determine supported domains ===
        all_domains = list(available_domains or ALL_DOMAINS)
        profile.supported_domains = [
            d for d in all_domains if profile.can_handle_domain(d)
        ]

        if not profile.supported_domains:
            errors.append("No compatible domains found for this model")

        # === Step 8: Check model weights exist ===
        has_weights = any(
            f.endswith((".safetensors", ".bin", ".pt"))
            for f in profile.model_files
        )
        if not has_weights:
            errors.append("No model weight files found (.safetensors, .bin, .pt)")

        # === Step 9: Final validation ===
        profile.validation_errors = errors
        profile.validation_warnings = warnings
        profile.is_valid = len(errors) == 0

        return profile

    @staticmethod
    def profile_and_repair(
        model_path: str,
        base_model_path: str = "",
        available_domains: list[str] | None = None,
    ) -> ModelProfile:
        """Profile a model and attempt to repair any issues.

        This is the more aggressive version that will:
        1. Copy missing processor configs from base model
        2. Fix tokenizer configs
        3. Validate loading actually works (dry run)

        Args:
            model_path: Path to model
            base_model_path: Path to base model (for VL merged checkpoints)
            available_domains: Available gym domains

        Returns:
            Repaired ModelProfile
        """
        profile = ModelProfiler.profile(
            model_path,
            available_domains=available_domains,
            base_model_hint=base_model_path,
        )

        if not profile.is_valid:
            return profile

        model_dir = Path(model_path)

        # Repair 1: Copy missing files from base model for VL models
        if profile.is_vl_model and base_model_path:
            base_dir = Path(base_model_path)
            files_to_copy = [
                "preprocessor_config.json",
                "chat_template.json",
            ]
            for fname in files_to_copy:
                src = base_dir / fname
                dst = model_dir / fname
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info(f"[ModelProfiler] Repaired: copied {fname} from base")
                    if fname == "preprocessor_config.json":
                        profile.has_processor_config = True

        # Repair 2: Ensure tokenizer has required files
        if profile.is_vl_model and base_model_path:
            base_dir = Path(base_model_path)
            # VL models often need special_tokens_map, added_tokens
            for fname in ["special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json"]:
                src = base_dir / fname
                dst = model_dir / fname
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info(f"[ModelProfiler] Repaired: copied {fname} from base")

        # Re-profile after repairs
        return ModelProfiler.profile(
            model_path,
            available_domains=available_domains,
            base_model_hint=base_model_path,
        )

    @staticmethod
    def validate_loading(profile: ModelProfile, timeout_s: int = 60) -> dict:
        """Validate that a model can actually be loaded (CPU dry run).

        Returns:
            dict with "success", "error", "load_time_s"
        """
        import time

        if not profile.is_valid:
            return {
                "success": False,
                "error": f"Profile invalid: {profile.validation_errors}",
                "load_time_s": 0,
            }

        t0 = time.time()
        try:
            # Only load config + tokenizer (not full model) to save time
            from transformers import AutoConfig, AutoTokenizer
            config = AutoConfig.from_pretrained(
                profile.model_path, trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(
                profile.model_path, trust_remote_code=True
            )

            # Verify the model class can be imported
            profile._resolve_class(profile.model_class, profile.model_import_path)

            # If VL, try loading processor
            if profile.requires_processor:
                proc = profile.load_processor()
                if proc is None:
                    return {
                        "success": False,
                        "error": "VL model but processor loading failed",
                        "load_time_s": time.time() - t0,
                    }

            return {
                "success": True,
                "error": None,
                "load_time_s": round(time.time() - t0, 2),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "load_time_s": round(time.time() - t0, 2),
            }


# ============================================================
# Gym Registration Helper
# ============================================================

class GymModelRegistry:
    """Registry of all models that have been profiled for the gym.

    Maintains profiles for all registered models and provides
    lookup by capability.
    """

    def __init__(self):
        self._profiles: dict[str, ModelProfile] = {}

    def register(
        self,
        model_path: str,
        base_model_path: str = "",
        available_domains: list[str] | None = None,
        repair: bool = True,
    ) -> ModelProfile:
        """Register a model with the gym.

        Auto-profiles the model, attempts repairs if needed,
        and stores the profile.

        Args:
            model_path: Path to model
            base_model_path: Base model path (for merged VL checkpoints)
            available_domains: Domains available in the gym
            repair: Attempt to repair missing files

        Returns:
            ModelProfile (check .is_valid before using)
        """
        if repair:
            profile = ModelProfiler.profile_and_repair(
                model_path,
                base_model_path=base_model_path,
                available_domains=available_domains,
            )
        else:
            profile = ModelProfiler.profile(
                model_path,
                available_domains=available_domains,
                base_model_hint=base_model_path,
            )

        self._profiles[model_path] = profile

        if profile.is_valid:
            logger.info(
                f"[GymModelRegistry] Registered: {profile.model_name} "
                f"({profile.architecture}, "
                f"modalities={profile.modalities}, "
                f"domains={len(profile.supported_domains)})"
            )
        else:
            logger.error(
                f"[GymModelRegistry] FAILED to register: {profile.model_name} "
                f"errors={profile.validation_errors}"
            )

        return profile

    def get_profile(self, model_path: str) -> ModelProfile | None:
        """Get the profile for a registered model."""
        return self._profiles.get(model_path)

    def get_all_profiles(self) -> list[ModelProfile]:
        """Get all registered profiles."""
        return list(self._profiles.values())

    def find_models_for_domain(self, domain: str) -> list[ModelProfile]:
        """Find all registered models that can handle a domain."""
        return [
            p for p in self._profiles.values()
            if p.is_valid and p.can_handle_domain(domain)
        ]

    def find_vl_models(self) -> list[ModelProfile]:
        """Find all VL-capable models."""
        return [
            p for p in self._profiles.values()
            if p.is_valid and p.is_vl_model
        ]

    def find_text_models(self) -> list[ModelProfile]:
        """Find all text-only models."""
        return [
            p for p in self._profiles.values()
            if p.is_valid and not p.is_vl_model
        ]

    def summary(self) -> str:
        """Print a summary of all registered models."""
        lines = ["=" * 70, "  GYM Model Registry", "=" * 70]
        for path, profile in self._profiles.items():
            status = "OK" if profile.is_valid else "INVALID"
            mods = ",".join(profile.modalities)
            doms = len(profile.supported_domains)
            mem = profile.estimated_memory_gb
            lines.append(
                f"  [{status:>7}] {profile.model_name:<30} "
                f"| {mods:<16} | {doms} domains | ~{mem:.0f}GB"
            )
        lines.append("=" * 70)
        return "\n".join(lines)
