"""Centralized model loading with automatic VL/CausalLM detection.

Prevents: BUG-006 (VL model loaded as CausalLM), BUG-007 (model_type variant mismatch)
See BUGLOG.md for details.

Usage:
    from bioagents.utils.model_loader import load_model_auto, is_vl_model

    model, is_vl = load_model_auto("/path/to/model")
    tokenizer = load_tokenizer("/path/to/model")
"""

from __future__ import annotations

from loguru import logger


def is_vl_model(model_path: str) -> bool:
    """Detect if a model is a Vision-Language model.

    Uses substring matching to handle variant model_type strings
    like 'qwen2_5_vl', 'qwen2_5_vl_text', 'qwen2_vl', etc.

    BUG-007 prevention: Never use exact string matching for model_type.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load config from {model_path}: {e}")
        return False

    model_type = getattr(config, "model_type", "").lower()
    architectures = [a.lower() for a in getattr(config, "architectures", []) or []]

    # Substring matching (BUG-007: model_type can be "qwen2_5_vl_text", etc.)
    if "qwen2" in model_type and "vl" in model_type:
        return True

    # Also check architectures list
    for arch in architectures:
        if "vl" in arch or "vision" in arch:
            return True

    return False


def load_model_auto(
    model_path: str,
    dtype=None,
    device_map: str | None = None,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = False,
):
    """Load a model with automatic VL/CausalLM detection.

    Returns:
        (model, is_vl): Tuple of loaded model and whether it's a VL model.

    BUG-006 prevention: VL models require Qwen2_5_VLForConditionalGeneration.
    BUG-007 prevention: Uses substring matching for model_type detection.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.bfloat16

    load_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device_map:
        load_kwargs["device_map"] = device_map
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    vl = is_vl_model(model_path)

    if vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        logger.info(f"Loading VL model from {model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs,
        )
    else:
        logger.info(f"Loading CausalLM from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **load_kwargs,
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled")

    return model, vl


def load_tokenizer(model_path: str):
    """Load tokenizer with proper pad_token setup."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# GYM Environment constants (BUG-009 prevention)
GYM_ENV_ID = "BioAgent-v0"
"""The registered Gymnasium environment ID. Always use this constant instead of string literals."""
