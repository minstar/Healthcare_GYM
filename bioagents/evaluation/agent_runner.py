"""LLM Agent Runner for BIOAgents.

Connects a language model to the BIOAgents environment for multi-turn
tool-use evaluation. Supports:
- vLLM-based fast inference
- HuggingFace transformers-based inference
- Multi-turn tool calling with automatic parsing
- Full trajectory logging

Usage:
    runner = AgentRunner(
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        backend="vllm",
    )
    results = runner.run_task(domain="clinical_diagnosis", task_id="dx_pneumonia_001")
"""

import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from loguru import logger

from bioagents.evaluation.rewards import compute_composite_reward


@dataclass
class RunConfig:
    """Configuration for an agent run."""
    model_name_or_path: str
    backend: Literal["vllm", "transformers"] = "transformers"
    domain: str = "clinical_diagnosis"
    task_ids: Optional[list[str]] = None       # None = run all tasks
    task_split: Optional[str] = None
    max_turns: int = 15
    temperature: float = 0.1
    top_p: float = 0.95
    max_new_tokens: int = 1024
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    log_dir: str = "logs/runs"
    seed: int = 42


@dataclass
class TurnRecord:
    """Record of a single turn in the agent-environment interaction."""
    turn_idx: int
    prompt: str = ""
    raw_output: str = ""
    parsed_tool_call: Optional[dict] = None
    tool_response: Optional[str] = None
    is_final_answer: bool = False
    latency_seconds: float = 0.0


@dataclass 
class TaskResult:
    """Result of running a single task."""
    task_id: str
    domain: str
    model_name: str
    turns: list[TurnRecord] = field(default_factory=list)
    total_turns: int = 0
    action_score: float = 0.0
    final_reward: float = 0.0
    completed: bool = False
    error: Optional[str] = None
    trajectory: dict = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    total_latency: float = 0.0


def _build_onboarding_guidance(domain: str) -> str:
    """Build compact agent onboarding guidance for the system prompt.

    This injects essential behavioral tips so that ANY model — even one
    that has never seen the full AGENT_GUIDELINE.md — can perform well
    in the GYM on its first attempt.
    """

    _DOMAIN_TIPS = {
        "clinical_diagnosis": (
            "1. Gather info first: patient summary → vitals → labs → history\n"
            "2. Use think() to build differential diagnosis\n"
            "3. Search guidelines for evidence-based management\n"
            "4. Record your diagnosis with ICD-10 code before submitting"
        ),
        "drug_interaction": (
            "1. Get patient medications first\n"
            "2. Check each interaction pair systematically\n"
            "3. Search for safer alternatives if severe interaction found\n"
            "4. Assess cumulative risk before recommending"
        ),
        "ehr_management": (
            "1. Start with patient summary (hadm_id)\n"
            "2. Check lab trends + vital alerts for dynamic patterns\n"
            "3. Calculate clinical scores (SOFA, NEWS) for severity\n"
            "4. Review procedures + discharge summary for full picture"
        ),
        "medical_qa": (
            "1. Analyze the question and identify key concepts\n"
            "2. Search PubMed or medical wiki for evidence\n"
            "3. Browse relevant articles for detailed information\n"
            "4. Use analyze_answer_options for MCQA before submitting"
        ),
        "triage_emergency": (
            "1. Get presentation → ABC assessment immediately\n"
            "2. Vitals + GCS for acuity level\n"
            "3. Calculate ESI level using the algorithm\n"
            "4. Order STAT labs/imaging per protocol\n"
            "5. Submit with ESI level + disposition + orders"
        ),
        "psychiatry": (
            "1. Get presentation → psychiatric history\n"
            "2. Perform mental status exam\n"
            "3. Use validated scales: PHQ-9, GAD-7, Columbia-SSRS\n"
            "4. Screen substance use (AUDIT/DAST)\n"
            "5. Submit: diagnosis + risk level + treatment plan + disposition"
        ),
        "obstetrics": (
            "1. Get patient demographics + obstetric history\n"
            "2. Assess fetal status (FHR, BPP)\n"
            "3. Evaluate labor progress + Bishop score if applicable\n"
            "4. Check medication safety (teratogenicity)\n"
            "5. Follow ACOG protocols for management"
        ),
        "visual_diagnosis": (
            "1. Analyze the medical image with focus areas\n"
            "2. Get patient context for clinical correlation\n"
            "3. Search similar cases for comparison\n"
            "4. Compare with prior studies if available\n"
            "5. Record diagnosis with confidence level"
        ),
        "radiology_report": (
            "1. Get study info + clinical history\n"
            "2. Analyze findings systematically\n"
            "3. Get prior reports for comparison\n"
            "4. Use reporting checklist for completeness\n"
            "5. Submit structured report: indication → technique → findings → impression"
        ),
        "cross_domain": (
            "1. Identify the primary clinical concern\n"
            "2. Use domain-specific tools as needed across specialties\n"
            "3. Ensure continuity of care between phases\n"
            "4. Provide a comprehensive multi-specialty assessment"
        ),
    }

    tips = _DOMAIN_TIPS.get(domain, "")
    tip_block = f"\n### Domain-Specific Workflow\n{tips}" if tips else ""

    return f"""## Agent Behavior Guide
**You are scored on 5 dimensions**: Accuracy (30%), Process (25%), Safety (20%), Format (15%), Coherence (10%).

### Critical Rules
- **Use tools before answering.** Gathering evidence is mandatory. Never answer from memory alone.
- **Use think() liberally.** Show your clinical reasoning. It improves Process and Coherence scores.
- **Use 3-8 turns.** 1-turn answers get premature_stop penalty. 12+ turns get over_investigation penalty.
- **Always end with submit_answer.** Your response is only recorded when you submit.
- **One tool call per turn.** Respond with ONLY the JSON object — no extra text.
- **Check safety.** Drug interactions, contraindications, critical values — flag them explicitly.
{tip_block}"""


def build_system_prompt(
    policy: str,
    tools: list[dict],
    domain: str = "clinical_diagnosis",
    task: Optional[dict] = None,
    agent_profile: Optional[dict] = None,
    reward_strategy: str = "grpo",
) -> str:
    """Build the system prompt with policy, tool definitions, and adaptive guidance.

    Args:
        policy: Environment policy text
        tools: Tool definitions (OpenAI format)
        domain: Task domain name
        task: Optional task dict for adaptive guidance
        agent_profile: Optional agent reflection/profile for weakness-aware guidance
        reward_strategy: Current reward strategy (grpo/mrpo/sarl/adaptive)

    Returns:
        Complete system prompt with adaptive tool usage guidance
    """
    tool_section = json.dumps(tools, indent=2, ensure_ascii=False)
    
    # Domain-specific system prompts for optimal agent performance
    _DOMAIN_PROMPTS = {
        "medical_qa": {
            "role": "You are a medical AI assistant that answers medical questions using evidence-based reasoning. Search for evidence, analyze options, and submit your answer with clear clinical reasoning.",
            "final": "When you are ready, use the submit_answer tool to submit your final answer.",
        },
        "clinical_diagnosis": {
            "role": "You are a clinical diagnostician AI. Review patient history, vital signs, lab results, and imaging to formulate differential diagnoses. Follow clinical guidelines and order appropriate workup.",
            "final": "When you have gathered enough information, provide your clinical assessment including: primary diagnosis, differential diagnoses, recommended tests, and management plan.",
        },
        "drug_interaction": {
            "role": "You are a clinical pharmacology AI specializing in drug-drug interactions. Review medication profiles, check for interactions, assess severity, and provide evidence-based management recommendations.",
            "final": "When done, use submit_answer to provide your interaction assessment and management recommendation.",
        },
        "visual_diagnosis": {
            "role": "You are a medical imaging AI assistant. Analyze medical images, interpret findings, compare with prior studies, and provide structured diagnostic assessments.",
            "final": "When you have completed your analysis, provide your diagnostic impression and recommendations.",
        },
        "ehr_management": {
            "role": "You are an EHR analysis AI. Navigate electronic health records, identify trends in lab values and vitals, reconcile medications, calculate clinical scores, and support discharge planning.",
            "final": "When done, use submit_answer to provide your clinical assessment based on the EHR data.",
        },
        "triage_emergency": {
            "role": "You are an emergency triage AI. Rapidly assess patient presentations, determine ESI (Emergency Severity Index) levels, identify life threats, and activate appropriate emergency protocols. Time is critical.",
            "final": "When done, use submit_answer to provide the ESI level and recommended actions.",
        },
        "radiology_report": {
            "role": "You are a radiology AI assistant. Generate structured radiology reports following ACR standards. Describe findings systematically, compare with priors, apply classification systems (BI-RADS, TI-RADS, LI-RADS, Fleischner), and provide clear impressions.",
            "final": "When done, use submit_report or submit_answer to provide your structured radiology report.",
        },
        "psychiatry": {
            "role": "You are a psychiatry AI assistant. Conduct mental status examinations, assess suicide risk using validated scales (PHQ-9, GAD-7, Columbia), evaluate for psychosis and substance use, and develop treatment plans following APA guidelines.",
            "final": "When done, use submit_answer to provide your psychiatric assessment and treatment plan.",
        },
        "obstetrics": {
            "role": "You are an obstetrics AI assistant. Assess maternal and fetal status, interpret fetal heart tracings, manage labor and delivery complications, and follow ACOG guidelines. Patient safety for both mother and fetus is paramount.",
            "final": "When done, use submit_answer to provide your obstetric assessment and management plan.",
        },
        "cross_domain": {
            "role": "You are a multi-specialty clinical AI managing complex patient pathways that span multiple departments. Coordinate across specialties, ensure continuity of care, and follow evidence-based clinical pathways.",
            "final": "When you have completed this phase of the clinical pathway, provide your assessment and plan for the next phase.",
        },
    }

    domain_info = _DOMAIN_PROMPTS.get(domain, {})
    if domain_info:
        role = domain_info["role"]
        final_instruction = domain_info["final"]
    else:
        role = "You are a medical AI assistant operating in a clinical environment. Follow the policy below and use the available tools to help with patient care."
        final_instruction = "When you have gathered enough information and want to give your final assessment, respond with your clinical analysis as plain text (no JSON)."
    
    # ── Build agent onboarding guidance ──
    onboarding = _build_onboarding_guidance(domain)

    base_prompt = f"""{role}

## Policy
{policy}

{onboarding}

## Available Tools
You have access to the following tools. To use a tool, respond with ONLY a JSON object in this exact format:
```json
{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
```

Do NOT include any other text when making a tool call. One tool call per response.

{final_instruction}

## Tool Definitions
{tool_section}"""

    # Inject adaptive tool usage guidance if task info is available
    if task is not None:
        try:
            from bioagents.gym.tool_guidance import GuidanceInjector
            injector = GuidanceInjector(
                agent_profile=agent_profile,
                reward_strategy=reward_strategy,
            )
            base_prompt = injector.inject(
                system_prompt=base_prompt,
                domain=domain,
                task=task,
                tools=tools,
            )
        except Exception:
            pass  # Graceful fallback: no guidance

    return base_prompt


def _normalize_tool_call(parsed: dict) -> Optional[dict]:
    """Normalize various tool-call dict shapes to {name, arguments}."""
    # Standard: {"name": "...", "arguments": {...}}
    if "name" in parsed and isinstance(parsed.get("arguments"), dict):
        return {"name": parsed["name"], "arguments": parsed["arguments"]}
    if "name" in parsed:
        args = parsed.get("arguments") or parsed.get("parameters") or parsed.get("params") or {}
        return {"name": parsed["name"], "arguments": args if isinstance(args, dict) else {}}
    # Alt key: {"function": "...", "arguments": {...}}
    if "function" in parsed:
        args = parsed.get("arguments") or parsed.get("parameters") or {}
        return {"name": parsed["function"], "arguments": args if isinstance(args, dict) else {}}
    # Alt key: {"tool": "...", "args": {...}} (common in some frameworks)
    if "tool" in parsed:
        args = parsed.get("args") or parsed.get("arguments") or parsed.get("input") or {}
        return {"name": parsed["tool"], "arguments": args if isinstance(args, dict) else {}}
    # Alt key: {"action": "...", "action_input": {...}} (ReAct / LangChain style)
    if "action" in parsed and parsed["action"] not in ("Final Answer",):
        args = parsed.get("action_input") or parsed.get("arguments") or {}
        return {"name": parsed["action"], "arguments": args if isinstance(args, dict) else {}}
    return None


def parse_tool_call(text: str) -> Optional[dict]:
    """Parse a tool call from model output.
    
    Supports formats:
    1. Pure JSON: {"name": "...", "arguments": {...}}
    2. JSON in code block: ```json\\n{...}\\n```
    3. JSON embedded in text with markers
    4. XML-style: <tool_call>...</tool_call>, <|tool_call|>...<|/tool_call|>
    5. ReAct: Action: tool_name\\nAction Input: {...}
    6. Alt keys: function/tool/action instead of name
    """
    text = text.strip()
    
    # ── Pre-check: skip obviously non-tool-call text ──
    # If the text is very short and has no JSON / XML indicators, skip
    has_json_hint = "{" in text
    has_xml_hint = "<tool_call" in text.lower() or "<|tool_call" in text
    has_react_hint = "Action:" in text or "action:" in text
    
    # ── Try 0: XML-style tool call tags ──
    # Qwen-style: <|tool_call|>{...}<|/tool_call|>
    xml_patterns = [
        r'<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>',
        r'<tool_call>\s*(.*?)\s*</tool_call>',
        r'<function_call>\s*(.*?)\s*</function_call>',
        r'<\|plugin\|>\s*(.*?)\s*<\|/plugin\|>',
    ]
    for pat in xml_patterns:
        xml_match = re.search(pat, text, re.DOTALL)
        if xml_match:
            inner = xml_match.group(1).strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    norm = _normalize_tool_call(parsed)
                    if norm:
                        return norm
            except json.JSONDecodeError:
                pass
    
    # ── Try 1: Extract JSON from code blocks ──
    code_block_match = re.search(r'```(?:json|tool_call)?\s*\n?({.*?})\s*\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                norm = _normalize_tool_call(parsed)
                if norm:
                    return norm
        except json.JSONDecodeError:
            pass
    
    # ── Try 2: Direct JSON parse ──
    if has_json_hint:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                norm = _normalize_tool_call(parsed)
                if norm:
                    return norm
        except json.JSONDecodeError:
            pass
    
    # ── Try 3: ReAct format ──
    # "Action: search_pubmed\nAction Input: {"query": "..."}"
    if has_react_hint:
        react_match = re.search(
            r'[Aa]ction\s*:\s*([^\n{]+?)(?:\n|\s+)[Aa]ction\s*[Ii]nput\s*:\s*(.+)',
            text, re.DOTALL
        )
        if react_match:
            tool_name = react_match.group(1).strip().strip('"').strip("'")
            args_str = react_match.group(2).strip()
            if tool_name:
                try:
                    args = json.loads(args_str)
                    if isinstance(args, dict):
                        return {"name": tool_name, "arguments": args}
                except json.JSONDecodeError:
                    # args might be plain text
                    return {"name": tool_name, "arguments": {"input": args_str}}
    
    if not has_json_hint:
        return None

    # ── Try 4: Find JSON-like pattern in text ──
    json_match = re.search(
        r'(\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\})',
        text, re.DOTALL,
    )
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            norm = _normalize_tool_call(parsed)
            if norm:
                return norm
        except json.JSONDecodeError:
            pass
    
    # ── Try 5: More lenient nested JSON search ──
    matches = list(re.finditer(r'\{', text))
    for m in matches[:10]:  # limit iterations for large outputs
        start = m.start()
        depth = 0
        for i in range(start, min(start + 2000, len(text))):  # cap scan length
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            norm = _normalize_tool_call(parsed)
                            if norm:
                                return norm
                    except json.JSONDecodeError:
                        pass
                    break
    
    return None


def repair_model_config(model_path: str) -> bool:
    """Repair model config.json for cross-version transformers compatibility.

    When a model is saved with transformers >=5.x (nested text_config +
    rope_parameters) but loaded with transformers 4.x (flat + rope_scaling),
    the attention layer receives rope_scaling=None, causing TypeError in
    Qwen2.5-VL attention forward.

    This function converts the config in-place to the flat format expected by
    transformers 4.x while remaining readable by 5.x.

    Returns True if config was repaired, False if no repair was needed.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = json.load(f)

    # Only repair if nested text_config with rope_parameters exists
    # AND top-level rope_scaling is missing
    text_cfg = config.get("text_config", {})
    if not (
        text_cfg
        and "rope_parameters" in text_cfg
        and "rope_scaling" not in config
    ):
        return False

    logger.info(f"Repairing config format: {config_path}")

    # Reference base config for Qwen2.5-VL-7B fields
    rope_params = text_cfg["rope_parameters"]

    new_config = {}
    new_config["architectures"] = config.get("architectures", ["Qwen2_5_VLForConditionalGeneration"])

    # Promote text_config fields to top level
    text_fields = [
        "attention_dropout", "bos_token_id", "eos_token_id", "hidden_act",
        "hidden_size", "initializer_range", "intermediate_size",
        "max_position_embeddings", "max_window_layers",
        "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
        "rms_norm_eps", "sliding_window", "use_cache", "use_sliding_window",
        "vocab_size", "pad_token_id",
    ]
    for field in text_fields:
        if field in text_cfg:
            new_config[field] = text_cfg[field]

    # model_type at top level
    new_config["model_type"] = config.get("model_type", text_cfg.get("model_type", "qwen2_5_vl"))
    if new_config["model_type"] == "qwen2_5_vl_text":
        new_config["model_type"] = "qwen2_5_vl"

    # Convert rope_parameters → rope_scaling
    new_config["rope_scaling"] = {
        "mrope_section": rope_params.get("mrope_section", [16, 24, 24]),
        "rope_type": rope_params.get("rope_type", "default"),
        "type": rope_params.get("type", "default"),
    }
    new_config["rope_theta"] = rope_params.get("rope_theta", 1000000.0)

    # Token IDs
    for tid in ["image_token_id", "video_token_id", "vision_end_token_id",
                "vision_start_token_id", "vision_token_id"]:
        if tid in config:
            new_config[tid] = config[tid]

    new_config["tie_word_embeddings"] = config.get("tie_word_embeddings", False)
    new_config["torch_dtype"] = config.get("dtype", config.get("torch_dtype", "bfloat16"))
    new_config["transformers_version"] = "4.57.3"

    # Sliding window default
    if new_config.get("sliding_window") is None:
        new_config["sliding_window"] = 32768

    # Vision config
    vision_cfg = config.get("vision_config", {})
    clean_vision = {k: v for k, v in vision_cfg.items() if k != "dtype"}
    new_config["vision_config"] = clean_vision

    # Backup + write
    backup_path = str(config_path) + ".bak_autorepair"
    if not Path(backup_path).exists():
        import shutil
        shutil.copy2(config_path, backup_path)

    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2)

    logger.info(f"Config repaired: {config_path}")
    return True


class AgentRunner:
    """Runs LLM agents in BIOAgents environments."""
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = Path(self.config.model_name_or_path).name
        self.run_id = f"{model_short}_{self.config.domain}_{timestamp}"
        self.log_path = Path(self.config.log_dir) / self.run_id
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
        }
        with open(self.log_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def load_model(self):
        """Load the language model."""
        # Auto-repair config for cross-version transformers compatibility
        model_path = self.config.model_name_or_path
        if Path(model_path).is_dir():
            repair_model_config(model_path)

        if self.config.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()
    
    def _load_vllm(self):
        """Load model with vLLM."""
        from vllm import LLM, SamplingParams
        
        logger.info(f"Loading {self.config.model_name_or_path} with vLLM (tp={self.config.tensor_parallel_size})")
        self.model = LLM(
            model=self.config.model_name_or_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=8192,
            seed=self.config.seed,
        )
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            stop=["```\n\n", "\n\nUser:", "\n\nHuman:"],
        )
        logger.info("vLLM model loaded successfully")
    
    def _load_transformers(self):
        """Load model with HuggingFace transformers using ModelProfile.

        The ModelProfile system auto-detects:
        - Correct model class (AutoModelForCausalLM vs Qwen2_5_VLForConditionalGeneration etc.)
        - Whether a processor is needed (VL models)
        - Supported modalities and domains
        - Loading kwargs (use_cache, etc.)

        This eliminates manual model-type branching and makes adding
        new model architectures a one-line registry entry.
        """
        import torch
        from transformers import AutoTokenizer

        model_path = self.config.model_name_or_path
        logger.info(f"Loading {model_path} with transformers (via ModelProfile)")

        # Use ModelProfile for auto-detection
        from bioagents.gym.model_profile import ModelProfiler
        profile = ModelProfiler.profile(model_path)

        if not profile.is_valid:
            logger.error(
                f"Model profile invalid: {profile.validation_errors}. "
                f"Falling back to legacy loading."
            )
            self._load_transformers_legacy()
            return

        logger.info(
            f"Model profiled: {profile.model_name} "
            f"(type={profile.model_type}, arch={profile.architecture}, "
            f"VL={profile.is_vl_model}, class={profile.model_class})"
        )

        self._is_vl_model = profile.is_vl_model
        self._model_profile = profile

        # Load tokenizer / processor using profile instructions
        if profile.requires_processor:
            self.processor = profile.load_processor()
            if self.processor is not None:
                self.tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
            else:
                logger.warning(
                    "Processor loading failed, falling back to tokenizer only"
                )
                self.tokenizer = profile.load_tokenizer()
        else:
            self.tokenizer = profile.load_tokenizer()
            self.processor = None

        # Load model using profile's model class
        self.model = profile.load_model(device_map="auto")
        logger.info(
            f"Model loaded via ModelProfile "
            f"(VL={profile.is_vl_model}, class={profile.model_class})"
        )

    def _load_transformers_legacy(self):
        """Legacy model loading (fallback when ModelProfile fails)."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoProcessor,
            AutoConfig,
        )

        model_path = self.config.model_name_or_path
        logger.info(f"Loading {model_path} with legacy loader")

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        architectures = getattr(config, "architectures", [])

        self._is_vl_model = any(
            "vl" in a.lower() or "vision" in a.lower()
            for a in (architectures or [])
        ) or "vl" in model_type.lower()

        # Load tokenizer / processor
        if self._is_vl_model:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
                self.tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                self.processor = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.processor = None

        # Load model
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

        try:
            if is_qwen_vl:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, **load_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, **load_kwargs
                )
        except Exception as e:
            logger.warning(f"First load attempt failed: {e}, retrying without sdpa")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs
            )

        self.model.eval()
        logger.info(f"Model loaded via legacy loader (VL={self._is_vl_model})")
    
    def generate(self, messages: list[dict]) -> str:
        """Generate a response from the model."""
        if self.config.backend == "vllm":
            return self._generate_vllm(messages)
        else:
            return self._generate_transformers(messages)
    
    def _generate_vllm(self, messages: list[dict]) -> str:
        """Generate with vLLM using chat template."""
        from vllm import SamplingParams
        
        # Use the tokenizer from vLLM engine
        tokenizer = self.model.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        outputs = self.model.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def _generate_transformers(self, messages: list[dict]) -> str:
        """Generate with HuggingFace transformers."""
        import torch
        
        # For VL models, use processor if available
        if self._is_vl_model and self.processor is not None:
            # Text-only input for VL model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], padding=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            # Apply chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>\n{content}\n"
                text += "<|assistant|>\n"
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                top_p=self.config.top_p if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    def run_task(self, task: dict, env) -> TaskResult:
        """Run a single task in the environment.
        
        Args:
            task: Task dictionary from tasks.json
            env: BioAgentGymEnv instance
            
        Returns:
            TaskResult with full trajectory and scores
        """
        task_id = task["id"]
        result = TaskResult(
            task_id=task_id,
            domain=self.config.domain,
            model_name=Path(self.config.model_name_or_path).name,
            start_time=datetime.now().isoformat(),
        )
        
        logger.info(f"Starting task: {task_id}")
        
        # Reset environment
        obs, info = env.reset(options={"task_id": task_id})
        
        # Build conversation with adaptive tool guidance
        system_prompt = build_system_prompt(
            info["policy"],
            info["tools"],
            domain=self.config.domain,
            task=task,
            agent_profile=getattr(self, "_agent_profile", None),
            reward_strategy=getattr(self, "_reward_strategy", "grpo"),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs},
        ]
        
        try:
            for turn_idx in range(self.config.max_turns):
                turn = TurnRecord(turn_idx=turn_idx)
                
                # Generate
                t0 = time.time()
                raw_output = self.generate(messages)
                turn.latency_seconds = time.time() - t0
                turn.raw_output = raw_output
                
                logger.debug(f"Turn {turn_idx}: {raw_output[:200]}...")
                
                # Parse tool call
                tool_call = parse_tool_call(raw_output)
                
                if tool_call is not None:
                    turn.parsed_tool_call = tool_call
                    tool_name = tool_call.get("name", "")
                    
                    # Check if this is a terminating tool (submit_answer)
                    is_submit = tool_name == "submit_answer"
                    
                    # Detect repeated tool call (stuck model) — skip for submit_answer
                    if not is_submit:
                        recent_tool_calls = [
                            t.parsed_tool_call.get("name", "") if t.parsed_tool_call else ""
                            for t in result.turns[-3:]
                        ]
                        if recent_tool_calls.count(tool_name) >= 2:
                            logger.warning(
                                f"Tool '{tool_name}' called 3+ times in a row. "
                                "Injecting nudge to move forward."
                            )
                            messages.append({"role": "assistant", "content": json.dumps(tool_call)})
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"You have already called '{tool_name}' multiple times with similar arguments. "
                                    "Please proceed with your analysis using the information gathered so far. "
                                    "Use a DIFFERENT tool or provide your final answer."
                                ),
                            })
                            result.turns.append(turn)
                            continue
                    
                    # Execute tool via environment
                    action = json.dumps(tool_call)
                    observation, reward, terminated, truncated, step_info = env.step(action)
                    
                    # Normalize observation to string
                    if isinstance(observation, dict):
                        observation_str = json.dumps(observation, indent=2, ensure_ascii=False)
                    elif isinstance(observation, (list, tuple)):
                        observation_str = json.dumps(observation, indent=2, ensure_ascii=False)
                    else:
                        observation_str = str(observation) if observation is not None else ""

                    turn.tool_response = observation_str

                    # Add to messages
                    messages.append({"role": "assistant", "content": json.dumps(tool_call)})
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tool_name}:\n{observation_str}"
                    })
                    
                    result.turns.append(turn)
                    
                    # Force termination on submit_answer
                    if is_submit:
                        logger.info(f"submit_answer called with: {tool_call.get('arguments', {})}")
                        break
                    
                    if terminated or truncated:
                        break
                else:
                    # Final answer (no tool call)
                    turn.is_final_answer = True
                    messages.append({"role": "assistant", "content": raw_output})
                    result.turns.append(turn)
                    
                    # Check for repetition (model stuck in a loop)
                    if len(result.turns) >= 3:
                        recent = [t.raw_output[:100] for t in result.turns[-3:]]
                        if len(set(recent)) == 1:
                            logger.warning(f"Repetition detected at turn {turn_idx}, stopping.")
                            break
                    
                    # Agent gave final answer - break the loop
                    break
            
            # Get final trajectory and reward
            try:
                trajectory = env.get_trajectory()
                result.trajectory = trajectory
                result.final_reward = trajectory["final_reward"]
            except Exception:
                result.trajectory = {}
                result.final_reward = 0.0
            
            result.total_turns = len(result.turns)
            result.action_score = self._compute_action_score(task, env._tool_call_log)
            # For QA tasks, also compute accuracy
            if "answer" in task or "correct_answer" in task:
                qa_acc = self._compute_qa_accuracy(task, env._tool_call_log)
                result.trajectory["qa_accuracy"] = qa_acc
            
            # Compute composite reward using the new reward module
            final_answer_text = ""
            for t in reversed(result.turns):
                if t.is_final_answer and t.raw_output:
                    final_answer_text = t.raw_output
                    break
                if t.parsed_tool_call and t.parsed_tool_call.get("name") == "submit_answer":
                    final_answer_text = t.parsed_tool_call.get("arguments", {}).get("answer", "")
                    break
            
            correct_answer = task.get("answer", task.get("correct_answer", ""))
            expected_actions = task.get("evaluation_criteria", {}).get("actions", [])
            
            reward_details = compute_composite_reward(
                response=final_answer_text,
                correct_answer=correct_answer,
                tool_call_log=env._tool_call_log,
                expected_actions=expected_actions,
                is_final=True,
            )
            result.trajectory["reward_details"] = reward_details
            result.final_reward = reward_details["total"]
            
            result.completed = True
            
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}")
            result.error = str(e)
            import traceback
            result.error += "\n" + traceback.format_exc()
        
        result.end_time = datetime.now().isoformat()
        result.total_latency = sum(t.latency_seconds for t in result.turns)
        
        logger.info(
            f"Task {task_id}: turns={result.total_turns}, "
            f"action_score={result.action_score:.3f}, "
            f"reward={result.final_reward:.3f}, "
            f"latency={result.total_latency:.1f}s"
        )
        
        return result
    
    def _compute_qa_accuracy(self, task: dict, tool_call_log: list) -> float:
        """Check if the agent submitted the correct answer (for QA domains)."""
        correct_answer = task.get("answer", task.get("correct_answer", ""))
        if not correct_answer:
            return 0.0

        # Find the submit_answer tool call
        for tc in reversed(tool_call_log):
            if tc["tool_name"] == "submit_answer":
                submitted = tc["arguments"].get("answer", "").strip()
                # For multiple-choice, compare first letter
                if len(correct_answer.strip()) <= 2:
                    if submitted.upper() == correct_answer.strip().upper():
                        return 1.0
                    # Also check if they submitted the full option text
                    options = task.get("options", {})
                    correct_text = options.get(correct_answer.strip(), "").lower()
                    if correct_text and submitted.lower() == correct_text:
                        return 1.0
                    return 0.0
                else:
                    # Free text comparison
                    if submitted.lower() == correct_answer.strip().lower():
                        return 1.0
                    if correct_answer.strip().lower() in submitted.lower():
                        return 0.5
                    return 0.0

        # No answer submitted
        return 0.0

    def _compute_action_score(self, task: dict, tool_call_log: list) -> float:
        """Compute action score based on expected vs actual tool calls."""
        eval_criteria = task.get("evaluation_criteria", {})
        expected_actions = eval_criteria.get("actions", [])
        
        if not expected_actions:
            return 1.0
        
        matched = 0
        for exp in expected_actions:
            exp_name = exp.get("name", "")
            compare_args = exp.get("compare_args", [])
            exp_args = exp.get("arguments", {})
            
            for tc in tool_call_log:
                if tc["tool_name"] == exp_name:
                    if compare_args:
                        all_match = all(
                            str(tc["arguments"].get(k, "")).lower() == str(exp_args.get(k, "")).lower()
                            for k in compare_args
                            if k in exp_args
                        )
                        if all_match:
                            matched += 1
                            break
                    else:
                        matched += 1
                        break
        
        return matched / len(expected_actions)
    
    def run_all_tasks(self) -> list[TaskResult]:
        """Run all tasks (or specified task_ids) and return results."""
        from bioagents.gym.agent_env import BioAgentGymEnv
        
        # Create environment
        gym_env = BioAgentGymEnv(
            domain=self.config.domain,
            task_split=self.config.task_split,
            max_turns=self.config.max_turns,
        )
        
        # Get tasks
        if self.config.task_ids:
            task_ids = self.config.task_ids
        else:
            task_ids = [t["id"] for t in gym_env._tasks]
        
        logger.info(f"Running {len(task_ids)} tasks with {Path(self.config.model_name_or_path).name}")
        
        results = []
        for task_id in task_ids:
            task = gym_env._task_map[task_id]
            result = self.run_task(task, gym_env)
            results.append(result)
            self._save_task_result(result)
        
        # Save summary
        self._save_summary(results)
        
        return results
    
    def _save_task_result(self, result: TaskResult):
        """Save individual task result."""
        path = self.log_path / f"task_{result.task_id}.json"
        data = {
            "task_id": result.task_id,
            "domain": result.domain,
            "model_name": result.model_name,
            "total_turns": result.total_turns,
            "action_score": result.action_score,
            "final_reward": result.final_reward,
            "completed": result.completed,
            "error": result.error,
            "total_latency": result.total_latency,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "turns": [
                {
                    "turn_idx": t.turn_idx,
                    "raw_output": t.raw_output,
                    "parsed_tool_call": t.parsed_tool_call,
                    "tool_response": t.tool_response[:500] if t.tool_response else None,
                    "is_final_answer": t.is_final_answer,
                    "latency_seconds": t.latency_seconds,
                }
                for t in result.turns
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_summary(self, results: list[TaskResult]):
        """Save a summary of all task results."""
        summary = {
            "run_id": self.run_id,
            "model": Path(self.config.model_name_or_path).name,
            "domain": self.config.domain,
            "backend": self.config.backend,
            "num_tasks": len(results),
            "num_completed": sum(1 for r in results if r.completed),
            "num_errors": sum(1 for r in results if r.error),
            "avg_action_score": sum(r.action_score for r in results) / max(len(results), 1),
            "avg_reward": sum(r.final_reward for r in results) / max(len(results), 1),
            "avg_turns": sum(r.total_turns for r in results) / max(len(results), 1),
            "total_latency": sum(r.total_latency for r in results),
            "per_task": [
                {
                    "task_id": r.task_id,
                    "action_score": r.action_score,
                    "final_reward": r.final_reward,
                    "turns": r.total_turns,
                    "latency": r.total_latency,
                    "completed": r.completed,
                    "error": r.error is not None,
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat(),
        }
        
        path = self.log_path / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary table
        print("\n" + "=" * 80)
        print(f"  RUN SUMMARY: {self.run_id}")
        print("=" * 80)
        print(f"  Model: {summary['model']}")
        print(f"  Domain: {summary['domain']}")
        print(f"  Backend: {summary['backend']}")
        print(f"  Tasks: {summary['num_completed']}/{summary['num_tasks']} completed")
        print(f"  Avg Action Score: {summary['avg_action_score']:.3f}")
        print(f"  Avg Reward: {summary['avg_reward']:.3f}")
        print(f"  Avg Turns: {summary['avg_turns']:.1f}")
        print(f"  Total Latency: {summary['total_latency']:.1f}s")
        print("-" * 80)
        print(f"  {'Task ID':<30} {'Score':>8} {'Reward':>8} {'Turns':>6} {'Time':>8}")
        print("-" * 80)
        for t in summary["per_task"]:
            status = "✓" if t["completed"] else "✗"
            print(f"  {status} {t['task_id']:<28} {t['action_score']:>8.3f} {t['final_reward']:>8.3f} {t['turns']:>6} {t['latency']:>7.1f}s")
        print("=" * 80)
        
        logger.info(f"Results saved to {self.log_path}")
