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


# --------------------------------------------------------------------------- #
#  Multimodal tool forwarding — the one runtime switch on the image path
# --------------------------------------------------------------------------- #
#
# Until 2026-07-29 ``_generate_sglang_multimodal`` accepted ``tools`` and never
# forwarded it to the server. That call is the ONLY way the tool catalog reaches
# the model on an image input under prompt_mode "default" or "strong_tool"
# (react renders its own catalog into the system prompt, so react never depended
# on it). Every VQA row produced by those two modes therefore measured a
# TOOLLESS agent under a heading that says tool-using — which is precisely the
# thing the Base+AR condition exists to demonstrate.
#
# The fix is one line. Keeping the broken behaviour REACHABLE is what this
# switch is for: the rebuttal has to report both arms, and the only honest way
# to do that is one binary with one flag flipped, not a diff of two trees that
# also differ elsewhere.
#
#   HCGYM_MULTIMODAL_TOOLS  unset | 1 | true | yes | on | forward
#       Forward the catalog. The CORRECT behaviour, and the DEFAULT, so that
#       nobody reproduces the bug by forgetting a flag. The toolless arm is the
#       one that needs an explicit opt-in.
#   HCGYM_MULTIMODAL_TOOLS  0 | false | no | off | withhold
#       Withhold it, reproducing the pre-fix measurement on purpose.
#
# Any other value raises. A typo must never silently select an arm — least of
# all the buggy one.
#
# Scope, deliberately narrow. This gates the ``tools=`` key of the multimodal
# chat-completions request and nothing else. It does not touch the system
# prompt, the decoding parameters, the turn budget, the task order, the parser
# or the scorer. It is not consulted on the text-only path (which never calls
# the multimodal function at all) and it cannot move prompt_mode="react", which
# arrives here with tools=None either way via ``native_tools_for_prompt_mode``.
MULTIMODAL_TOOLS_ENV = "HCGYM_MULTIMODAL_TOOLS"

_MM_FORWARD_VALUES = frozenset({"1", "true", "yes", "on", "forward"})
_MM_WITHHOLD_VALUES = frozenset({"0", "false", "no", "off", "withhold"})

# Observability only — incremented by the multimodal request builder, read by
# ``multimodal_tool_forwarding()``. Never consulted by any decision, so it
# cannot change what the model sees. Its point is that a text-only benchmark
# lands ``multimodal_requests: 0`` in its own results file, which turns "this
# switch cannot affect medqa" from an assertion into a recorded number.
_MM_REQUEST_STATS = {"multimodal_requests": 0, "multimodal_requests_with_tools": 0}


def multimodal_tools_enabled() -> bool:
    """Whether to attach the tool catalog to a multimodal request.

    Read from the environment on every call, so a single process could in
    principle run both arms; in practice one job runs one arm. Default True.
    """
    raw = os.environ.get(MULTIMODAL_TOOLS_ENV)
    if raw is None or raw.strip() == "":
        return True
    value = raw.strip().lower()
    if value in _MM_FORWARD_VALUES:
        return True
    if value in _MM_WITHHOLD_VALUES:
        return False
    raise ValueError(
        f"{MULTIMODAL_TOOLS_ENV}={raw!r} is not a recognised value. Use one of "
        f"{sorted(_MM_FORWARD_VALUES)} to forward the tool catalog on image "
        f"inputs (the default and the correct behaviour), or one of "
        f"{sorted(_MM_WITHHOLD_VALUES)} to deliberately reproduce the pre-fix "
        f"toolless measurement. Refusing to guess: a typo here silently "
        f"changes which condition a VQA number belongs to."
    )


def multimodal_tools_arm() -> str:
    """Stable label for the arm in effect: "forward" or "withheld"."""
    return "forward" if multimodal_tools_enabled() else "withheld"


def reset_multimodal_request_stats() -> None:
    """Zero the multimodal request counters (call once per benchmark)."""
    for key in _MM_REQUEST_STATS:
        _MM_REQUEST_STATS[key] = 0


def multimodal_tool_forwarding() -> dict:
    """Provenance block for the results artifact.

    Written into every results file so a VQA number can never be read without
    knowing which arm produced it.
    """
    raw = os.environ.get(MULTIMODAL_TOOLS_ENV)
    enabled = multimodal_tools_enabled()
    return {
        "arm": "forward" if enabled else "withheld",
        "enabled": enabled,
        "env_var": MULTIMODAL_TOOLS_ENV,
        "env_value": raw,  # None when unset, i.e. running on the default
        "default_when_unset": True,
        "scope": (
            "tools= on the sglang multimodal (image) chat-completions request "
            "only; prompt_mode=react and every text-only benchmark are "
            "unaffected in both settings"
        ),
        **_MM_REQUEST_STATS,
    }


@dataclass
class RunConfig:
    """Configuration for an agent run."""
    model_name_or_path: str
    backend: Literal["vllm", "transformers", "sglang"] = "transformers"
    server_url: Optional[str] = None  # SGLang server URL (e.g., http://localhost:30000)
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
    # Ablation: disable tools to measure "w/ tools" vs "w/o tools" performance
    # (SciAgentGYM-style). When True, agent gets only the task ticket and must
    # answer without any tool calls. max_turns is set to 1 (single-shot).
    no_tools: bool = False
    # Ablation: disable think() tool to measure reasoning chain impact.
    # When True, think() is removed from tool definitions, forcing the
    # agent to reason within its response text only (no explicit CoT).
    # Used for "Short reasoning vs Long reasoning" ablation study.
    no_think: bool = False
    # Prompting baseline (eval-only, zero training). Selects how the
    # tool-use contract is phrased on top of the SAME domain role, policy,
    # scoring rubric, domain workflow tips and tool set:
    #   "default"     — the Base+AR condition. Byte-for-byte unchanged.
    #   "strong_tool" — Base+AR with a materially stronger tool-use contract
    #                   (mandatory first call, search budget + stopping rule,
    #                   explicit submit contract, anti-hallucination rules).
    #   "react"       — explicit ReAct scaffolding (Thought / Action /
    #                   Action Input / Observation), same tools. ReAct is the
    #                   SOLE tool-calling contract: the template's native
    #                   contract is suppressed and the identical tool catalog
    #                   is rendered into the prompt text instead.
    # These isolate "better prompting / tool scaffolding" from TT-OPD.
    prompt_mode: Literal["default", "strong_tool", "react"] = "default"


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
    # Which of parse_tool_call's branches actually accepted this turn (see
    # TOOL_CALL_FORMATS). Set from the parser itself rather than re-derived,
    # so the label can never drift from the branch that fired. Needed to make
    # the ReAct arm interpretable: a low score under prompt_mode="react" means
    # something entirely different depending on whether the model emitted ReAct
    # at all.
    tool_call_format: str = ""


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
    # Per-task output-format adherence (see summarize_format_adherence).
    format_adherence: dict = field(default_factory=dict)


def _build_onboarding_guidance(domain: str, prompt_mode: str = "default") -> str:
    """Build compact agent onboarding guidance for the system prompt.

    This injects essential behavioral tips so that ANY model — even one
    that has never seen the full AGENT_GUIDELINE.md — can perform well
    in the GYM on its first attempt.

    ``prompt_mode`` only swaps the two *output-format* sentences (the ones
    that mandate bare JSON), because ReAct emits Thought/Action/Action Input
    prose instead. Everything else — scoring rubric, confidence policy,
    knowledge-tool list and the per-domain workflow tips — is held constant
    across modes, so a mode difference cannot be confounded with lost domain
    grounding. For prompt_mode="default" the returned string is byte-for-byte
    identical to the pre-existing implementation.
    """

    _DOMAIN_TIPS = {
        "clinical_diagnosis": (
            "1. Gather info first: get_patient_info → get_vital_signs → get_lab_results → get_clinical_notes\n"
            "2. Use think() to build differential diagnosis\n"
            "3. Search guidelines for evidence-based management\n"
            "4. Record your diagnosis with ICD-10 code before submitting"
        ),
        "drug_interaction": (
            "1. get_patient_medications → see current med list\n"
            "2. check_interaction or check_all_interactions → check each pair\n"
            "3. search_alternatives → if severe interaction found\n"
            "4. Assess cumulative risk before recommending"
        ),
        "ehr_management": (
            "1. **get_patient_summary(hadm_id=...)** → overview of the admission (ALWAYS start here)\n"
            "2. **get_lab_results(hadm_id=...)** → lab data; then get_lab_trend for specific lab trends\n"
            "3. **get_vital_signs(hadm_id=...)** → vital signs; then detect_vital_alerts for abnormalities\n"
            "4. **get_clinical_scores(hadm_id=...)** → SOFA, NEWS2, qSOFA severity scores\n"
            "5. **get_medication_orders(hadm_id=...)** → current medications\n"
            "6. **get_procedures / get_discharge_summary / get_admission_history** as needed\n"
            "7. Use think() to synthesize findings, then submit_answer\n\n"
            "NOTE: The task provides a **hadm_id** (hospital admission ID). Pass it as hadm_id, NOT patient_id.\n"
            "Example: {\"name\": \"get_patient_summary\", \"arguments\": {\"hadm_id\": \"HADM_10001\"}}"
        ),
        "medical_qa": (
            "1. Analyze the question and identify key concepts\n"
            "2. Search PubMed or medical wiki for evidence\n"
            "3. Browse relevant articles for detailed information\n"
            "4. Use analyze_answer_options for MCQA before submitting"
        ),
        "triage_emergency": (
            "1. get_patient_presentation → assess_airway_breathing → immediately\n"
            "2. get_vital_signs → calculate_gcs for acuity\n"
            "3. calculate_esi_level → determine ESI\n"
            "4. order_stat_labs / order_imaging per protocol\n"
            "5. Submit with ESI level + disposition + orders"
        ),
        "psychiatry": (
            "1. get_patient_presentation → get_psychiatric_history\n"
            "2. perform_mental_status_exam\n"
            "3. Use validated scales: administer_phq9, administer_gad7, assess_suicide_risk\n"
            "4. screen_substance_use (AUDIT/DAST)\n"
            "5. Submit: diagnosis + risk level + treatment plan + disposition"
        ),
        "obstetrics": (
            "1. get_patient_presentation → get_obstetric_history\n"
            "2. assess_fetal_status (FHR, variability, decelerations)\n"
            "3. assess_labor_progress + calculate_bishop_score if applicable\n"
            "4. check_medication_safety (teratogenicity)\n"
            "5. Follow ACOG protocols: check_ob_protocol"
        ),
        "visual_diagnosis": (
            "1. analyze_medical_image with focus areas\n"
            "2. get_patient_context for clinical correlation\n"
            "3. search_similar_cases for comparison\n"
            "4. compare_with_prior if prior studies available\n"
            "5. record_visual_diagnosis with confidence level"
        ),
        "radiology_report": (
            "1. get_study_info + get_clinical_history\n"
            "2. analyze_findings systematically\n"
            "3. get_prior_reports for comparison\n"
            "4. get_reporting_checklist for completeness\n"
            "5. submit_report: indication → technique → findings → impression"
        ),
        "cross_domain": (
            "1. Read the pathway phase context carefully\n"
            "2. Use the phase-specific domain tools (triage, diagnosis, imaging, etc.)\n"
            "3. Use think() to note key findings for continuity across phases\n"
            "4. Provide assessment relevant to the current phase before submitting"
        ),
    }

    tips = _DOMAIN_TIPS.get(domain, "")
    tip_block = f"\n### Domain-Specific Workflow\n{tips}" if tips else ""

    # The only mode-dependent text in this function: the two sentences that
    # hard-code the bare-JSON turn format. ReAct replaces them with the
    # equivalent constraint stated in ReAct syntax.
    if prompt_mode == "react":
        no_prose_rule = (
            "**Do NOT** write a long text response without an Action. Each turn should "
            "contain exactly ONE Thought / Action / Action Input triple — nothing after "
            "the Action Input line."
        )
        one_call_rule = (
            "- **One Action per turn.** Respond with ONLY the "
            "Thought / Action / Action Input triple."
        )
    else:
        no_prose_rule = (
            "**Do NOT** write a long text response without tool calls. Each turn should "
            "contain ONLY a single JSON tool call — no surrounding text."
        )
        one_call_rule = "- **One tool call per turn.** Respond with ONLY the JSON object."

    return f"""## Agent Behavior Guide
**You are scored on 5 dimensions**: Accuracy (30%), Process (25%), Safety (20%), Format (15%), Coherence (10%).

### Tool Usage Guidelines
**Use tools when they add value to your reasoning.** Direct answers without any tool usage receive a premature_stop penalty.

**Recommended workflow:**
1. Use think() to assess whether you need external evidence
2. If the question requires clinical data, guidelines, or literature — call the appropriate tools
3. If you are highly confident from your existing knowledge (e.g. well-known factual questions), you may submit after minimal tool use (think + submit_answer)
4. Use submit_answer to provide your final assessment

{no_prose_rule}

### Confidence-Based Submission
- If you are **highly confident** in the answer after think(), you may call submit_answer without exhaustive searching. Unnecessary tool calls on well-known facts can introduce noise and reduce accuracy.
- For **complex clinical scenarios**, multi-step reasoning, or unfamiliar topics — use 3-8 turns of tool calls to gather evidence before submitting.
- **Always use think() at least once** before submit_answer to demonstrate reasoning.

### Scoring Rules
- **2-8 turns** is the ideal range depending on question complexity.
- **Always end with submit_answer.** Your response is only recorded when you submit.
{one_call_rule}
- **Check safety.** Drug interactions, contraindications, critical values — flag them explicitly.

### Knowledge Search Tools (Available in ALL Domains)
You have access to medical knowledge search. **Search for evidence before clinical decisions.**
- **search(queries, max_results=8)** — Unified search across PubMed, Wikipedia, evidence passages, and guidelines.
- **search_evidence(query, max_results=5, category="")** — Deep search: 581K PubMed/PMC evidence passages.
- **search_guidelines(condition)** — Clinical practice guidelines (AHA, ACOG, SSC, IDSA, etc.).
- **browse(url_or_id)** / **browse_article(pmid)** / **browse_wiki_entry(entry_id)** — Read full content.
{tip_block}"""


# ═══════════════════════════════════════════════════════════════════════
#  Eval-only prompting baselines  (zero training, no reward, no weights)
# ═══════════════════════════════════════════════════════════════════════
# These exist to settle an open question about the method: are TT-OPD's gains
# attributable to it, or merely to better prompting / tool scaffolding? Each mode
# keeps the domain role, policy, scoring rubric, domain workflow tips and
# the tool set FIXED, and varies only the tool-use contract.
PROMPT_MODES = ("default", "strong_tool", "react")


def _iter_tool_fns(tools: Optional[list[dict]]):
    """Yield the OpenAI-format ``function`` sub-dict of each tool spec."""
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and fn.get("name"):
            yield fn


def _tool_names(tools: Optional[list[dict]]) -> list[str]:
    return [fn["name"] for fn in _iter_tool_fns(tools)]


def _render_tool_signature(fn: dict) -> str:
    """Render one tool as ``name(arg: type, opt?: type)`` + its required list.

    Built from the live registry spec, never hard-coded: ``submit_answer``
    has a DIFFERENT signature per domain (medical_qa: answer/reasoning;
    triage_emergency: esi_level/disposition/reasoning; psychiatry: five
    fields), so a hard-coded contract would be wrong in four domains.
    """
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = list(params.get("required") or [])
    parts = [
        f"{k}: {(v or {}).get('type', 'string')}" if k in required
        else f"{k}?: {(v or {}).get('type', 'string')}"
        for k, v in props.items()
    ]
    req = ", ".join(required) if required else "(none)"
    return f"`{fn['name']}({', '.join(parts)})` — required: {req}"


def _submit_fns(tools: Optional[list[dict]]) -> list[dict]:
    """All terminating tools present in this domain (submit_answer / submit_report)."""
    return [fn for fn in _iter_tool_fns(tools) if fn["name"].startswith("submit")]


def render_tool_catalog(tools: Optional[list[dict]]) -> str:
    """Render the FULL tool specification as system-prompt text.

    This is the catalog half of ``apply_chat_template(tools=...)``, extracted
    so a mode can take the catalog WITHOUT taking the model's native
    tool-calling contract along with it (see ``native_tools_for_prompt_mode``).

    Byte-for-byte identical to what the template injects: the Qwen3.5 chat
    template emits ``<tools>`` + one ``tool | tojson`` per line + ``</tools>``,
    and Jinja's ``tojson`` here is exactly ``json.dumps(tool,
    ensure_ascii=False)``. Verified against the real template, not assumed —
    see scripts/rebuttal/verify_react_transcript.py::check_react_tool_catalog.

    Nothing is summarised, filtered, re-ordered or truncated: every tool the
    other prompt modes receive appears here with the same name, the same
    description and the same argument schema, so a mode that uses this render
    is never confounded with a tool-catalog ablation.
    """
    specs = [t for t in (tools or []) if isinstance(t, dict)]
    if not specs:
        return "<tools>\n</tools>"
    body = "\n".join(json.dumps(t, ensure_ascii=False) for t in specs)
    return f"<tools>\n{body}\n</tools>"


def native_tools_for_prompt_mode(
    tools: Optional[list[dict]],
    prompt_mode: str,
) -> Optional[list[dict]]:
    """What to hand ``apply_chat_template(tools=...)`` under this prompt mode.

    Passing ``tools=`` does two things at once: it injects the tool CATALOG,
    and it injects the model's NATIVE tool-calling contract ("If you choose to
    call a function ONLY reply in the following format ... <tool_call>
    <function=...>"). For prompt_mode="react" the second half is exactly the
    competing contract that made the arm unmeasurable — the model answered the
    native contract and emitted ReAct on 0.2% of turns — so react suppresses
    the template injection and carries the catalog itself, as text, via
    ``render_tool_catalog`` inside ``_build_react_block``.

    Every other mode is untouched: ``default`` and ``strong_tool`` return the
    same list object they were given, so their rendered prompt cannot move.
    """
    if prompt_mode == "react":
        return None
    return tools


def _render_submit_contract(tools: Optional[list[dict]]) -> str:
    fns = _submit_fns(tools)
    if not fns:
        return "- (no submit tool exposed in this domain)"
    return "\n".join(f"- {_render_tool_signature(fn)}" for fn in fns)


# Per-domain worked-example content. Tool NAMES are never taken from here —
# they are resolved against the live tool list at render time (see
# _build_react_block), so the example can never cite a tool that does not
# exist in the registry. Only the medical content is curated here.
_DOMAIN_VIGNETTES = {
    "medical_qa": {
        "thought": "The stem describes progressive high-frequency hearing loss on cisplatin. I should confirm the ototoxicity mechanism before choosing between the options rather than answering from memory.",
        "query": "cisplatin ototoxicity mechanism cochlear outer hair cells reactive oxygen species",
        "thought2": "The passages confirm ROS-mediated outer hair cell death, which matches option B.",
        "submit_args": {
            "answer": "B",
            "reasoning": "Cisplatin accumulates in cochlear outer hair cells and generates reactive oxygen species, causing irreversible high-frequency sensorineural hearing loss.",
        },
    },
    "clinical_diagnosis": {
        "thought": "Fever, productive cough, and a focal infiltrate suggest community-acquired pneumonia. Before I commit to a disposition I should check the severity-scoring threshold.",
        "query": "CURB-65 severity score community-acquired pneumonia admission threshold",
        "thought2": "CURB-65 of 2 places this patient above the outpatient threshold, so inpatient management is indicated.",
        "submit_args": {
            "answer": "Community-acquired pneumonia, CURB-65 = 2; admit for inpatient management.",
            "reasoning": "Confusion and age >= 65 give CURB-65 = 2, which guidelines place above the outpatient-treatment threshold.",
        },
    },
    "drug_interaction": {
        "thought": "The patient is on warfarin and is about to start trimethoprim-sulfamethoxazole. I should verify the interaction mechanism and severity before advising.",
        "query": "warfarin trimethoprim-sulfamethoxazole CYP2C9 inhibition INR elevation bleeding risk",
        "thought2": "TMP-SMX inhibits CYP2C9 and displaces warfarin from albumin, so INR will rise sharply.",
        "submit_args": {
            "answer": "Avoid the combination if an alternative antibiotic exists; otherwise reduce the warfarin dose and recheck INR within 3 days.",
            "reasoning": "TMP-SMX inhibits CYP2C9-mediated S-warfarin clearance and displaces protein binding, producing a major, well-documented INR elevation.",
        },
    },
    "ehr_management": {
        "thought": "The admission shows a rising lactate and falling platelets. I should confirm how the severity score is interpreted before summarising the trajectory.",
        "query": "SOFA score interpretation sepsis organ dysfunction mortality thresholds",
        "thought2": "A SOFA rise of >= 2 points defines sepsis-associated organ dysfunction, which this admission meets.",
        "submit_args": {
            "answer": "Sepsis with a 3-point SOFA rise driven by coagulation and renal subscores; escalate monitoring.",
            "reasoning": "Platelet and creatinine trends across the admission produce a SOFA increase of 3, exceeding the >= 2 threshold for sepsis-associated organ dysfunction.",
        },
    },
    "triage_emergency": {
        "thought": "Crushing chest pain with diaphoresis in a 62-year-old is a high-risk presentation. I should confirm the ESI criteria before assigning a level.",
        "query": "ESI level 2 criteria high-risk chest pain emergency severity index",
        "thought2": "This meets the ESI-2 high-risk criterion: it does not require immediate life-saving intervention, but must not wait.",
        "submit_args": {
            "esi_level": 2,
            "disposition": "Immediate ED bed with continuous cardiac monitoring",
            "reasoning": "High-risk chest pain with diaphoresis meets the ESI-2 high-risk criterion; the patient is stable enough to defer ESI-1 resuscitation but cannot wait.",
        },
    },
    "psychiatry": {
        "thought": "The patient reports passive ideation with a recent loss. I should confirm the risk-stratification criteria before assigning a disposition.",
        "query": "Columbia suicide severity rating scale risk stratification disposition criteria",
        "thought2": "Passive ideation without plan or intent, with protective factors present, stratifies as moderate rather than high risk.",
        "submit_args": {
            "diagnosis": "Major depressive disorder, single episode, moderate",
            "risk_level": "moderate",
            "treatment_plan": "Start an SSRI, arrange weekly psychotherapy, and create a written safety plan.",
            "disposition": "Discharge with 72-hour outpatient follow-up and a safety plan",
            "reasoning": "C-SSRS indicates passive ideation without plan or intent, with intact protective factors, so intensive outpatient follow-up is appropriate.",
        },
    },
    "obstetrics": {
        "thought": "The tracing shows recurrent variable decelerations with moderate variability. I should confirm the ACOG category and its management before acting.",
        "query": "ACOG category II fetal heart tracing recurrent variable decelerations intrauterine resuscitation",
        "thought2": "Moderate variability with recurrent variables is Category II, which calls for intrauterine resuscitation rather than immediate delivery.",
        "submit_args": {
            "diagnosis": "Category II fetal heart tracing with recurrent variable decelerations, likely cord compression",
            "management_plan": "Maternal repositioning, IV fluid bolus, supplemental oxygen, and consider amnioinfusion; reassess in 30 minutes.",
            "urgency": "urgent",
            "reasoning": "Preserved moderate variability argues against acidemia, so intrauterine resuscitation with close reassessment precedes any decision for operative delivery.",
        },
    },
    "visual_diagnosis": {
        "thought": "The image shows an opacity in the right lower zone. I should check the differential for that pattern before recording a diagnosis.",
        "query": "chest radiograph right lower lobe consolidation air bronchograms differential diagnosis",
        "thought2": "Air bronchograms within the opacity favour consolidation over effusion or mass.",
        "submit_args": {
            "answer": "Right lower lobe consolidation consistent with lobar pneumonia",
            "reasoning": "A homogeneous right-lower-zone opacity containing air bronchograms and silhouetting the right hemidiaphragm indicates alveolar consolidation.",
        },
    },
    "radiology_report": {
        "thought": "The indication is a solitary pulmonary nodule. I should confirm the current surveillance criteria before writing the impression.",
        "query": "Fleischner Society 2017 guidelines solid pulmonary nodule surveillance size threshold",
        "thought2": "A 6 mm solid nodule in a high-risk patient warrants a 6-12 month follow-up CT.",
        "submit_args": {
            "answer": "6 mm solid right upper lobe nodule; recommend follow-up CT at 6-12 months per Fleischner 2017.",
        },
    },
    "cross_domain": {
        "thought": "This phase hands off from triage to inpatient diagnosis. I should confirm the guideline for the handoff decision before writing the assessment.",
        "query": "sepsis bundle one hour lactate blood cultures antibiotics handoff to inpatient team",
        "thought2": "The one-hour bundle elements are complete, so the handoff can proceed with an explicit lactate recheck.",
        "submit_args": {
            "answer": "One-hour sepsis bundle complete; hand off to the medical team with a 2-hour lactate recheck pending.",
            "reasoning": "Cultures, broad-spectrum antibiotics and a 30 mL/kg fluid bolus were delivered within the hour, so the remaining task is trend reassessment.",
        },
    },
}

_GENERIC_VIGNETTE = {
    "thought": "I should ground this in retrieved evidence rather than answering from memory.",
    "query": "evidence for the clinical question stated in the ticket",
    "thought2": "The retrieved passages are sufficient to answer.",
    "submit_args": {},
}


def _pick_tool(tools: Optional[list[dict]], preferred: list[str]) -> Optional[str]:
    """Return the first name in ``preferred`` that actually exists in ``tools``."""
    available = set(_tool_names(tools))
    for name in preferred:
        if name in available:
            return name
    return None


def _example_submit_args(fn: dict, vignette: dict) -> dict:
    """Fill the submit tool's REQUIRED arguments from the curated vignette.

    Any required argument the vignette does not cover is filled with an
    obvious placeholder, so the rendered example always type-checks against
    the live spec instead of silently omitting a required field.
    """
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = list(params.get("required") or [])
    curated = vignette.get("submit_args") or {}
    args = {}
    for key in required:
        if key in curated:
            args[key] = curated[key]
        else:
            args[key] = 0 if (props.get(key) or {}).get("type") == "integer" else f"<{key}>"
    # Include curated optional args too, when the spec really has them.
    for key, val in curated.items():
        if key not in args and key in props:
            args[key] = val
    return args


def _build_strong_tool_block(domain: str, tools: Optional[list[dict]]) -> str:
    """Base+AR with a materially stronger tool-use contract.

    Targets the three failure modes the paper measured on TT-OPD rollouts
    (5,280 trajectories): 15.8% of tool calls used hallucinated (invalid)
    names; the top 3 tools (search_evidence, think, search_medical_wiki)
    absorbed 72.9% of all invocations; and 50 of 135 tools were never
    invoked at all. Each rule below is aimed at one of those, plus the
    mandatory-first-call / search-budget / submit-contract requirements.
    """
    names = _tool_names(tools)
    n_tools = len(names)
    search_tool = _pick_tool(tools, ["search", "search_evidence"]) or "search"
    submit_contract = _render_submit_contract(tools)
    submit_names = [fn["name"] for fn in _submit_fns(tools)] or ["submit_answer"]
    primary_submit = submit_names[0]

    return f"""## Tool-Use Contract (STRICT — read before every turn)

You have **{n_tools} tools** in this domain. They are listed in the tool
specification you were given. Treat that list as the complete and only set
of actions available to you.

### 1. Evidence before assertion
- Your FIRST turn must be a tool call, never a prose answer.
- Do not state a clinical fact, a number, a dose, a score threshold or a
  guideline recommendation that you have not either (a) read in a tool
  result this episode, or (b) explicitly flagged as recalled-from-memory
  and therefore unverified.
- `think` is **not** evidence. A `think` call retrieves nothing. Never
  count it toward the evidence you have gathered.

### 2. Use the exact tool names — never invent one
- Call ONLY names that appear verbatim in the tool specification. A call to
  a name that is not in that list is a wasted turn: the environment returns
  an error, and you lose one of your limited turns.
- Do not guess a plausible-sounding name, do not pluralise, abbreviate,
  translate or re-case a name, and do not merge two tools into one.
- If you cannot find a tool that fits, fall back to `{search_tool}` rather
  than inventing one.
- Copy argument names verbatim from the specification too. An argument the
  spec does not define is silently dropped.

### 3. Spend the search budget on breadth, not repetition
- Budget: **at most 5 information-gathering calls** before you submit.
- Re-issuing the same search with reworded phrasing almost never returns
  new evidence. If a query returns nothing useful, change the *tool* or the
  *concept*, not the wording.
- Before defaulting to a general search tool, scan the tool list for a
  tool that answers the sub-question directly — most of the domain-specific
  tools are more precise than a keyword search, and most of them go unused.
- Call each distinct tool at most twice per episode.

### 4. Stopping rule — stop early, and stop deliberately
Stop gathering and submit as soon as ANY of these holds:
1. You can name the specific evidence that decides the answer; or
2. Two consecutive calls returned nothing that changed your assessment; or
3. You have used 5 information-gathering calls.
Continuing past the stopping rule does not raise your score — it lowers it,
because unnecessary calls add noise and consume turns you need in order to
submit.

### 5. Submit contract — the episode only counts if you submit
Your answer is recorded **only** when you call a submit tool. An episode
that runs out of turns without one scores zero regardless of how good your
reasoning was. In this domain:
{submit_contract}
- Fill EVERY required argument. A missing required argument invalidates the
  submission.
- Call `{primary_submit}` while you still have at least one turn left.
- Submit exactly once, then stop."""


def _build_react_block(domain: str, tools: Optional[list[dict]]) -> str:
    """Explicit ReAct-style scaffolding over the same tool set.

    Format and terminology follow ReAct: Yao et al., "ReAct: Synergizing
    Reasoning and Acting in Language Models", ICLR 2023 (arXiv:2210.03629) —
    the canonical Thought / Action / Action Input / Observation loop, with
    the Observation supplied by the environment rather than by the model.

    The worked example is assembled from tools resolved against the LIVE
    registry for this domain (see _pick_tool / _example_submit_args), so it
    can never name a tool that does not exist, and it degrades correctly
    under the no_think ablation (which strips `think` from the tool list).

    ── ReAct is the ONLY tool-calling contract in this prompt ──
    ``apply_chat_template(tools=...)`` injects two things that an earlier
    version of this arm could not separate: the tool CATALOG and the model's
    NATIVE call contract ("If you choose to call a function ONLY reply in the
    following format ... <tool_call><function=...>"). Leaving both in place and
    resolving the conflict in prose did not work — measured on Qwen3.5-9B /
    MedQA, the model emitted its native XML on 4,418 of 4,429 turns and ReAct
    on 10 (react_rate 0.002), so the arm's accuracy measured a prompt clash
    rather than ReAct.

    The two halves are now separable. ``render_tool_catalog`` reproduces the
    catalog the template would have injected, byte-for-byte, directly in this
    block, and ``native_tools_for_prompt_mode`` withholds ``tools=`` from the
    template for react only. The model therefore sees the SAME tools, names
    and argument schemas as every other mode — no summary, no subset, so this
    is not a tool-catalog ablation — with ReAct as the single emission format
    it has been given.

    The residual risk that the model follows its pretraining format anyway is
    not hidden but MEASURED: every turn is labelled with the format branch that
    parsed it and reported as ``format_adherence.react_rate`` in the run
    summary, which is the caveat this arm must be read with.
    """
    vignette = _DOMAIN_VIGNETTES.get(domain, _GENERIC_VIGNETTE)
    available = set(_tool_names(tools))

    # Resolve every tool the example will name against the real registry.
    search_tool = _pick_tool(tools, ["search_evidence", "search", "search_guidelines"])
    think_tool = "think" if "think" in available else None
    submit_fns = _submit_fns(tools)
    submit_fn = submit_fns[0] if submit_fns else None

    lines = []
    if think_tool:
        lines.append(f"Thought: {vignette['thought']}")
        lines.append(f"Action: {think_tool}")
        lines.append(
            "Action Input: "
            + json.dumps({"thought": vignette["thought"]}, ensure_ascii=False)
        )
        lines.append("")
        lines.append("Observation: (returned by the environment)")
        lines.append("")

    if search_tool:
        search_args: dict = {"queries": vignette["query"]} if search_tool == "search" \
            else ({"condition": vignette["query"]} if search_tool == "search_guidelines"
                  else {"query": vignette["query"], "max_results": 5})
        lines.append(
            f"Thought: I still need external evidence before I can commit to an answer."
        )
        lines.append(f"Action: {search_tool}")
        lines.append("Action Input: " + json.dumps(search_args, ensure_ascii=False))
        lines.append("")
        lines.append("Observation: (returned by the environment)")
        lines.append("")

    if submit_fn:
        lines.append(f"Thought: {vignette.get('thought2', 'I now have enough evidence to answer.')}")
        lines.append(f"Action: {submit_fn['name']}")
        lines.append(
            "Action Input: "
            + json.dumps(_example_submit_args(submit_fn, vignette), ensure_ascii=False)
        )

    example = "\n".join(lines) if lines else "(no tools available)"
    submit_contract = _render_submit_contract(tools)
    n_tools = len(available)
    catalog = render_tool_catalog(tools)

    return f"""# Tools

You have access to the following {n_tools} functions, and to no others. Each
entry gives the function's exact name, what it does, and the exact names and
types of its arguments.

{catalog}

## Response Format: ReAct (Thought / Action / Action Input / Observation)

Solve this task by interleaving reasoning and acting. On EVERY turn emit
exactly this three-line structure and nothing else:

Thought: <one or two sentences of reasoning about what you know, what is
missing, and which single tool will close that gap>
Action: <the exact name of ONE tool, copied verbatim from the catalog above —
{n_tools} tools are available in this domain>
Action Input: <a single JSON object holding that tool's arguments, using the
argument names exactly as the catalog above defines them>

Then **STOP**. Do not write anything after the `Action Input:` line.

### This is the only calling format you have
The `<tools>` catalog above is the authoritative list of **which tools exist,
what they are named, and what arguments they take**. ReAct is the
authoritative **emission format**, and the only one defined for this task:
every action is issued as a `Thought:` / `Action:` / `Action Input:` triple in
plain text. Do not wrap a call in `<tool_call>` or `<function=...>` tags, do
not emit a bare JSON object, and do not use any other tool-calling syntax you
may have learned — nothing here parses those, and a turn that uses one is a
turn in which you took no action at all.

### The Observation is NOT yours to write
`Observation:` is produced by the environment and handed back to you on the
next turn, prefixed exactly like that. Never write an Observation yourself,
never predict what a tool will return, and never continue the transcript past
your own Action Input. Text you invent after `Action Input:` is not a tool
result — it is a fabrication, and it will be scored as one.

### Loop
Repeat Thought → Action → Action Input, reading each Observation as it
arrives, until you have the evidence you need. Then finish by taking a
submit action. Your answer is recorded only when you do:
{submit_contract}

### Worked example ({domain})
{example}

Follow this format exactly, starting with `Thought:`."""


def build_system_prompt(
    policy: str,
    tools: list[dict],
    domain: str = "clinical_diagnosis",
    task: Optional[dict] = None,
    agent_profile: Optional[dict] = None,
    reward_strategy: str = "grpo",
    prompt_mode: str = "default",
) -> str:
    """Build the system prompt with policy, tool definitions, and adaptive guidance.

    Args:
        policy: Environment policy text
        tools: Tool definitions (OpenAI format)
        domain: Task domain name
        task: Optional task dict for adaptive guidance
        agent_profile: Optional agent reflection/profile for weakness-aware guidance
        reward_strategy: Current reward strategy (grpo/mrpo/sarl/adaptive)
        prompt_mode: Prompting baseline — "default" (Base+AR, unchanged),
            "strong_tool" (stronger tool-use contract) or "react" (explicit
            ReAct scaffolding). Modes COMPOSE with the per-domain role and
            final instruction rather than replacing them, so a mode is never
            confounded with lost domain grounding.

    Returns:
        Complete system prompt with adaptive tool usage guidance
    """
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(
            f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}"
        )
    # How the tool catalog reaches the model, per mode:
    #   default / strong_tool — via apply_chat_template(tools=...), which also
    #       injects the model's native tool-calling contract. Unchanged.
    #   react — rendered into the prompt text by _build_react_block (see
    #       render_tool_catalog), with the template injection suppressed by
    #       native_tools_for_prompt_mode, so ReAct is the only contract present.
    # (A `tool_section = json.dumps(tools, indent=2)` local used to be computed
    # here and never interpolated into any prompt. It is deleted rather than
    # left in place: its presence read as "the prompt already carries the tool
    # spec", which is the misreading that kept the react arm unmeasurable.)

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
    onboarding = _build_onboarding_guidance(domain, prompt_mode=prompt_mode)

    base_prompt = f"""{role}

## Policy
{policy}

{onboarding}

{final_instruction}"""

    # ── Prompting-baseline block (eval-only) ──
    # Appended AFTER the domain-specific final instruction so the domain role,
    # policy, rubric and workflow tips above are preserved verbatim; the mode
    # block only adds/tightens the tool-use contract. For "default" nothing is
    # appended, so the Base+AR prompt is byte-for-byte what it was before.
    if prompt_mode == "strong_tool":
        base_prompt += "\n\n" + _build_strong_tool_block(domain, tools)
    elif prompt_mode == "react":
        base_prompt += "\n\n" + _build_react_block(domain, tools)

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


# ── Strict ReAct extraction (Yao et al., ReAct, ICLR 2023) ──────────────
# Models decorate ReAct headers as `**Action:**`, `### Action:`, `> Action:`.
# Tolerated on both sides of the keyword. "Action" is never followed by ":"
# in "Action Input:", so the action pattern cannot match the input header.
_REACT_DECOR = r'[*_`#>\-\s]*'
_REACT_ACTION_RE = re.compile(
    _REACT_DECOR + r'Action' + _REACT_DECOR + r':' + _REACT_DECOR
    + r'([A-Za-z_][A-Za-z0-9_.\-]*)',
    re.IGNORECASE,
)
_REACT_INPUT_RE = re.compile(
    _REACT_DECOR + r'Action\s+Input' + _REACT_DECOR + r':',
    re.IGNORECASE,
)
# The Action Input payload ends at the next ReAct header on a new line.
# Anchoring to "\n" prevents cutting inside a JSON string value.
_REACT_STOP_RE = re.compile(
    r'\n[*_`#>\-\s]*(?:Observation|Thought|Action|Final\s+Answer)[*_`#\s]*:',
    re.IGNORECASE,
)
_REACT_FENCE_RE = re.compile(r'^```[A-Za-z_]*\s*\n?(.*?)\n?\s*```\s*$', re.DOTALL)


def _react_args(payload: str) -> dict:
    """Coerce a ReAct ``Action Input:`` payload into an arguments dict."""
    payload = payload.strip()
    fence = _REACT_FENCE_RE.match(payload)
    if fence:
        payload = fence.group(1).strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            return {"input": parsed}
    except (json.JSONDecodeError, ValueError):
        pass
    # Brace-balanced scan: recovers the object when the model appends prose.
    start = payload.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(payload)):
            if payload[i] == "{":
                depth += 1
            elif payload[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(payload[start:i + 1])
                        if isinstance(candidate, dict):
                            return candidate
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return {"input": payload}


def _parse_react_block(text: str) -> Optional[dict]:
    """Extract a tool call from canonical ReAct output, or return None.

    Fires ONLY when the text carries BOTH an ``Action:`` header and an
    ``Action Input:`` header. Any text lacking that pair is left entirely to
    the pre-existing branches, so the default (Base+AR) condition is
    unaffected — a bare-JSON turn never reaches this function's body.

    Fixes three ways HEAD mishandled its own ReAct branch:
      * the ``(.+)`` DOTALL tail swallowed a trailing ``Observation:`` /
        ``Thought:`` continuation into the argument string (the ReAct format
        *induces* that continuation — the original paper stops generation at
        ``Observation:``, and neither backend here lists it as a stop token);
      * a fenced ``Action Input: ```json ...``` `` was intercepted upstream by
        the code-block branch, which returned the *argument value* as the tool
        name whenever the arguments contained a ``name``/``action`` key;
      * markdown-decorated headers (``**Action:**``) matched nothing at all,
        so the turn was misread as a final answer and the episode ended.
    """
    m_input = _REACT_INPUT_RE.search(text)
    if not m_input:
        return None

    # Take the Action header CLOSEST to (and before) the Action Input header.
    # Scanning backwards is what stops a stray "…the next action: …" inside
    # the Thought from being mistaken for the action name.
    action_name = None
    for m in _REACT_ACTION_RE.finditer(text):
        if m.end() <= m_input.start():
            action_name = m.group(1).strip()
        else:
            break
    if not action_name:
        return None

    payload = text[m_input.end():]
    m_stop = _REACT_STOP_RE.search(payload)
    if m_stop:
        payload = payload[:m_stop.start()]

    return {"name": action_name, "arguments": _react_args(payload)}


# ── Output-format labels, one per accepting branch of parse_tool_call ──
# Reported by summarize_format_adherence so that a prompting arm's result can
# be read correctly: under prompt_mode="react", a turn labelled anything other
# than a react_* label is a turn where the model ignored the mandated format.
TOOL_CALL_FORMATS = (
    "xml_tool_call",    # <tool_call>{...}</tool_call> and friends
    "xml_qwen35",       # <tool_call><function=name><parameter=k>v</parameter>
    "react_strict",     # Action: / Action Input: header pair (canonical ReAct)
    "json_code_block",  # ```json\n{...}\n```
    "json_direct",      # the whole turn is one JSON object
    "react_loose",      # legacy permissive Action:/Action Input: regex
    "json_embedded",    # a {"name":..,"arguments":{..}} object inside prose
    "json_scan",        # brace-balanced scan recovered an object from prose
    "none",             # no tool call — prose / final answer
)
# Labels that count as "the model actually emitted ReAct". Both are produced by
# an Action:/Action Input: header pair; they are kept distinct in the histogram
# because react_loose indicates the strict extractor declined (e.g. an
# irregularly spaced "Action  Input:" header), which is useful diagnostics.
REACT_FORMATS = frozenset({"react_strict", "react_loose"})


def parse_tool_call(text: str) -> Optional[dict]:
    """Parse a tool call from model output.

    Thin wrapper over :func:`parse_tool_call_with_format`, kept as the public
    entry point (``grpo_trainer`` and the eval scripts call this name).
    """
    return parse_tool_call_with_format(text)[0]


def parse_tool_call_with_format(text: str) -> tuple[Optional[dict], str]:
    """Parse a tool call and report WHICH format branch accepted it.

    Supports formats:
    1. Pure JSON: {"name": "...", "arguments": {...}}
    2. JSON in code block: ```json\\n{...}\\n```
    3. JSON embedded in text with markers
    4. XML-style: <tool_call>...</tool_call>, <|tool_call|>...<|/tool_call|>
    5. ReAct: Action: tool_name\\nAction Input: {...}
    6. Alt keys: function/tool/action instead of name

    Branch order: native XML tool-call tags win outright (they are
    unambiguous); then strict ReAct, which requires BOTH an ``Action:`` and an
    ``Action Input:`` header and therefore cannot trigger on non-ReAct output;
    then the pre-existing JSON branches in their original order.

    Returns:
        ``(tool_call_or_None, format_label)`` where ``format_label`` is one of
        :data:`TOOL_CALL_FORMATS`. The label is emitted by the branch that
        actually returned, so it cannot drift from the parse the way a second,
        separate format-detector would. Branch order and every returned
        tool-call dict are byte-identical to the previous implementation — only
        the extra tuple element is new.
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
                        return norm, "xml_tool_call"
            except json.JSONDecodeError:
                pass

    # ── Try 0b: Qwen3.5 XML tool call format ──
    # Format: <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>
    qwen35_match = re.search(
        r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>',
        text, re.DOTALL,
    )
    if qwen35_match:
        func_name = qwen35_match.group(1).strip()
        params_block = qwen35_match.group(2).strip()
        # Parse <parameter=key>value</parameter> pairs
        param_pairs = re.findall(
            r'<parameter=([^>]+)>(.*?)</parameter>', params_block, re.DOTALL
        )
        arguments = {}
        for key, val in param_pairs:
            val = val.strip()
            # Try to parse as JSON value (number, bool, object, array)
            try:
                arguments[key.strip()] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                arguments[key.strip()] = val
        if func_name:
            return {"name": func_name, "arguments": arguments}, "xml_qwen35"

    # ── Try 0c: Strict ReAct block ──
    # Placed ahead of the code-block branch because an explicit
    # Action:/Action Input: header pair is a stronger signal than a bare
    # ```json fence, and the fence branch was silently mis-binding fenced
    # ReAct arguments to the wrong tool name. Returns None unless BOTH ReAct
    # headers are present, so nothing else changes.
    # (guard on the literal precondition, not on has_react_hint, so that
    #  case variants like "ACTION INPUT:" are covered too)
    if "action input" in text.lower():
        react_strict = _parse_react_block(text)
        if react_strict:
            return react_strict, "react_strict"

    # ── Try 1: Extract JSON from code blocks ──
    code_block_match = re.search(r'```(?:json|tool_call)?\s*\n?({.*?})\s*\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                norm = _normalize_tool_call(parsed)
                if norm:
                    return norm, "json_code_block"
        except json.JSONDecodeError:
            pass

    # ── Try 2: Direct JSON parse ──
    if has_json_hint:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                norm = _normalize_tool_call(parsed)
                if norm:
                    return norm, "json_direct"
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
                        return {"name": tool_name, "arguments": args}, "react_loose"
                except json.JSONDecodeError:
                    # args might be plain text
                    return ({"name": tool_name, "arguments": {"input": args_str}},
                            "react_loose")

    if not has_json_hint:
        return None, "none"

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
                return norm, "json_embedded"
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
                                return norm, "json_scan"
                    except json.JSONDecodeError:
                        pass
                    break

    return None, "none"


# ═══════════════════════════════════════════════════════════════════════
#  Multi-turn transcript rendering  (prompt_mode-aware)
# ═══════════════════════════════════════════════════════════════════════
# The system prompt is only half of a prompting baseline; the other half is
# what the transcript teaches the model once the loop starts. Before this,
# EVERY mode replayed the assistant's turn as canonical bare JSON and returned
# tool results as "Tool result for <name>:", so a react-mode episode contra-
# dicted its own system prompt from turn 1 onward — the model's strongest
# in-context evidence was that assistant turns are bare JSON, and the promised
# `Observation:` never appeared. The two helpers below are the ONLY places the
# transcript is rendered, and both are exact no-ops for "default" and
# "strong_tool" (verified byte-for-byte by
# scripts/rebuttal/verify_react_transcript.py).


def _truncate_at_hallucinated_observation(text: str) -> str:
    """Cut a ReAct turn at the first header AFTER its executed ``Action Input:``.

    ReAct assumes generation STOPS at ``Observation:`` — the environment then
    supplies it. Neither backend here can set that stop string, so a model
    routinely continues its own turn with a fabricated ``Observation:`` and a
    further Thought/Action that the environment never executed. Replaying that
    verbatim would place invented tool output into the context indistinguish-
    ably from real output, i.e. it would let the model feed itself.

    The cut is made at exactly the boundary ``_parse_react_block`` already uses
    to delimit the arguments it executed, so what is replayed is precisely the
    triple the environment acted on — no more, no less. Turns without an
    ``Action Input:`` header, and turns that never run past it, are untouched.
    """
    m_input = _REACT_INPUT_RE.search(text)
    if not m_input:
        return text
    m_stop = _REACT_STOP_RE.search(text, m_input.end())
    if not m_stop:
        return text
    return text[:m_stop.start()].rstrip()


def format_assistant_turn(
    raw_output: str,
    tool_call: Optional[dict],
    prompt_mode: str = "default",
) -> str:
    """Render the assistant's own turn for replay in the conversation history.

    In ``react`` mode the turn is replayed AS THE MODEL WROTE IT — which is
    what ReAct specifies and what the react system prompt promises — with the
    single exception of a self-continued, never-executed tail (see
    _truncate_at_hallucinated_observation). Replaying the model's own text is
    also the honest choice for measurement: normalising a non-ReAct turn into
    ReAct (or into JSON) would let the harness hide the model's real behaviour
    from both the model and the adherence metric.

    Every other mode keeps the pre-existing canonical-JSON rewrite, so the
    ``default`` (Base+AR) condition in the results table is untouched.
    """
    if prompt_mode == "react":
        return _truncate_at_hallucinated_observation(raw_output)
    return json.dumps(tool_call)


def format_tool_observation(
    tool_name: str,
    observation_str: str,
    prompt_mode: str = "default",
) -> str:
    """Render an environment tool result for the next user-role message.

    ``react`` mode uses the canonical ReAct ``Observation:`` prefix (Yao et
    al., ICLR 2023) — the exact string the react system prompt tells the model
    the environment will hand back. Other modes keep the pre-existing
    ``Tool result for <name>:`` prefix verbatim.
    """
    if prompt_mode == "react":
        return f"Observation: {observation_str}"
    return f"Tool result for {tool_name}:\n{observation_str}"


def summarize_format_adherence(turns: list) -> dict:
    """Aggregate per-turn ``tool_call_format`` labels over one episode.

    Without this number a react-arm score is uninterpretable: a low score can
    mean "ReAct scaffolding is weak for this model" or "the model never emitted
    ReAct at all", and those two readings call for opposite conclusions.

    Returns a dict with:
      ``n_turns``        assistant turns in the episode
      ``n_tool_turns``   turns that produced a parsed tool call
      ``n_react``        turns accepted by a ReAct branch (see REACT_FORMATS)
      ``react_rate``     n_react / n_turns  (0.0 when there are no turns)
      ``formats``        histogram over TOOL_CALL_FORMATS labels
    """
    labels = [getattr(t, "tool_call_format", "") or "none" for t in turns]
    hist: dict[str, int] = {}
    for lab in labels:
        hist[lab] = hist.get(lab, 0) + 1
    n_react = sum(1 for lab in labels if lab in REACT_FORMATS)
    n_tool = sum(1 for lab in labels if lab != "none")
    return {
        "n_turns": len(labels),
        "n_tool_turns": n_tool,
        "n_react": n_react,
        "react_rate": (n_react / len(labels)) if labels else 0.0,
        "formats": hist,
    }


def aggregate_format_adherence(per_task: list[dict]) -> dict:
    """Pool per-task adherence dicts into a run-level summary.

    Pooled over TURNS (not averaged over tasks) so a 1-turn task cannot weigh
    as much as an 8-turn one; the per-task mean is reported alongside it.
    """
    valid = [d for d in per_task if d]
    n_turns = sum(d.get("n_turns", 0) for d in valid)
    n_react = sum(d.get("n_react", 0) for d in valid)
    n_tool = sum(d.get("n_tool_turns", 0) for d in valid)
    hist: dict[str, int] = {}
    for d in valid:
        for lab, c in (d.get("formats") or {}).items():
            hist[lab] = hist.get(lab, 0) + c
    return {
        "n_tasks": len(valid),
        "n_turns": n_turns,
        "n_tool_turns": n_tool,
        "n_react": n_react,
        # Turn-pooled: the headline number.
        "react_rate": (n_react / n_turns) if n_turns else 0.0,
        # Task-averaged: reported so a skewed turn distribution is visible.
        "react_rate_task_mean": (
            sum(d.get("react_rate", 0.0) for d in valid) / len(valid)
        ) if valid else 0.0,
        "formats": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
    }


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

    # Skip repair for non-Qwen2.5-VL models (e.g., Qwen3.5)
    model_type = config.get("model_type", "")
    if model_type not in ("qwen2_5_vl", "qwen2_vl"):
        return False

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

        if self.config.backend == "sglang":
            self._load_sglang()
        elif self.config.backend == "vllm":
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

    def _load_sglang(self):
        """Initialize SGLang client (server must be running separately)."""
        from openai import OpenAI

        server_url = self.config.server_url
        if not server_url:
            raise ValueError("server_url must be set for sglang backend")

        self._sglang_client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key="none",
        )

        # Discover model name from server
        try:
            models = self._sglang_client.models.list()
            self._sglang_model_name = models.data[0].id if models.data else "default"
        except Exception:
            self._sglang_model_name = "default"

        # Set tokenizer from the model path for chat template
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True,
        )
        self._is_vl_model = False
        self.processor = None
        self.model = None  # No local model

        logger.info(f"SGLang client initialized: {server_url} (model={self._sglang_model_name})")

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
        # B041: Qwen3.5 SDPA + batch>1 hangs generate(); force eager attention
        self.model = profile.load_model(device_map="auto", attn_implementation="eager")
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
            "vl" in a.lower() or "vision" in a.lower() or "Qwen3_5" in a
            for a in (architectures or [])
        ) or "vl" in model_type.lower() or model_type == "qwen3_5"

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

        is_qwen3_5 = model_type == "qwen3_5"
        is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

        try:
            if is_qwen3_5:
                from transformers import Qwen3_5ForConditionalGeneration
                self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_path, **load_kwargs
                )
            elif is_qwen_vl:
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
    
    def generate(self, messages: list[dict], tools: Optional[list[dict]] = None) -> str:
        """Generate a response from the model.

        Args:
            messages: Conversation messages
            tools: OpenAI-format tool definitions to pass to apply_chat_template(tools=...)
        """
        if self.config.backend == "sglang":
            return self._generate_sglang(messages, tools=tools)
        elif self.config.backend == "vllm":
            return self._generate_vllm(messages, tools=tools)
        else:
            return self._generate_transformers(messages, tools=tools)
    
    def _generate_sglang(self, messages: list[dict], tools: Optional[list[dict]] = None) -> str:
        """Generate with SGLang server via OpenAI-compatible API.

        Supports both text-only and multimodal (image) inputs.
        For multimodal: uses chat.completions with base64-encoded images.
        For text-only: uses completions with locally-rendered chat template.
        """
        # Check if any message has multimodal content (images)
        has_images = any(
            isinstance(msg.get("content"), list) and
            any(c.get("type") == "image" for c in msg["content"] if isinstance(c, dict))
            for msg in messages
        )

        if has_images:
            return self._generate_sglang_multimodal(messages, tools=tools)

        # Text-only: apply chat template locally for exact parity with HF backend
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tools=tools if tools else None,
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

        response = self._sglang_client.completions.create(
            model=self._sglang_model_name,
            prompt=text,
            max_tokens=self.config.max_new_tokens,
            temperature=max(self.config.temperature, 0.01),
            top_p=self.config.top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return response.choices[0].text.strip()

    def _generate_sglang_multimodal(self, messages: list[dict], tools: Optional[list[dict]] = None) -> str:
        """Generate with SGLang for multimodal (image) inputs via chat completions.

        FIXED 2026-07-29: ``tools`` used to be accepted here and never forwarded,
        so on the image benchmarks (vqa_rad, slake, ...) default and strong_tool
        received NO tool catalog — this call is the only way it reaches them.
        The VQA rows would have measured a toolless agent under a heading that
        says tool-using. react was unaffected; it carries its catalog in the
        system prompt.

        Fixed before any VQA number was produced in this reproduction, so no
        reported baseline moves. Note the same gap existed in the original code,
        which means the paper's own Base+AR VQA row may itself be a toolless
        measurement — flagged in the rebuttal notes rather than assumed.

        Both behaviours remain reachable from this one binary via
        ``HCGYM_MULTIMODAL_TOOLS`` (default = forward = fixed), so the two arms
        of that comparison differ in one flag rather than in two code states.
        """
        import base64

        # Convert messages to OpenAI chat format with base64 images
        api_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Multimodal content
                api_content = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image":
                            # Convert file:// URI to base64
                            image_path = part.get("image", "")
                            if image_path.startswith("file://"):
                                image_path = image_path[7:]
                            try:
                                with open(image_path, "rb") as f:
                                    img_data = base64.b64encode(f.read()).decode()
                                # Detect format
                                ext = image_path.rsplit(".", 1)[-1].lower()
                                media_type = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
                                api_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{media_type};base64,{img_data}"}
                                })
                            except Exception as e:
                                logger.warning(f"Failed to load image {image_path}: {e}")
                        elif part.get("type") == "text":
                            api_content.append({"type": "text", "text": part.get("text", "")})
                        else:
                            api_content.append(part)
                    elif isinstance(part, str):
                        api_content.append({"type": "text", "text": part})
                api_messages.append({"role": msg["role"], "content": api_content})
            else:
                api_messages.append(msg)

        create_kwargs = dict(
            model=self._sglang_model_name,
            messages=api_messages,
            max_tokens=self.config.max_new_tokens,
            temperature=max(self.config.temperature, 0.01),
            top_p=self.config.top_p,
        )

        # The ONE thing HCGYM_MULTIMODAL_TOOLS controls: whether the tool
        # catalog reaches the model. Everything above this line — the rendered
        # prompt, the images, the decoding parameters — is built before the
        # switch is read and is identical in both arms, which is what makes the
        # two runs differenceable. See MULTIMODAL_TOOLS_ENV at the top of this
        # module. react arrives with tools=None either way and so is inert here.
        _MM_REQUEST_STATS["multimodal_requests"] += 1
        if tools and multimodal_tools_enabled():
            create_kwargs["tools"] = tools
            _MM_REQUEST_STATS["multimodal_requests_with_tools"] += 1

        response = self._sglang_client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content.strip()

    def _generate_vllm(self, messages: list[dict], tools: Optional[list[dict]] = None) -> str:
        """Generate with vLLM using chat template."""
        from vllm import SamplingParams

        # Use the tokenizer from vLLM engine
        tokenizer = self.model.get_tokenizer()
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tools=tools if tools else None,
                tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        outputs = self.model.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def _generate_transformers(self, messages: list[dict], tools: Optional[list[dict]] = None) -> str:
        """Generate with HuggingFace transformers."""
        import torch

        # For VL models, use processor if available
        if self._is_vl_model and self.processor is not None:
            # Apply chat template (handles both text-only and multimodal messages)
            try:
                text = self.processor.apply_chat_template(
                    messages, tools=tools if tools else None,
                    tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            # Extract vision info from multimodal messages
            image_inputs, video_inputs = None, None
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception:
                pass
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            # Apply chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages, tools=tools if tools else None,
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except Exception:
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
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
        
        # ── no_tools ablation (SciAgentGYM-style w/o tools baseline) ──
        tools_for_prompt = None
        if self.config.no_tools:
            no_tool_prompt = (
                f"You are a medical AI assistant. Answer the following clinical "
                f"question directly without using any tools. Provide your best "
                f"answer based on your medical knowledge.\n\n"
                f"Domain: {self.config.domain}\n"
            )
            messages = [
                {"role": "system", "content": no_tool_prompt},
                {"role": "user", "content": obs},
            ]
        else:
            # Build conversation with adaptive tool guidance
            tools_for_prompt = info["tools"]

            # ── no_think ablation: remove think() from tool definitions ──
            if self.config.no_think and tools_for_prompt:
                tools_for_prompt = [
                    t for t in tools_for_prompt
                    if t.get("function", {}).get("name") != "think"
                ]

            system_prompt = build_system_prompt(
                info["policy"],
                tools_for_prompt,
                domain=self.config.domain,
                task=task,
                agent_profile=getattr(self, "_agent_profile", None),
                reward_strategy=getattr(self, "_reward_strategy", "grpo"),
                prompt_mode=self.config.prompt_mode,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obs},
            ]

        # Store tools for passing to apply_chat_template(tools=...)
        # This ensures the tokenizer converts tool specs to the model's native format
        # (e.g., Qwen3.5 XML format) matching the training-time format.
        # prompt_mode="react" is the exception: the template injection would add
        # a second, competing tool-calling contract, so react carries the same
        # catalog as prompt text instead (native_tools_for_prompt_mode).
        openai_tools = (
            native_tools_for_prompt_mode(tools_for_prompt, self.config.prompt_mode)
            if not self.config.no_tools else None
        )

        try:
            for turn_idx in range(self.config.max_turns):
                turn = TurnRecord(turn_idx=turn_idx)

                # Generate
                t0 = time.time()
                raw_output = self.generate(messages, tools=openai_tools)
                turn.latency_seconds = time.time() - t0
                turn.raw_output = raw_output
                
                logger.debug(f"Turn {turn_idx}: {raw_output[:200]}...")
                
                # Parse tool call (and record WHICH format branch accepted it,
                # so prompt_mode="react" adherence is measurable)
                tool_call, tool_fmt = parse_tool_call_with_format(raw_output)
                turn.tool_call_format = tool_fmt

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
                            messages.append({
                                "role": "assistant",
                                "content": format_assistant_turn(
                                    raw_output, tool_call, self.config.prompt_mode
                                ),
                            })
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

                    # Add to messages. In react mode the assistant turn is
                    # replayed as the model wrote it and the result comes back
                    # as "Observation:" — matching what the react system prompt
                    # promises. Other modes are unchanged.
                    messages.append({
                        "role": "assistant",
                        "content": format_assistant_turn(
                            raw_output, tool_call, self.config.prompt_mode
                        ),
                    })
                    messages.append({
                        "role": "user",
                        "content": format_tool_observation(
                            tool_name, observation_str, self.config.prompt_mode
                        ),
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
            eval_criteria = task.get("evaluation_criteria", {})
            expected_actions = eval_criteria.get("actions", [])
            nl_assertions = eval_criteria.get("nl_assertions", [])
            
            reward_details = compute_composite_reward(
                response=final_answer_text,
                correct_answer=correct_answer,
                tool_call_log=env._tool_call_log,
                expected_actions=expected_actions,
                nl_assertions=nl_assertions,
                is_final=True,
            )
            result.trajectory["reward_details"] = reward_details
            result.trajectory["assertion_evaluation"] = reward_details.get("assertion_details", {})
            result.final_reward = reward_details["total"]

            result.completed = True

        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}")
            result.error = str(e)
            import traceback
            result.error += "\n" + traceback.format_exc()

        # Output-format adherence — computed even on the error path, so a run
        # can never report prompt_mode="react" without the evidence that the
        # model did (or did not) emit ReAct.
        result.format_adherence = summarize_format_adherence(result.turns)
        result.format_adherence["prompt_mode"] = self.config.prompt_mode
        if isinstance(result.trajectory, dict):
            result.trajectory["format_adherence"] = result.format_adherence

        result.end_time = datetime.now().isoformat()
        result.total_latency = sum(t.latency_seconds for t in result.turns)

        logger.info(
            f"Task {task_id}: turns={result.total_turns}, "
            f"action_score={result.action_score:.3f}, "
            f"reward={result.final_reward:.3f}, "
            f"react_rate={result.format_adherence['react_rate']:.2f}, "
            f"latency={result.total_latency:.1f}s"
        )
        
        return result
    
    def _compute_qa_accuracy(self, task: dict, tool_call_log: list) -> float:
        """Check if the agent submitted the correct answer (for QA domains)."""
        correct_answer = task.get("answer", task.get("correct_answer", ""))
        if not correct_answer:
            return 0.0

        # Every form of the gold that a model may legitimately submit. The
        # benchmark loader stores `answer` as the option TEXT while models answer
        # with the LETTER, so a multiple-choice task used to take the free-text
        # branch below and score 0.0 for a correct letter, always. That made the
        # Reflexion ladder report 0.0% on all 400 medqa tasks and all four
        # strategies, against 83.1% for the same model through the eval harness,
        # which does resolve letter against text. It would have done the same to
        # STaR, whose default acceptance signal is this function: it would have
        # accepted no trajectories at all and the baseline would have been vacuous
        # rather than visibly broken.
        options = task.get("options") or {}
        gold_forms = {correct_answer.strip().lower()}
        if options:
            # Resolve the gold to exactly ONE option, and prefer an exact TEXT
            # match over reading the gold as a letter.
            #
            # medqa_709's option texts are literally "B", "C", "D", "E", and its
            # gold is "B" -- which is option A's TEXT, not option B. Adding the
            # letter->text direction unconditionally resolved that gold to option
            # B as well, so submitting "B" or "C" scored 1.0 on a task whose
            # answer is A. Measured: 3 such credits across medqa's 1223 MC tasks.
            key = next(
                (l for l, t in options.items() if isinstance(t, str) and t.strip() == correct_answer.strip()),
                None,
            )
            if key is None and correct_answer.strip() in options:
                key = correct_answer.strip()
            if key is not None:
                gold_forms = {key.strip().lower(), str(options[key]).strip().lower()}
                # 15 of 6,545 multiple-choice tasks are degenerate: an option's
                # TEXT is itself an option LETTER (medqa_709's four options read
                # "B", "C", "D", "E"). Accepting the text form there makes a
                # submitted "B" ambiguous between option B and option A's text,
                # and the models are instructed to answer with a letter, so the
                # letter reading is the one to keep.
                letters = {str(l).strip().upper() for l in options}
                if str(options[key]).strip().upper() in letters:
                    gold_forms = {key.strip().lower()}

        # Find the submit_answer tool call
        for tc in reversed(tool_call_log):
            if tc["tool_name"] == "submit_answer":
                submitted = tc["arguments"].get("answer", "").strip()
                if submitted.lower() in gold_forms:
                    return 1.0
                # A task WITH options is decided here and nowhere else. Falling
                # through to the free-text branch below hands 0.5 to any answer
                # that merely CONTAINS the gold, and one option's text is often a
                # superset of another's: medqa_39's gold is "Arcuate fasciculus"
                # and option D reads "Arcuate fasciculus + inferior frontal gyrus
                # + superior temporal gyrus", so submitting D scored 0.5 on a task
                # whose answer is A. That is the same substring-containment defect
                # the visual-QA scorer was replaced for.
                if options:
                    return 0.0
                # For multiple-choice, compare first letter
                if len(correct_answer.strip()) <= 2:
                    if submitted.upper() == correct_answer.strip().upper():
                        return 1.0
                    # `options` is present but null on 142/3390 train and 20/850
                    # test tasks, so the default in .get() never fires and .get on
                    # None raises. The early return above means this only ever ran
                    # for INCORRECT rollouts, which is why it stayed hidden.
                    correct_text = (options.get(correct_answer.strip()) or "").lower()
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
            "prompt_mode": getattr(self.config, "prompt_mode", "default"),
            "format_adherence": result.format_adherence,
            "turns": [
                {
                    "turn_idx": t.turn_idx,
                    "raw_output": t.raw_output,
                    "parsed_tool_call": t.parsed_tool_call,
                    "tool_call_format": t.tool_call_format,
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
            "prompt_mode": getattr(self.config, "prompt_mode", "default"),
            # Which arm of the multimodal tool-forwarding comparison produced
            # these numbers. Recorded next to prompt_mode, not in a log line,
            # because a VQA score is uninterpretable without it: the same
            # binary can produce a tool-using or a toolless measurement.
            "multimodal_tool_forwarding": multimodal_tool_forwarding(),
            # Which output format the model actually emitted. Under
            # prompt_mode="react" this is the caveat the arm must be read with:
            # the chat template still renders the model's NATIVE tool-call
            # contract alongside the ReAct instruction (see _build_react_block),
            # so react_rate reports how often ReAct actually won.
            "format_adherence": aggregate_format_adherence(
                [r.format_adherence for r in results]
            ),
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
                    "react_rate": r.format_adherence.get("react_rate", 0.0),
                    "formats": r.format_adherence.get("formats", {}),
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
        print(f"  Prompt mode: {summary['prompt_mode']}")
        _mm = summary["multimodal_tool_forwarding"]
        print(
            f"  Multimodal tools: {_mm['arm']} "
            f"({_mm['env_var']}={_mm['env_value']!r}; "
            f"{_mm['multimodal_requests_with_tools']}/{_mm['multimodal_requests']} "
            f"image requests carried the catalog)"
        )
        _fa = summary["format_adherence"]
        print(
            f"  Format adherence: react_rate={_fa['react_rate']:.3f} "
            f"({_fa['n_react']}/{_fa['n_turns']} turns)  formats={_fa['formats']}"
        )
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
