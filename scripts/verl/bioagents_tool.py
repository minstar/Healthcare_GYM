"""BIOAgents Medical Tools for veRL Multi-Turn GRPO Training.

Each tool_config.yaml entry creates a BIOAgentsMedicalTool instance with a specific
tool name (e.g., search_pubmed, get_patient_info). On execute, uses self.name
to dispatch to the appropriate BIOAgents domain toolkit method.

The domain is determined from the dataset's extra_info.domain field via create().
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional
from uuid import uuid4

sys.path.insert(0, os.environ.get("HCGYM_ROOT", str(Path(__file__).resolve().parents[2])))

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

# Map data domains to BIOAgents module names
DATA_TO_BIO_DOMAIN = {
    "text_qa": "medical_qa",
    "multimodal_vqa": "visual_diagnosis",
    "clinical_diagnosis": "clinical_diagnosis",
    "drug_interaction": "drug_interaction",
    "ehr_management": "ehr_management",
    "triage_emergency": "triage_emergency",
    "psychiatry": "psychiatry",
    "obstetrics": "obstetrics",
    "radiology_report": "radiology_report",
}

# Toolkit cache keyed by (domain, instance_id) to avoid stale data across samples
_toolkit_cache: dict[tuple[str, str], Any] = {}
_knowledge_backend = None


def _get_knowledge_backend():
    """Get or init shared knowledge search backend."""
    global _knowledge_backend
    if _knowledge_backend is None:
        try:
            from bioagents.tools.knowledge_tools import MedicalKnowledgeBackend
            db_path = os.environ.get("MEDICAL_FTS_DB", str(Path(os.environ.get("HCGYM_ROOT", Path(__file__).resolve().parents[2])) / "databases" / "medical_knowledge_fts.sqlite"))
            if os.path.exists(db_path):
                _knowledge_backend = MedicalKnowledgeBackend(db_path=db_path)
            else:
                _knowledge_backend = MedicalKnowledgeBackend()
        except Exception as e:
            logger.warning(f"Knowledge backend init failed: {e}")
    return _knowledge_backend


def _get_toolkit(bio_domain: str, db_data: dict = None, instance_id: str = ""):
    """Get or create a domain toolkit, keyed by (domain, instance) to avoid stale data."""
    cache_key = (bio_domain, instance_id)
    if cache_key in _toolkit_cache:
        return _toolkit_cache[cache_key]

    toolkit = None
    try:
        if bio_domain == "medical_qa":
            from bioagents.domains.medical_qa.tools import MedicalQATools
            from bioagents.domains.medical_qa.data_model import MedicalQADB
            db = MedicalQADB.from_dict(db_data) if db_data else MedicalQADB()
            toolkit = MedicalQATools(db)
        elif bio_domain == "clinical_diagnosis":
            from bioagents.domains.clinical_diagnosis.tools import ClinicalTools
            from bioagents.domains.clinical_diagnosis.data_model import ClinicalDB
            db = ClinicalDB.from_dict(db_data) if db_data else ClinicalDB()
            toolkit = ClinicalTools(db)
        elif bio_domain == "drug_interaction":
            from bioagents.domains.drug_interaction.tools import DrugInteractionTools
            from bioagents.domains.drug_interaction.data_model import DrugInteractionDB
            db = DrugInteractionDB.from_dict(db_data) if db_data else DrugInteractionDB()
            toolkit = DrugInteractionTools(db)
        elif bio_domain == "visual_diagnosis":
            from bioagents.domains.visual_diagnosis.tools import VisualDiagnosisTools
            from bioagents.domains.visual_diagnosis.data_model import VisualDiagnosisDB
            db = VisualDiagnosisDB.from_dict(db_data) if db_data else VisualDiagnosisDB()
            toolkit = VisualDiagnosisTools(db)
        elif bio_domain == "ehr_management":
            from bioagents.domains.ehr_management.tools import EHRTools
            from bioagents.domains.ehr_management.data_model import EHRDB
            db = EHRDB.from_dict(db_data) if db_data else EHRDB()
            toolkit = EHRTools(db)
        elif bio_domain == "triage_emergency":
            from bioagents.domains.triage_emergency.tools import TriageEmergencyTools
            from bioagents.domains.triage_emergency.data_model import TriageEmergencyDB
            db = TriageEmergencyDB.from_dict(db_data) if db_data else TriageEmergencyDB()
            toolkit = TriageEmergencyTools(db)
        elif bio_domain == "psychiatry":
            from bioagents.domains.psychiatry.tools import PsychiatryTools
            from bioagents.domains.psychiatry.data_model import PsychiatryDB
            db = PsychiatryDB.from_dict(db_data) if db_data else PsychiatryDB()
            toolkit = PsychiatryTools(db)
        elif bio_domain == "obstetrics":
            from bioagents.domains.obstetrics.tools import ObstetricsTools
            from bioagents.domains.obstetrics.data_model import ObstetricsDB
            db = ObstetricsDB.from_dict(db_data) if db_data else ObstetricsDB()
            toolkit = ObstetricsTools(db)
        elif bio_domain == "radiology_report":
            from bioagents.domains.radiology_report.tools import RadiologyReportTools
            from bioagents.domains.radiology_report.data_model import RadiologyReportDB
            db = RadiologyReportDB.from_dict(db_data) if db_data else RadiologyReportDB()
            toolkit = RadiologyReportTools(db)
    except Exception as e:
        logger.warning(f"Failed to load {bio_domain} tools: {e}")

    # Limit cache size to prevent memory leak (per-instance keying)
    if len(_toolkit_cache) > 256:
        _toolkit_cache.clear()

    _toolkit_cache[cache_key] = toolkit
    return toolkit


# Knowledge search tool names
KNOWLEDGE_TOOLS = frozenset({
    "search", "search_pubmed", "search_medical_wiki",
    "search_medical_literature", "retrieve_evidence",
    "search_knowledge", "search_medical_knowledge",
    "browse_article", "browse_wiki_entry", "search_evidence",
    "search_guidelines", "cross_reference", "get_citation",
    "search_clinical_trials", "search_drug_interactions",
    "search_adverse_effects", "search_diagnostic_accuracy",
    "summarize_evidence",
})

# Tools that handle 'think' and 'submit_answer' locally
SPECIAL_TOOLS = frozenset({"think", "submit_answer", "submit_report"})

# ── Consolidated tool dispatch ──────────────────────────────────────
# Maps (consolidated_tool_name, aspect/type) → original toolkit method name.
# When the model calls a consolidated tool with an aspect/type parameter,
# we route to the original method that the domain toolkit implements.
CONSOLIDATED_DISPATCH = {
    # get_patient_info → aspect-based routing
    ("get_patient_info", "demographics"): "get_patient_info",
    ("get_patient_info", "history"): "get_patient_history",
    ("get_patient_info", "medications"): "get_medications",
    ("get_patient_info", "allergies"): "get_allergy_list",
    ("get_patient_info", "social"): "get_social_history",
    ("get_patient_info", "psychiatric"): "get_psychiatric_history",
    ("get_patient_info", "obstetric"): "get_obstetric_history",
    ("get_patient_info", "immunizations"): "get_immunization_history",
    ("get_patient_info", "summary"): "get_patient_summary",
    ("get_patient_info", "presentation"): "get_patient_presentation",
    ("get_patient_info", "context"): "get_patient_context",
    # get_patient_records → record_type routing
    ("get_patient_records", "notes"): "get_clinical_notes",
    ("get_patient_records", "admissions"): "get_admission_info",
    ("get_patient_records", "discharge"): "get_discharge_summary",
    ("get_patient_records", "procedures"): "get_procedures",
    ("get_patient_records", "medications"): "get_medication_orders",
    ("get_patient_records", "nursing"): "get_nursing_assessments",
    ("get_patient_records", "code_status"): "get_code_status",
    # get_drug_info → aspect routing
    ("get_drug_info", "overview"): "get_drug_info",
    ("get_drug_info", "pharmacokinetics"): "get_pharmacokinetics",
    ("get_drug_info", "dosing"): "check_dosage",
    ("get_drug_info", "monitoring"): "check_therapeutic_drug_monitoring",
    ("get_drug_info", "class"): "search_drugs_by_class",
    # check_interaction → routing (renamed from check_drug_interactions to match model prior)
    ("check_interaction", None): "check_drug_interaction",
    # check_medication_safety → check_type routing
    ("check_medication_safety", "general"): "check_medication_safety",
    ("check_medication_safety", "pregnancy"): "check_pregnancy_safety",
    ("check_medication_safety", "renal"): "calculate_renal_dose_adjustment",
    ("check_medication_safety", "hepatic"): "calculate_hepatic_dose_adjustment",
    ("check_medication_safety", "alternatives"): "search_alternatives",
    # get_vital_signs with trend
    ("get_vital_signs", True): "get_vital_signs_trend",
    ("get_vital_signs", False): "get_vital_signs",
    # get_lab_results with trend
    ("get_lab_results", True): "get_lab_trend",
    ("get_lab_results", False): "get_lab_results",
    # order_test → urgency routing
    ("order_test", "stat"): "order_stat_labs",
    ("order_test", "routine"): "order_lab_test",
    # analyze_medical_image → analysis_type routing
    ("analyze_medical_image", "findings"): "analyze_medical_image",
    ("analyze_medical_image", "classify"): "classify_finding",
    ("analyze_medical_image", "quality"): "assess_image_quality",
    ("analyze_medical_image", "metrics"): "calculate_image_metrics",
    ("analyze_medical_image", "measure"): "measure_lesion",
    ("analyze_medical_image", "annotate"): "annotate_regions",
    ("analyze_medical_image", "track"): "track_lesion_changes",
    # search_imaging_knowledge → search_type routing
    ("search_imaging_knowledge", "knowledge"): "search_imaging_knowledge",
    ("search_imaging_knowledge", "similar_cases"): "search_similar_cases",
    ("search_imaging_knowledge", "compare_prior"): "compare_with_prior",
    ("search_imaging_knowledge", "differential"): "get_differential_visual",
    # calculate_clinical_score → score_name routing
    ("calculate_clinical_score", "curb65"): "calculate_curb65",
    ("calculate_clinical_score", "wells"): "calculate_wells_score",
    ("calculate_clinical_score", "wells_pe"): "calculate_wells_score",
    ("calculate_clinical_score", "chads2_vasc"): "calculate_chads2_vasc",
    ("calculate_clinical_score", "meld"): "calculate_meld_score",
    ("calculate_clinical_score", "apache2"): "calculate_apache2",
    ("calculate_clinical_score", "gcs"): "calculate_gcs",
    ("calculate_clinical_score", "esi"): "calculate_esi_level",
    ("calculate_clinical_score", "sirs"): "calculate_sirs_criteria",
    ("calculate_clinical_score", "qsofa"): "calculate_qsofa",
    ("calculate_clinical_score", "heart"): "calculate_heart_score",
    ("calculate_clinical_score", "trauma"): "calculate_trauma_score",
    ("calculate_clinical_score", "bmi"): "calculate_bmi",
    ("calculate_clinical_score", "bishop"): "calculate_bishop_score",
    ("calculate_clinical_score", "readmission"): "calculate_readmission_risk",
    ("calculate_clinical_score", "gestational_age"): "calculate_gestational_age",
    ("calculate_clinical_score", "modified_bpp"): "calculate_modified_bpp",
    # search_clinical_guidelines → guideline_type routing
    ("search_clinical_guidelines", "treatment"): "search_clinical_guidelines",
    ("search_clinical_guidelines", "protocol"): "check_protocol",
    ("search_clinical_guidelines", "emergency"): "check_ob_protocol",
    ("search_clinical_guidelines", "icd_code"): "lookup_icd_code",
    # record_clinical_action → action_type routing
    ("record_clinical_action", "diagnosis"): "record_diagnosis",
    ("record_clinical_action", "note"): "add_clinical_note",
    ("record_clinical_action", "prescription"): "prescribe_medication",
    ("record_clinical_action", "order"): "place_order",
    ("record_clinical_action", "consult"): "request_consult",
    ("record_clinical_action", "emergency"): "activate_emergency_protocol",
    # perform_assessment → assessment_type routing
    ("perform_assessment", "mental_status"): "perform_mental_status_exam",
    ("perform_assessment", "suicide_risk"): "assess_suicide_risk",
    ("perform_assessment", "violence_risk"): "assess_violence_risk",
    ("perform_assessment", "substance_use"): "screen_substance_use",
    ("perform_assessment", "eating_disorder"): "screen_eating_disorder",
    ("perform_assessment", "capacity"): "assess_capacity",
    ("perform_assessment", "phq9"): "administer_phq9",
    ("perform_assessment", "gad7"): "administer_gad7",
    ("perform_assessment", "mmse"): "administer_mmse",
    ("perform_assessment", "pcl5"): "administer_pcl5",
    ("perform_assessment", "cage"): "administer_cage",
    ("perform_assessment", "sepsis"): "screen_sepsis",
    ("perform_assessment", "gestational_diabetes"): "screen_gestational_diabetes",
    ("perform_assessment", "airway"): "assess_airway_breathing",
    # assess_obstetric_status → assessment_type routing
    ("assess_obstetric_status", "fetal"): "assess_fetal_status",
    ("assess_obstetric_status", "labor"): "assess_labor_progress",
    ("assess_obstetric_status", "biophysical"): "get_biophysical_profile",
    ("assess_obstetric_status", "ctg"): "interpret_ctg",
    ("assess_obstetric_status", "risk"): "get_risk_assessment",
    ("assess_obstetric_status", "gyn"): "get_gyn_assessment",
}

# Parameter that controls sub-routing for each consolidated tool
DISPATCH_PARAM = {
    "get_patient_info": "aspect",
    "get_patient_records": "record_type",
    "get_drug_info": "aspect",
    "check_medication_safety": "check_type",
    "get_vital_signs": "include_trend",
    "get_lab_results": "include_trend",
    "order_test": "urgency",
    "analyze_medical_image": "analysis_type",
    "search_imaging_knowledge": "search_type",
    "calculate_clinical_score": "score_name",
    "search_clinical_guidelines": "guideline_type",
    "record_clinical_action": "action_type",
    "perform_assessment": "assessment_type",
    "assess_obstetric_status": "assessment_type",
}

# Default values for dispatch parameters
DISPATCH_DEFAULTS = {
    "get_patient_info": "demographics",
    "get_patient_records": "notes",
    "get_drug_info": "overview",
    "check_medication_safety": "general",
    "get_vital_signs": False,
    "get_lab_results": False,
    "order_test": "routine",
    "analyze_medical_image": "findings",
    "search_imaging_knowledge": "knowledge",
    "search_clinical_guidelines": "treatment",
}


class BIOAgentsMedicalTool(BaseTool):
    """Medical tool that dispatches to BIOAgents domain toolkits.

    Each instance represents one specific tool (e.g., search_pubmed, get_vital_signs).
    self.name is set from tool_schema.function.name by BaseTool.__init__.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory."""
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        extra_info = create_kwargs.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except (json.JSONDecodeError, TypeError):
                extra_info = {}

        data_domain = extra_info.get("domain", "text_qa")
        bio_domain = DATA_TO_BIO_DOMAIN.get(data_domain, "medical_qa")

        self._instances[instance_id] = {
            "bio_domain": bio_domain,
            "extra_info": extra_info,
        }

        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute this tool with the given parameters.

        self.name identifies which tool function to call.
        parameters are the arguments parsed from the model's tool call.
        """
        instance = self._instances.get(instance_id, {})
        bio_domain = instance.get("bio_domain", "medical_qa")
        tool_name = self.name  # From tool_schema.function.name

        # Handle special tools
        if tool_name in SPECIAL_TOOLS:
            return self._handle_special(tool_name, parameters)

        # Handle knowledge search tools
        if tool_name in KNOWLEDGE_TOOLS:
            return await self._execute_knowledge_search(tool_name, parameters)

        # ── Consolidated tool dispatch ──
        # If tool_name has a dispatch parameter, resolve to original method name
        actual_method_name = tool_name
        toolkit_params = dict(parameters)  # copy to avoid mutating

        # Direct name aliases (no sub-param routing needed)
        DIRECT_ALIASES = {
            "check_interaction": "check_drug_interaction",
        }
        if tool_name in DIRECT_ALIASES:
            actual_method_name = DIRECT_ALIASES[tool_name]

        if tool_name in DISPATCH_PARAM:
            dispatch_key = DISPATCH_PARAM[tool_name]
            dispatch_val = toolkit_params.pop(dispatch_key, DISPATCH_DEFAULTS.get(tool_name))
            lookup = (tool_name, dispatch_val)
            if lookup in CONSOLIDATED_DISPATCH:
                actual_method_name = CONSOLIDATED_DISPATCH[lookup]
            # For calculate_clinical_score, pass score_name as the main arg
            if tool_name == "calculate_clinical_score" and dispatch_val:
                toolkit_params["score_name"] = dispatch_val

        # Dispatch to domain toolkit (per-instance to avoid stale data)
        db_data = instance.get("extra_info", {}).get("db_data")
        toolkit = _get_toolkit(bio_domain, db_data, instance_id=instance_id)
        if toolkit is None:
            return ToolResponse(text=f"Domain '{bio_domain}' tools unavailable."), 0.0, {}

        method = getattr(toolkit, actual_method_name, None)
        if method is None:
            # Fallback: try original tool_name directly
            method = getattr(toolkit, tool_name, None)
        if method is None:
            return ToolResponse(text=f"Tool '{actual_method_name}' not found in {bio_domain}."), 0.0, {}

        try:
            result = method(**toolkit_params)
            if isinstance(result, (dict, list)):
                text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
            else:
                text = str(result)
            return ToolResponse(text=text), 0.0, {"tool_name": actual_method_name}
        except Exception as e:
            return ToolResponse(text=f"Error: {e}"), 0.0, {"error": str(e)}

    def _handle_special(
        self, tool_name: str, parameters: dict
    ) -> tuple[ToolResponse, float, dict]:
        """Handle think and submit_answer tools."""
        if tool_name == "think":
            thought = parameters.get("thought", parameters.get("reasoning", ""))
            return ToolResponse(text=f"Recorded: {thought[:200]}"), 0.0, {}
        elif tool_name in ("submit_answer", "submit_report"):
            answer = parameters.get("answer", "")
            reasoning = parameters.get("reasoning", "")
            return ToolResponse(text=f"Answer submitted: {answer}"), 0.0, {
                "answer": answer, "reasoning": reasoning
            }
        return ToolResponse(text="Unknown special tool."), 0.0, {}

    async def _execute_knowledge_search(
        self, tool_name: str, params: dict
    ) -> tuple[ToolResponse, float, dict]:
        """Execute knowledge search tools."""
        query = params.get("query", "")
        max_results = int(params.get("max_results", 5))

        backend = _get_knowledge_backend()
        if backend is None:
            return ToolResponse(text="Knowledge search unavailable."), 0.0, {}

        try:
            # Route to specific search method if available
            if hasattr(backend, tool_name):
                method = getattr(backend, tool_name)
                result = method(**params)
            else:
                result = backend.search(query, limit=max_results)

            if result:
                text = json.dumps(
                    result[:max_results] if isinstance(result, list) else result,
                    indent=2, ensure_ascii=False, default=str,
                )
            else:
                text = f"No results for: {query}"
            return ToolResponse(text=text), 0.0, {"tool_name": tool_name}
        except Exception as e:
            return ToolResponse(text=f"Search error: {e}"), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Tool reward is handled by reward_fn.py. Return 0 here."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance."""
        self._instances.pop(instance_id, None)
