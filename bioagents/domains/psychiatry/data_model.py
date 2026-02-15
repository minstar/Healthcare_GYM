"""Data models for the Psychiatry / Mental Health domain.

Simulates a psychiatric clinic and emergency setting with:
- Patient mental health assessments (PHQ-9, GAD-7, Columbia, MMSE)
- Psychiatric medication management
- Suicide/self-harm risk assessment
- Substance use screening (AUDIT, DAST)
- Treatment plans and safety planning
"""

import os
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from bioagents.environment.db import DB


class MentalStatusExam(BaseModel):
    """Mental Status Examination findings."""
    appearance: str = Field(default="", description="General appearance, grooming, hygiene")
    behavior: str = Field(default="", description="Psychomotor activity, eye contact, cooperation")
    speech: str = Field(default="", description="Rate, rhythm, volume, tone")
    mood: str = Field(default="", description="Patient's stated mood")
    affect: str = Field(default="", description="Observed emotional expression")
    thought_process: str = Field(default="", description="Linear, tangential, flight of ideas, etc.")
    thought_content: str = Field(default="", description="Delusions, obsessions, suicidal/homicidal ideation")
    perceptions: str = Field(default="", description="Hallucinations (auditory, visual, etc.)")
    cognition: str = Field(default="", description="Orientation, attention, memory")
    insight: str = Field(default="good/fair/poor")
    judgment: str = Field(default="good/fair/poor")


class ScreeningScore(BaseModel):
    """Standardized psychiatric screening result."""
    instrument: str = Field(description="e.g., PHQ-9, GAD-7, AUDIT, Columbia-SSRS")
    total_score: int = Field(description="Total score")
    severity: str = Field(description="Severity category")
    item_responses: Dict[str, int] = Field(default_factory=dict, description="Individual item scores")
    risk_flags: List[str] = Field(default_factory=list, description="Flagged risk items")
    interpretation: str = Field(default="")


class PsychMedication(BaseModel):
    """Psychiatric medication record."""
    drug_name: str
    drug_class: str = Field(description="SSRI, SNRI, antipsychotic, benzodiazepine, mood stabilizer, etc.")
    dosage: str
    frequency: str
    start_date: str = ""
    indication: str = ""
    side_effects_reported: List[str] = Field(default_factory=list)
    adherence: str = Field(default="good", description="good/fair/poor/unknown")


class SafetyPlan(BaseModel):
    """Suicide safety plan (Stanley-Brown model)."""
    warning_signs: List[str] = Field(default_factory=list)
    coping_strategies: List[str] = Field(default_factory=list)
    social_contacts_distraction: List[str] = Field(default_factory=list)
    professionals_to_contact: List[str] = Field(default_factory=list)
    crisis_resources: List[str] = Field(default_factory=list)
    means_restriction: str = ""


class PsychPatient(BaseModel):
    """Patient presenting for psychiatric evaluation."""
    patient_id: str
    name: str = ""
    age: int
    sex: Literal["M", "F", "NB"] = "M"
    chief_complaint: str = ""
    referral_source: str = Field(default="self", description="self, PCP, ED, court, family")
    psychiatric_history: Union[List[str], Dict[str, Any]] = Field(default_factory=list)
    current_diagnoses: List[Dict] = Field(default_factory=list, description="[{diagnosis, icd10, status}]")
    medications: List[PsychMedication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    medical_comorbidities: List[str] = Field(default_factory=list)
    substance_use: Dict = Field(default_factory=dict, description="{alcohol, tobacco, cannabis, opioids, stimulants, ...}")
    mental_status_exam: Optional[MentalStatusExam] = None
    screening_scores: Union[List[ScreeningScore], Dict[str, Any]] = Field(default_factory=list)
    safety_plan: Optional[SafetyPlan] = None
    social_support: str = ""
    housing_status: str = Field(default="stable")
    employment: str = ""
    legal_issues: str = ""
    trauma_history: str = ""
    family_psychiatric_history: Union[List[str], str] = Field(default_factory=list)
    vitals: Dict = Field(default_factory=dict)
    correct_diagnosis: str = ""
    correct_treatment_plan: List[str] = Field(default_factory=list)
    suicide_risk_level: str = Field(default="low", description="low/moderate/high/imminent")


class PsychiatryDB(DB):
    """Psychiatry domain database."""
    patients: Dict[str, PsychPatient] = Field(default_factory=dict)
    screening_templates: Dict[str, Dict] = Field(default_factory=dict, description="Screening instrument templates")
    medication_formulary: Dict[str, Dict] = Field(default_factory=dict, description="Available psychiatric medications")
    treatment_guidelines: Dict[str, Dict] = Field(default_factory=dict, description="Evidence-based guidelines")


_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "psychiatry",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> PsychiatryDB:
    return PsychiatryDB.load(DB_PATH)
