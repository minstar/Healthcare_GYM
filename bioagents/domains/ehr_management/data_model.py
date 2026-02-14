"""Data models for the EHR Management domain.

Defines the Electronic Health Record database schema including:
- Admission records (ADT events)
- Lab result trends (time-series)
- Vital sign monitoring (time-series)
- Procedures and imaging
- Discharge summaries and follow-up
- Quality indicators (readmission risk, mortality prediction)

Reference: MIMIC-III/IV schema (evaluations/mimic-code)
"""

import os
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from bioagents.environment.db import DB


# =============================================================
# Sub-models — EHR records
# =============================================================


class Demographics(BaseModel):
    """Patient demographics (MIMIC patients table)."""
    patient_id: str = Field(description="Unique patient identifier (MRN)")
    name: str = Field(description="Patient name (de-identified)")
    age: int = Field(description="Patient age at admission")
    sex: Literal["M", "F", "O"] = Field(description="Biological sex")
    date_of_birth: str = Field(description="Date of birth (YYYY-MM-DD)")
    ethnicity: str = Field(default="", description="Self-reported ethnicity")
    language: str = Field(default="English")
    insurance: str = Field(default="", description="Insurance type")
    marital_status: str = Field(default="", description="Marital status")


class Admission(BaseModel):
    """A single hospital admission episode (MIMIC admissions table)."""
    hadm_id: str = Field(description="Hospital admission ID")
    patient_id: str = Field(description="Patient MRN")
    admit_time: str = Field(description="Admission datetime (YYYY-MM-DD HH:MM)")
    discharge_time: Optional[str] = Field(default=None, description="Discharge datetime")
    admit_type: Literal["emergency", "urgent", "elective", "newborn"] = Field(
        description="Admission type"
    )
    admit_location: str = Field(default="", description="Admitted from (e.g., ER, transfer)")
    discharge_location: Optional[str] = Field(default=None, description="Discharged to")
    diagnosis_at_admission: str = Field(default="", description="Primary diagnosis at admission")
    icd_codes: List[str] = Field(default_factory=list, description="ICD-10 diagnosis codes")
    drg_code: Optional[str] = Field(default=None, description="DRG code for billing")
    los_days: Optional[float] = Field(default=None, description="Length of stay (days)")
    icu_stays: List[str] = Field(default_factory=list, description="ICU stay IDs (if any)")
    is_readmission: bool = Field(default=False, description="30-day readmission flag")


class ICUStay(BaseModel):
    """An ICU stay episode (MIMIC icustays table)."""
    icustay_id: str = Field(description="ICU stay ID")
    hadm_id: str = Field(description="Hospital admission ID")
    patient_id: str = Field(description="Patient MRN")
    icu_type: str = Field(description="ICU type (MICU, SICU, CCU, NICU, etc.)")
    intime: str = Field(description="ICU admission time")
    outtime: Optional[str] = Field(default=None, description="ICU discharge time")
    los_icu_hours: Optional[float] = Field(default=None, description="ICU length of stay (hours)")


class LabEvent(BaseModel):
    """A single lab measurement (MIMIC labevents table)."""
    itemid: str = Field(description="Lab item identifier")
    label: str = Field(description="Lab test name (e.g., 'Creatinine', 'Hemoglobin')")
    value: float = Field(description="Numeric result value")
    valueuom: str = Field(default="", description="Unit of measurement")
    flag: Optional[Literal["normal", "abnormal", "delta"]] = Field(default=None)
    ref_range_lower: Optional[float] = Field(default=None)
    ref_range_upper: Optional[float] = Field(default=None)
    charttime: str = Field(description="Measurement datetime")


class VitalEvent(BaseModel):
    """A single vital sign measurement (MIMIC chartevents)."""
    charttime: str = Field(description="Measurement datetime")
    heart_rate: Optional[int] = Field(default=None, description="Heart rate (bpm)")
    sbp: Optional[int] = Field(default=None, description="Systolic blood pressure (mmHg)")
    dbp: Optional[int] = Field(default=None, description="Diastolic blood pressure (mmHg)")
    mean_bp: Optional[int] = Field(default=None, description="Mean arterial pressure (mmHg)")
    resp_rate: Optional[int] = Field(default=None, description="Respiratory rate (breaths/min)")
    temperature: Optional[float] = Field(default=None, description="Temperature (°C)")
    spo2: Optional[int] = Field(default=None, description="SpO2 (%)")
    fio2: Optional[float] = Field(default=None, description="FiO2 fraction")
    gcs_total: Optional[int] = Field(default=None, description="Glasgow Coma Scale total")


class MedicationOrder(BaseModel):
    """A medication order (MIMIC prescriptions table)."""
    order_id: str = Field(description="Order identifier")
    drug: str = Field(description="Drug generic name")
    drug_type: Literal[
        "MAIN", "BASE", "ADDITIVE",
        "BLOOD", "ELECTROLYTE", "TPN", "LIPID", "ANTIBIOTICS",
    ] = Field(default="MAIN")
    dose_val: str = Field(default="", description="Dose value")
    dose_unit: str = Field(default="", description="Dose unit")
    route: str = Field(default="", description="Route of administration (PO, IV, etc.)")
    frequency: str = Field(default="", description="Administration frequency")
    start_time: str = Field(description="Order start datetime")
    end_time: Optional[str] = Field(default=None, description="Order end datetime")
    status: Literal["active", "discontinued", "completed"] = Field(default="active")


class Procedure(BaseModel):
    """A procedure or intervention (MIMIC procedureevents)."""
    procedure_id: str = Field(description="Procedure identifier")
    procedure_name: str = Field(description="Procedure name")
    icd_procedure_code: Optional[str] = Field(default=None, description="ICD procedure code")
    procedure_time: str = Field(description="Procedure datetime")
    performed_by: str = Field(default="", description="Performing clinician")
    notes: str = Field(default="", description="Procedure notes")
    outcome: str = Field(default="", description="Procedure outcome")


class DischargeSummary(BaseModel):
    """Discharge summary note (MIMIC noteevents)."""
    note_id: str = Field(description="Note identifier")
    hadm_id: str = Field(description="Hospital admission ID")
    chartdate: str = Field(description="Note date")
    category: str = Field(default="Discharge summary")
    text: str = Field(description="Full discharge summary text")
    diagnoses: List[str] = Field(default_factory=list, description="Discharge diagnoses")
    discharge_medications: List[str] = Field(default_factory=list, description="Discharge meds")
    follow_up_instructions: str = Field(default="", description="Follow-up plan")


class ClinicalScore(BaseModel):
    """A calculated clinical severity score."""
    score_name: str = Field(description="Score name (SOFA, APACHE-II, SAPS-II, NEWS, etc.)")
    score_value: float = Field(description="Calculated score value")
    interpretation: str = Field(default="", description="Clinical interpretation")
    components: Dict[str, float] = Field(default_factory=dict, description="Score components")
    calculated_at: str = Field(description="Calculation datetime")


class QualityIndicator(BaseModel):
    """Hospital quality / outcome indicator for the admission."""
    readmission_risk: float = Field(default=0.0, description="30-day readmission risk (0-1)")
    mortality_risk: float = Field(default=0.0, description="In-hospital mortality risk (0-1)")
    expected_los: float = Field(default=0.0, description="Expected length of stay (days)")
    sepsis_flag: bool = Field(default=False, description="Sepsis-3 criteria met?")
    aki_stage: Optional[int] = Field(default=None, description="Acute kidney injury stage (0-3)")
    notes: str = Field(default="")


# =============================================================
# Composite patient EHR record
# =============================================================


class EHRRecord(BaseModel):
    """Complete EHR record for a single patient admission."""
    demographics: Demographics
    admission: Admission
    icu_stays: List[ICUStay] = Field(default_factory=list)
    lab_events: List[LabEvent] = Field(default_factory=list)
    vital_events: List[VitalEvent] = Field(default_factory=list)
    medication_orders: List[MedicationOrder] = Field(default_factory=list)
    procedures: List[Procedure] = Field(default_factory=list)
    discharge_summary: Optional[DischargeSummary] = Field(default=None)
    clinical_scores: List[ClinicalScore] = Field(default_factory=list)
    quality_indicators: Optional[QualityIndicator] = Field(default=None)
    prior_admissions: List[str] = Field(
        default_factory=list, description="Prior hadm_ids for this patient"
    )


# =============================================================
# Main Database
# =============================================================


class EHRDB(DB):
    """EHR Management domain database.

    Contains electronic health records for multiple patient admissions,
    supporting clinical queries, trend analysis, outcome prediction,
    and discharge planning tasks.
    """
    records: Dict[str, EHRRecord] = Field(
        default_factory=dict,
        description="EHR records indexed by hadm_id",
    )
    patient_index: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping from patient_id → list of hadm_ids",
    )
    lab_reference_ranges: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Lab reference ranges: {label: {lower, upper, unit}}",
    )
    icd_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="ICD-10 code → description mapping",
    )
    query_log: List[dict] = Field(
        default_factory=list,
        description="Log of EHR queries performed by the agent",
    )


# =============================================================
# Data paths
# =============================================================

_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "ehr_management",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> EHRDB:
    """Load the EHR management database."""
    return EHRDB.load(DB_PATH)
