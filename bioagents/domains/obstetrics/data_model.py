"""Data models for the Obstetrics & Gynecology domain.

Simulates a prenatal clinic and labor & delivery unit with:
- Prenatal patient records
- Fetal monitoring strips
- Labor progress tracking
- Gynecologic assessments
- Risk scoring (Bishop, ACOG risk factors)
"""

import os
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, Field

from bioagents.environment.db import DB


class PrenatalLab(BaseModel):
    """Prenatal laboratory result."""
    test_name: str
    value: str
    unit: str = ""
    reference_range: str = ""
    flag: str = Field(default="normal", description="normal/high/low/critical")
    trimester: Union[int, str] = Field(default=1, description="1, 2, or 3")
    date: str = ""


class FetalMonitoring(BaseModel):
    """Fetal heart rate monitoring strip interpretation."""
    baseline_fhr: Optional[int] = Field(default=None, description="Baseline fetal heart rate (bpm)")
    variability: Optional[str] = Field(default=None, description="absent/minimal/moderate/marked")
    accelerations: Union[bool, str] = Field(default=True, description="Presence of accelerations")
    decelerations: str = Field(default="none", description="none/early/variable/late/prolonged")
    contractions: Union[Dict, str] = Field(default_factory=dict, description="Frequency, duration, intensity")
    category: Union[int, str] = Field(default=1, description="Category I, II, or III")
    interpretation: str = ""


class LaborProgress(BaseModel):
    """Labor progress record."""
    cervical_dilation_cm: float = Field(
        description="0-10 cm",
        validation_alias=AliasChoices('cervical_dilation_cm', 'dilation_cm'),
    )
    effacement_percent: int = Field(description="0-100%")
    station: int = Field(description="-5 to +5")
    membrane_status: str = Field(default="intact", description="intact/ruptured/AROM/SROM")
    contraction_pattern: str = ""
    time_in_labor_hours: float = 0
    phase: str = Field(default="latent", description="latent/active/transition/second_stage/third_stage")


class ObPatient(BaseModel):
    """Obstetric / Gynecologic patient."""
    patient_id: str
    name: str = ""
    age: int
    gravida: int = Field(default=1, description="Total pregnancies")
    para: int = Field(default=0, description="Deliveries >= 20 weeks")
    gestational_age_weeks: Optional[float] = Field(default=None, description="Gestational age in weeks")
    edd: str = Field(default="", description="Estimated due date")
    chief_complaint: str = ""
    prenatal_labs: Union[List[PrenatalLab], Dict[str, Any]] = Field(default_factory=list)
    fetal_monitoring: Optional[FetalMonitoring] = None
    labor_progress: Optional[LaborProgress] = None
    risk_factors: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    obstetric_history: List[str] = Field(default_factory=list)
    surgical_history: List[str] = Field(default_factory=list)
    medications: List[Dict] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vitals: Dict = Field(default_factory=dict)
    blood_type: str = ""
    gbs_status: str = Field(default="unknown", description="positive/negative/unknown")
    placenta_location: str = ""
    amniotic_fluid_index: Optional[float] = None
    bishop_score: Union[int, Dict[str, Any], None] = None
    biophysical_profile: Optional[Dict] = None
    correct_diagnosis: str = ""
    correct_management: Union[List[str], str] = Field(default_factory=list)
    urgency: str = Field(default="routine", description="routine/urgent/emergent")

    # Gynecology-specific
    gyn_complaint: Union[str, Dict[str, Any]] = ""
    last_menstrual_period: str = ""
    menstrual_history: Union[str, Dict[str, Any]] = ""
    contraceptive_use: str = ""
    pap_smear_history: str = ""
    stis_history: List[str] = Field(default_factory=list)


class ObstetricsDB(DB):
    """Obstetrics & Gynecology domain database."""
    patients: Dict[str, ObPatient] = Field(default_factory=dict)
    protocols: Dict[str, Dict] = Field(default_factory=dict, description="OB protocols (e.g., shoulder dystocia, PPH)")
    medication_formulary: Dict[str, Dict] = Field(default_factory=dict)
    reference_ranges: Dict[str, Dict] = Field(default_factory=dict, description="Trimester-specific lab ranges")


_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "obstetrics",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> ObstetricsDB:
    return ObstetricsDB.load(DB_PATH)
