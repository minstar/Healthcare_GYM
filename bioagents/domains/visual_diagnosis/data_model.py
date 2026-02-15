"""Data models for the Visual Diagnosis domain.

Defines the medical imaging database schema including:
- Medical images (X-ray, CT, MRI, pathology, etc.)
- Image reports and findings
- Visual QA questions and annotations
- Patient context for image interpretation
"""

import os
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from bioagents.environment.db import DB


# --- Sub-models ---


class ImageMetadata(BaseModel):
    """Metadata for a medical image."""
    image_id: str = Field(description="Unique image identifier")
    modality: Literal[
        "xray", "ct", "mri", "ultrasound", "pathology",
        "dermoscopy", "fundoscopy", "endoscopy",
        "mammography", "echocardiogram", "other"
    ] = Field(description="Imaging modality")
    body_part: str = Field(default="", description="Body part / region (e.g., 'chest', 'abdomen', 'brain')")
    view: str = Field(default="", description="View type (e.g., 'AP', 'lateral', 'axial')")
    description: str = Field(default="", description="Brief description of the image")
    image_path: Optional[str] = Field(default=None, description="Path to the image file (optional)")
    source_dataset: str = Field(default="", description="Source dataset (VQA-RAD, SLAKE, PathVQA, PMC-VQA)")


class ImageFinding(BaseModel):
    """A specific finding in a medical image."""
    finding_id: str = Field(description="Unique finding identifier")
    description: str = Field(description="Description of the finding")
    location: str = Field(default="", description="Location in the image (e.g., 'right lower lobe')")
    severity: Literal["normal", "mild", "moderate", "severe", "critical"] = Field(default="normal")
    confidence: Literal["low", "moderate", "high"] = Field(default="moderate")
    clinical_significance: str = Field(default="", description="Clinical significance of the finding")


class ImageReport(BaseModel):
    """Radiology / pathology report for an image."""
    report_id: str = Field(description="Unique report identifier")
    image_id: str = Field(description="Associated image ID")
    report_type: Literal[
        "radiology", "pathology", "dermatology", "ophthalmology",
        "cardiology", "other"
    ] = Field(default="radiology")
    indication: str = Field(default="", description="Clinical indication for the study")
    findings: List[ImageFinding] = Field(default_factory=list, description="Detailed findings")
    impression: str = Field(description="Overall impression / conclusion")
    technique: str = Field(default="", description="Imaging technique details")
    comparison: str = Field(default="", description="Comparison with prior studies")
    radiologist: str = Field(default="AI Analysis", description="Reporting radiologist")


class VisualQuestion(BaseModel):
    """A visual medical question about an image."""
    question_id: str = Field(description="Unique question identifier")
    image_id: str = Field(description="Associated image ID")
    question: str = Field(description="The question text")
    question_type: Literal["yes_no", "choice", "open_ended", "counting", "location"] = Field(
        default="open_ended"
    )
    answer: str = Field(description="Correct answer")
    options: Optional[List[str]] = Field(default=None, description="Answer options for choice questions")
    explanation: str = Field(default="", description="Explanation for the answer")
    category: str = Field(default="general", description="Medical category")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")
    source_dataset: str = Field(default="", description="Source VQA dataset")


class PatientImageContext(BaseModel):
    """Patient context relevant to image interpretation."""
    patient_id: str = Field(description="Patient identifier (may be anonymized)")
    age: Optional[int] = Field(default=None, description="Patient age")
    sex: Optional[str] = Field(default=None, description="Patient sex")
    clinical_history: str = Field(default="", description="Relevant clinical history")
    presenting_complaint: str = Field(default="", description="Presenting complaint")
    prior_diagnoses: List[str] = Field(default_factory=list, description="Prior diagnoses")


class SimilarCase(BaseModel):
    """A similar case for reference / comparison."""
    case_id: str = Field(description="Case identifier")
    image_id: str = Field(description="Image ID of the similar case")
    diagnosis: str = Field(description="Diagnosis of the similar case")
    similarity_score: float = Field(default=0.0, description="Similarity score (0-1)")
    key_features: List[str] = Field(default_factory=list, description="Key visual features")


# --- Main Database ---


class VisualDiagnosisDB(DB):
    """Visual Diagnosis domain database.

    Contains medical images, reports, VQA questions, and patient contexts
    for the visual medical diagnosis simulation.
    """
    images: Dict[str, ImageMetadata] = Field(
        default_factory=dict,
        description="Medical images indexed by image_id",
    )
    reports: Dict[str, ImageReport] = Field(
        default_factory=dict,
        description="Image reports indexed by report_id",
    )
    questions: Dict[str, VisualQuestion] = Field(
        default_factory=dict,
        description="Visual QA questions indexed by question_id",
    )
    patient_contexts: Dict[str, PatientImageContext] = Field(
        default_factory=dict,
        description="Patient contexts indexed by patient_id",
    )
    similar_cases: Dict[str, List[SimilarCase]] = Field(
        default_factory=dict,
        description="Similar cases indexed by image_id",
    )
    analysis_log: List[dict] = Field(
        default_factory=list,
        description="Log of image analyses performed",
    )


# --- Data paths ---

_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "visual_diagnosis",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> VisualDiagnosisDB:
    """Load the visual diagnosis database."""
    return VisualDiagnosisDB.load(DB_PATH)
