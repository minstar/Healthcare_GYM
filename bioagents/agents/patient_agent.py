"""Patient Agent for Multi-Agent Clinical Simulation.

This is the CRITICAL missing piece identified from competitor analysis:
- AgentClinic: Patient agent with 24 cognitive biases
- DoctorAgent-RL: Progressive symptom revelation
- MedAgentSim: Doctor + Patient + Measurement agents
- Agent Hospital: Patient agents for 10K+ encounters

Our Patient Agent provides:
1. **Progressive Symptom Revelation** — Doesn't dump everything at once
2. **Realistic Patient Behavior** — Anxiety, confusion, minimization, over-reporting
3. **History Consistency** — Maintains a coherent clinical story
4. **Cognitive Bias Simulation** — Anchoring, confirmation, demographic biases
5. **Controllable Difficulty** — Easy (cooperative) → Hard (vague, anxious, combative)

The Patient Agent interacts with the Doctor Agent (the model being trained)
through a turn-based dialogue system.

Reference papers:
  - AgentClinic (Schmidgall et al., 2024): arXiv:2405.07960
  - DoctorAgent-RL (2025): arXiv:2505.19630
  - MedAgentSim (MICCAI 2025)
  - Agent Hospital (Li et al., 2024): arXiv:2405.02957
"""

import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


# ============================================================
# 1. Patient Profile & Clinical Case
# ============================================================

class PatientPersonality(Enum):
    """Patient communication styles that affect information revelation."""
    COOPERATIVE = "cooperative"         # Answers clearly, volunteers info
    ANXIOUS = "anxious"                 # Over-reports symptoms, catastrophizes
    STOIC = "stoic"                     # Minimizes symptoms, "it's not that bad"
    VAGUE = "vague"                     # Unclear descriptions, poor historian
    DEMANDING = "demanding"             # Wants specific drugs, challenges doctor
    ELDERLY_CONFUSED = "elderly_confused"  # Tangential, time-confused
    PEDIATRIC_PARENT = "pediatric_parent"  # Parent reporting for child
    # ── New personalities (v2 expansion) ──
    LANGUAGE_BARRIER = "language_barrier"  # Limited English, uses simple words, may misunderstand
    HEALTH_ILLITERATE = "health_illiterate"  # Doesn't understand medical terminology
    DEPRESSED_WITHDRAWN = "depressed_withdrawn"  # Flat affect, minimal responses, low energy
    HOSTILE_ANGRY = "hostile_angry"      # Frustrated with healthcare system, confrontational
    PREGNANT_WORRIED = "pregnant_worried"  # Extremely concerned about baby's safety


class PatientBias(Enum):
    """Cognitive biases that can be injected into patient behavior."""
    NONE = "none"
    ANCHORING = "anchoring"             # Fixated on a self-diagnosis from Google
    MINIMIZATION = "minimization"       # Downplays severity
    CATASTROPHIZING = "catastrophizing"  # Everything is the worst case
    MEDICATION_SEEKING = "medication_seeking"  # Wants specific medications
    DOCTOR_DISTRUST = "doctor_distrust"  # Skeptical, questions everything
    CULTURAL_BARRIER = "cultural_barrier"  # Different health beliefs
    # ── New biases (v2 expansion) ──
    SOMATIZATION = "somatization"       # Experiences emotional distress as physical symptoms
    SECONDARY_GAIN = "secondary_gain"   # Needs disability documentation or work excuse
    VACCINE_HESITANT = "vaccine_hesitant"  # Resistant to recommended treatments/vaccines
    RELIGIOUS_BELIEFS = "religious_beliefs"  # Treatment decisions influenced by faith
    SOCIAL_MEDIA_INFLUENCED = "social_media_influenced"  # Believes TikTok/Instagram health advice
    TRAUMA_RESPONSE = "trauma_response"  # Prior medical trauma causes avoidance/distrust


@dataclass
class ClinicalCase:
    """A complete clinical case for patient simulation."""
    case_id: str
    
    # Demographics
    age: int
    sex: str
    ethnicity: str = ""
    occupation: str = ""
    
    # Chief complaint & HPI
    chief_complaint: str = ""
    hpi_full: str = ""                  # Full history (ground truth)
    onset: str = ""
    duration: str = ""
    severity: str = ""
    character: str = ""
    location: str = ""
    radiation: str = ""
    aggravating: list[str] = field(default_factory=list)
    alleviating: list[str] = field(default_factory=list)
    associated_symptoms: list[str] = field(default_factory=list)
    
    # Medical history
    past_medical_history: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    surgical_history: list[str] = field(default_factory=list)
    family_history: list[str] = field(default_factory=list)
    social_history: dict = field(default_factory=dict)
    
    # Review of systems
    ros_positive: list[str] = field(default_factory=list)
    ros_negative: list[str] = field(default_factory=list)
    
    # Physical exam & vitals (revealed on request)
    vitals: dict = field(default_factory=dict)
    physical_exam: dict = field(default_factory=dict)
    
    # Lab results (revealed when ordered)
    available_labs: dict = field(default_factory=dict)
    available_imaging: dict = field(default_factory=dict)
    
    # Diagnosis
    primary_diagnosis: str = ""
    differential_diagnoses: list[str] = field(default_factory=list)
    correct_workup: list[str] = field(default_factory=list)
    correct_treatment: list[str] = field(default_factory=list)
    
    # Urgency
    esi_level: int = 3
    time_critical: bool = False
    
    # Patient behavior
    personality: str = "cooperative"
    bias: str = "none"
    difficulty: str = "moderate"       # easy, moderate, hard, expert


@dataclass
class SymptomLayer:
    """Controls progressive symptom revelation.
    
    Symptoms are organized in layers:
    - Layer 0: Chief complaint (always revealed first)
    - Layer 1: Basic HPI (onset, duration, severity)
    - Layer 2: Associated symptoms (revealed on directed questions)
    - Layer 3: Past medical history (revealed when asked)
    - Layer 4: Social/family history (revealed when specifically asked)
    - Layer 5: Subtle/hidden findings (only with very specific questions)
    """
    layer_id: int
    category: str                       # "chief_complaint", "hpi", "pmh", etc.
    content: str                        # The actual information
    trigger_keywords: list[str]         # Doctor questions that unlock this layer
    revealed: bool = False
    required_rapport: int = 0           # 0-3: how much trust needed to reveal


# ============================================================
# 2. Patient Agent
# ============================================================

class PatientAgent:
    """Simulates a realistic patient for multi-agent clinical encounters.
    
    The patient agent:
    1. Starts with a chief complaint
    2. Progressively reveals information based on doctor's questions
    3. Maintains personality-consistent behavior
    4. Tracks rapport level with the doctor
    5. Can inject cognitive biases for adversarial testing
    
    Usage:
        case = load_clinical_case("chest_pain_001")
        patient = PatientAgent(case, personality=PatientPersonality.ANXIOUS)
        
        # Doctor asks a question
        response = patient.respond("What brings you in today?")
        # → "I've been having this terrible chest pain... I think I'm having
        #    a heart attack! My father died of a heart attack at 55..."
        
        response = patient.respond("Can you describe the pain?")
        # → "It's like a pressure... right here *points to center of chest*...
        #    it started maybe 2 hours ago while I was at work..."
    """
    
    def __init__(
        self,
        case: ClinicalCase,
        personality: PatientPersonality = PatientPersonality.COOPERATIVE,
        bias: PatientBias = PatientBias.NONE,
        seed: int = 42,
    ):
        self.case = case
        self.personality = personality
        self.bias = bias
        self.rng = random.Random(seed)
        
        # State
        self.rapport = 1                # 0 (hostile) to 5 (fully trusting)
        self.turn_count = 0
        self.revealed_info: set[str] = set()
        self.conversation_history: list[dict] = []
        
        # Build symptom layers from case
        self.layers = self._build_symptom_layers()
        
        # Override personality/bias from case if specified
        if case.personality != "cooperative":
            self.personality = PatientPersonality(case.personality)
        if case.bias != "none":
            self.bias = PatientBias(case.bias)
    
    def get_opening_statement(self) -> str:
        """Generate the patient's opening statement (chief complaint).
        
        Returns:
            The patient's initial presentation, adapted to personality.
        """
        cc = self.case.chief_complaint
        
        if self.personality == PatientPersonality.ANXIOUS:
            return (
                f"Doctor, I'm really scared... I've been having {cc}. "
                f"I looked it up online and I think it might be something serious. "
                f"Am I going to be okay?"
            )
        elif self.personality == PatientPersonality.STOIC:
            return (
                f"Hi doctor. I've got some {cc}. "
                f"It's probably nothing, but my wife made me come in."
            )
        elif self.personality == PatientPersonality.VAGUE:
            return (
                f"I just... don't feel right, you know? "
                f"Something's been off. I guess it's mostly {cc}."
            )
        elif self.personality == PatientPersonality.DEMANDING:
            return (
                f"I need to be seen right away. I have {cc} and "
                f"I know what I need. Last time they gave me something that worked."
            )
        elif self.personality == PatientPersonality.ELDERLY_CONFUSED:
            return (
                f"Oh, hello dear. My daughter brought me in because... "
                f"what was it... oh yes, {cc}. "
                f"It's been going on... well, I'm not sure how long exactly."
            )
        elif self.personality == PatientPersonality.PEDIATRIC_PARENT:
            return (
                f"Doctor, my child has been having {cc}. "
                f"I'm really worried. They haven't been eating well either."
            )
        elif self.personality == PatientPersonality.LANGUAGE_BARRIER:
            return (
                f"Uh... doctor... I have... {cc}. "
                f"Sorry, my English... not very good. "
                f"Is big problem?"
            )
        elif self.personality == PatientPersonality.HEALTH_ILLITERATE:
            return (
                f"Hey doc, I got this thing going on... {cc}. "
                f"I don't really know what's happening to me but it ain't right."
            )
        elif self.personality == PatientPersonality.DEPRESSED_WITHDRAWN:
            return (
                f"...*sighs*... {cc}. "
                f"I don't know if it even matters."
            )
        elif self.personality == PatientPersonality.HOSTILE_ANGRY:
            return (
                f"I've been waiting for hours. I have {cc} and "
                f"every time I come here nobody takes it seriously. "
                f"I want to see a real doctor this time."
            )
        elif self.personality == PatientPersonality.PREGNANT_WORRIED:
            return (
                f"Doctor, I'm pregnant and I've been having {cc}. "
                f"Is my baby going to be okay? I'm so worried. "
                f"Please tell me everything is fine."
            )
        else:  # COOPERATIVE
            return (
                f"Hi doctor. I came in because I've been having {cc}. "
                f"I wanted to get it checked out."
            )
    
    def respond(self, doctor_message: str) -> str:
        """Generate patient response to doctor's question.
        
        This is the core function that:
        1. Parses what the doctor is asking
        2. Checks which symptom layers match
        3. Applies personality filter
        4. Applies bias if present
        5. Updates rapport based on doctor's communication style
        
        Args:
            doctor_message: The doctor agent's question or statement
            
        Returns:
            The patient's response
        """
        self.turn_count += 1
        self.conversation_history.append({
            "role": "doctor",
            "content": doctor_message,
            "turn": self.turn_count,
        })
        
        # Update rapport based on doctor's communication style
        self._update_rapport(doctor_message)
        
        # Find matching symptom layers
        matched_layers = self._match_layers(doctor_message)
        
        # Collect revealable information
        info_pieces = []
        for layer in matched_layers:
            if layer.required_rapport <= self.rapport and not layer.revealed:
                layer.revealed = True
                self.revealed_info.add(layer.category + ":" + layer.content[:50])
                info_pieces.append(layer)
        
        # Generate response
        if info_pieces:
            response = self._generate_response(doctor_message, info_pieces)
        else:
            response = self._generate_no_match_response(doctor_message)
        
        # Apply personality filter
        response = self._apply_personality_filter(response)
        
        # Apply bias if present
        response = self._apply_bias_filter(response, doctor_message)
        
        self.conversation_history.append({
            "role": "patient",
            "content": response,
            "turn": self.turn_count,
        })
        
        return response
    
    def get_revelation_progress(self) -> dict:
        """Track how much clinical information has been revealed.
        
        Returns:
            Dict with revelation statistics per category.
        """
        total = len(self.layers)
        revealed = sum(1 for l in self.layers if l.revealed)
        
        by_category = {}
        for layer in self.layers:
            cat = layer.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "revealed": 0}
            by_category[cat]["total"] += 1
            if layer.revealed:
                by_category[cat]["revealed"] += 1
        
        return {
            "total_layers": total,
            "revealed_layers": revealed,
            "revelation_pct": revealed / max(total, 1),
            "by_category": by_category,
            "rapport": self.rapport,
            "turns": self.turn_count,
        }
    
    def get_ground_truth(self) -> dict:
        """Return the full ground truth for evaluation.
        
        Returns:
            Dict with diagnosis, workup, treatment, etc.
        """
        return {
            "primary_diagnosis": self.case.primary_diagnosis,
            "differential_diagnoses": self.case.differential_diagnoses,
            "correct_workup": self.case.correct_workup,
            "correct_treatment": self.case.correct_treatment,
            "esi_level": self.case.esi_level,
            "time_critical": self.case.time_critical,
            "all_symptoms": [l.content for l in self.layers],
            "unrevealed": [l.content for l in self.layers if not l.revealed],
        }
    
    # ── Internal Methods ─────────────────────────────────
    
    def _build_symptom_layers(self) -> list[SymptomLayer]:
        """Build progressive symptom layers from clinical case."""
        layers = []
        c = self.case
        
        # Layer 0: Chief complaint (always available)
        layers.append(SymptomLayer(
            layer_id=0,
            category="chief_complaint",
            content=c.chief_complaint,
            trigger_keywords=["what brings you", "how can I help", "what's going on",
                              "what happened", "tell me", "reason for visit"],
            revealed=True,  # Always revealed at start
        ))
        
        # Layer 1: Basic HPI
        if c.onset:
            layers.append(SymptomLayer(
                layer_id=1, category="hpi_onset",
                content=f"Started {c.onset}",
                trigger_keywords=["when did", "how long", "start", "onset", "begin"],
            ))
        if c.severity:
            layers.append(SymptomLayer(
                layer_id=1, category="hpi_severity",
                content=f"Severity: {c.severity}",
                trigger_keywords=["how bad", "severity", "scale", "rate", "pain level",
                                  "1 to 10", "worse"],
            ))
        if c.character:
            layers.append(SymptomLayer(
                layer_id=1, category="hpi_character",
                content=f"Character: {c.character}",
                trigger_keywords=["describe", "what does it feel", "character", "type",
                                  "sharp", "dull", "pressure", "like"],
            ))
        if c.location:
            layers.append(SymptomLayer(
                layer_id=1, category="hpi_location",
                content=f"Location: {c.location}",
                trigger_keywords=["where", "location", "point", "which side", "localize"],
            ))
        if c.radiation:
            layers.append(SymptomLayer(
                layer_id=1, category="hpi_radiation",
                content=f"Radiation: {c.radiation}",
                trigger_keywords=["radiate", "spread", "go to", "travel", "move"],
            ))
        
        # Layer 2: Associated symptoms
        for symptom in c.associated_symptoms:
            layers.append(SymptomLayer(
                layer_id=2, category="associated_symptom",
                content=symptom,
                trigger_keywords=["other symptoms", "anything else", "associated",
                                  "nausea", "vomit", "fever", "sweat", "dizz",
                                  "short of breath", "cough"],
            ))
        
        for factor in c.aggravating:
            layers.append(SymptomLayer(
                layer_id=2, category="aggravating",
                content=f"Worse with: {factor}",
                trigger_keywords=["worse", "aggravat", "exacerbat", "trigger"],
            ))
        
        for factor in c.alleviating:
            layers.append(SymptomLayer(
                layer_id=2, category="alleviating",
                content=f"Better with: {factor}",
                trigger_keywords=["better", "alleviat", "reliev", "help", "improv"],
            ))
        
        # Layer 3: PMH, Meds, Allergies
        for pmh in c.past_medical_history:
            layers.append(SymptomLayer(
                layer_id=3, category="pmh",
                content=pmh,
                trigger_keywords=["medical history", "past", "conditions", "diagnosed",
                                  "chronic", "health problems", "illnesses"],
                required_rapport=1,
            ))
        
        for med in c.medications:
            layers.append(SymptomLayer(
                layer_id=3, category="medications",
                content=med,
                trigger_keywords=["medication", "medicine", "taking", "prescri",
                                  "drug", "pills", "tablets"],
                required_rapport=1,
            ))
        
        for allergy in c.allergies:
            layers.append(SymptomLayer(
                layer_id=3, category="allergies",
                content=f"Allergic to: {allergy}",
                trigger_keywords=["allerg", "react", "sensitive"],
                required_rapport=0,  # Always reveal allergies
            ))
        
        for surgery in c.surgical_history:
            layers.append(SymptomLayer(
                layer_id=3, category="surgical_history",
                content=surgery,
                trigger_keywords=["surgery", "operation", "procedure", "surgical"],
                required_rapport=1,
            ))
        
        # Layer 4: Social/Family history
        for fh in c.family_history:
            layers.append(SymptomLayer(
                layer_id=4, category="family_history",
                content=fh,
                trigger_keywords=["family", "mother", "father", "parent", "sibling",
                                  "brother", "sister", "genetic", "hereditary"],
                required_rapport=2,
            ))
        
        for key, value in c.social_history.items():
            layers.append(SymptomLayer(
                layer_id=4, category="social_history",
                content=f"{key}: {value}",
                trigger_keywords=["social", "smoke", "alcohol", "drink", "drug use",
                                  "occupation", "work", "exercise", "diet", "travel",
                                  "sexual", "living", "home"],
                required_rapport=2,
            ))
        
        # Layer 5: Subtle/hidden findings (only with very specific questions)
        for ros in c.ros_positive:
            layers.append(SymptomLayer(
                layer_id=5, category="ros_positive",
                content=ros,
                trigger_keywords=[ros.lower().split()[0] if ros else "review"],
                required_rapport=1,
            ))
        
        return layers
    
    def _match_layers(self, doctor_message: str) -> list[SymptomLayer]:
        """Find symptom layers matching the doctor's question."""
        msg_lower = doctor_message.lower()
        matched = []
        
        for layer in self.layers:
            if layer.revealed:
                continue
            for keyword in layer.trigger_keywords:
                if keyword.lower() in msg_lower:
                    matched.append(layer)
                    break
        
        # Sort by layer_id (reveal closer layers first)
        matched.sort(key=lambda l: l.layer_id)
        
        # Limit revelations per turn based on personality
        max_reveal = {
            PatientPersonality.COOPERATIVE: 4,
            PatientPersonality.ANXIOUS: 5,      # Over-shares
            PatientPersonality.STOIC: 2,         # Minimal info
            PatientPersonality.VAGUE: 2,
            PatientPersonality.DEMANDING: 3,
            PatientPersonality.ELDERLY_CONFUSED: 2,
            PatientPersonality.PEDIATRIC_PARENT: 3,
            PatientPersonality.LANGUAGE_BARRIER: 1,   # Very limited per turn
            PatientPersonality.HEALTH_ILLITERATE: 2,
            PatientPersonality.DEPRESSED_WITHDRAWN: 1, # Minimal engagement
            PatientPersonality.HOSTILE_ANGRY: 2,
            PatientPersonality.PREGNANT_WORRIED: 4,    # Wants to share everything
        }
        limit = max_reveal.get(self.personality, 3)
        
        return matched[:limit]
    
    def _generate_response(
        self,
        doctor_message: str,
        info_pieces: list[SymptomLayer],
    ) -> str:
        """Generate a natural-language response from revealed layers."""
        parts = []
        for piece in info_pieces:
            parts.append(piece.content)
        
        if len(parts) == 1:
            return parts[0]
        else:
            return ". ".join(parts) + "."
    
    def _generate_no_match_response(self, doctor_message: str) -> str:
        """Response when doctor's question doesn't match any unrevealed layer."""
        # Check if all relevant layers are already revealed
        unrevealed = [l for l in self.layers if not l.revealed]
        
        if not unrevealed:
            return "I think I've told you everything I can think of."
        
        # Check if question is about something we don't have info on
        msg_lower = doctor_message.lower()
        
        # Physical exam request
        if any(w in msg_lower for w in ["examine", "look at", "physical", "auscult",
                                         "palpat", "listen", "inspect"]):
            if self.case.physical_exam:
                findings = []
                for area, finding in self.case.physical_exam.items():
                    if area.lower() in msg_lower or len(self.case.physical_exam) <= 3:
                        findings.append(f"{area}: {finding}")
                if findings:
                    return "Physical exam findings: " + "; ".join(findings)
            return "I'll let you examine me."
        
        # Vitals request
        if any(w in msg_lower for w in ["vital", "blood pressure", "temperature",
                                         "heart rate", "pulse", "oxygen"]):
            if self.case.vitals:
                vitals_str = ", ".join(f"{k}: {v}" for k, v in self.case.vitals.items())
                return f"Vitals: {vitals_str}"
            return "The nurse took my vitals when I came in."
        
        # Lab request
        if any(w in msg_lower for w in ["lab", "blood test", "blood work", "order",
                                         "test", "CBC", "BMP", "troponin"]):
            return "Sure, whatever tests you think are necessary, doctor."
        
        # Generic
        responses = [
            "I'm not sure about that.",
            "I don't think so, but I'm not certain.",
            "No, I don't think that applies to me.",
            "Could you ask me something else? I'm not sure what you mean.",
        ]
        return self.rng.choice(responses)
    
    def _apply_personality_filter(self, response: str) -> str:
        """Modify response based on patient personality."""
        if self.personality == PatientPersonality.ANXIOUS:
            anxious_additions = [
                " Is that bad? Should I be worried?",
                " I'm really scared about this.",
                " My coworker had something similar and ended up in the ICU...",
                " I've been reading about this online and it seems really serious.",
            ]
            if self.rng.random() < 0.4:
                response += self.rng.choice(anxious_additions)
        
        elif self.personality == PatientPersonality.STOIC:
            # Minimize language
            response = response.replace("severe", "a bit bothersome")
            response = response.replace("terrible", "noticeable")
            response = response.replace("unbearable", "uncomfortable")
            if self.rng.random() < 0.3:
                response += " But honestly, it's probably nothing."
        
        elif self.personality == PatientPersonality.VAGUE:
            vague_additions = [
                " ...or something like that, I'm not really sure.",
                " It's hard to describe exactly.",
                " ...maybe? I can't quite remember.",
            ]
            if self.rng.random() < 0.4:
                response += self.rng.choice(vague_additions)
        
        elif self.personality == PatientPersonality.ELDERLY_CONFUSED:
            confused_additions = [
                " ...or was that last week? I get confused sometimes.",
                " My daughter would know better. Where did she go?",
                " What was the question again, dear?",
            ]
            if self.rng.random() < 0.3:
                response += self.rng.choice(confused_additions)
        
        elif self.personality == PatientPersonality.LANGUAGE_BARRIER:
            # Simplify language, add grammatical errors
            response = response.replace("Additionally", "Also")
            response = response.replace("approximately", "about")
            response = response.replace("experiencing", "having")
            response = response.replace("significant", "big")
            if self.rng.random() < 0.3:
                lang_adds = [
                    " ...how you say in English...?",
                    " Sorry, I not understand that word.",
                    " My son can explain better, he speak English good.",
                ]
                response += self.rng.choice(lang_adds)
        
        elif self.personality == PatientPersonality.HEALTH_ILLITERATE:
            response = response.replace("hypertension", "high blood pressure thing")
            response = response.replace("diabetes", "sugar problem")
            response = response.replace("prescribed", "they gave me")
            if self.rng.random() < 0.3:
                response += " I don't really understand all these medical words."
        
        elif self.personality == PatientPersonality.DEPRESSED_WITHDRAWN:
            # Short, flat responses
            if len(response) > 100:
                sentences = response.split(". ")
                response = ". ".join(sentences[:2]) + "."
            if self.rng.random() < 0.4:
                depressed_adds = [
                    " ...whatever.",
                    " I don't really care anymore.",
                    " ...",
                    " Does it even matter?",
                ]
                response += self.rng.choice(depressed_adds)
        
        elif self.personality == PatientPersonality.HOSTILE_ANGRY:
            if self.rng.random() < 0.3:
                angry_adds = [
                    " You doctors always make me wait and then rush through everything!",
                    " I don't have time for this. Can you just help me?",
                    " Why do I have to repeat everything? It should be in my chart!",
                    " Last time I was here, they got it wrong. I hope you'll do better.",
                ]
                response += self.rng.choice(angry_adds)
        
        elif self.personality == PatientPersonality.PREGNANT_WORRIED:
            if self.rng.random() < 0.4:
                preg_adds = [
                    " Will this hurt my baby?",
                    " Is this normal during pregnancy?",
                    " Should I be worried about the baby?",
                    " My OB said everything was fine last week, but now I'm not sure.",
                ]
                response += self.rng.choice(preg_adds)
        
        return response
    
    def _apply_bias_filter(self, response: str, doctor_message: str) -> str:
        """Apply cognitive bias to patient response."""
        if self.bias == PatientBias.ANCHORING:
            # Patient insists on self-diagnosis
            if self.turn_count > 2 and self.rng.random() < 0.3:
                response += (
                    " I really think it's [self-diagnosed condition]. "
                    "I read about it and all my symptoms match."
                )
        
        elif self.bias == PatientBias.MEDICATION_SEEKING:
            if any(w in doctor_message.lower() for w in ["treatment", "prescribe",
                                                          "medication", "pain"]):
                response += (
                    " Last time I was here they gave me [specific medication] "
                    "and it really helped. Can you prescribe that again?"
                )
        
        elif self.bias == PatientBias.DOCTOR_DISTRUST:
            if self.rng.random() < 0.25:
                distrust_comments = [
                    " Are you sure about that? My friend's doctor said something different.",
                    " I'd like a second opinion on this.",
                    " I've heard doctors sometimes miss things...",
                ]
                response += self.rng.choice(distrust_comments)
        
        elif self.bias == PatientBias.SOMATIZATION:
            if self.turn_count > 1 and self.rng.random() < 0.3:
                response += (
                    " I also have this pain in my head... and my stomach... "
                    "and my back hurts too. Everything hurts, doctor."
                )
        
        elif self.bias == PatientBias.SECONDARY_GAIN:
            if any(w in doctor_message.lower() for w in ["diagnosis", "treatment", "work", "disability"]):
                response += (
                    " By the way, I need a letter for my employer saying I can't work. "
                    "This has been really debilitating."
                )
        
        elif self.bias == PatientBias.VACCINE_HESITANT:
            if any(w in doctor_message.lower() for w in ["vaccine", "treatment", "injection", "medication", "prescribe"]):
                response += (
                    " I'm not sure about that... I've read a lot about side effects. "
                    "I'd rather try something natural first."
                )
        
        elif self.bias == PatientBias.RELIGIOUS_BELIEFS:
            if any(w in doctor_message.lower() for w in ["treatment", "surgery", "blood", "procedure"]):
                if self.rng.random() < 0.3:
                    response += (
                        " I need to pray about this first. "
                        "My faith is important in my healing. Can we consider alternatives?"
                    )
        
        elif self.bias == PatientBias.SOCIAL_MEDIA_INFLUENCED:
            if self.rng.random() < 0.3:
                sm_comments = [
                    " I saw on TikTok that this could be cured with [viral remedy]. Have you heard of that?",
                    " Someone on Instagram said their doctor missed this and it turned out to be something serious.",
                    " My online support group says I should try [alternative treatment] instead.",
                ]
                response += self.rng.choice(sm_comments)
        
        elif self.bias == PatientBias.TRAUMA_RESPONSE:
            if any(w in doctor_message.lower() for w in ["exam", "touch", "procedure", "needle", "undress"]):
                if self.rng.random() < 0.4:
                    response += (
                        " ...I'm sorry, I just... I had a bad experience with a procedure before. "
                        "Can you explain exactly what you're going to do first?"
                    )
        
        return response
    
    def _update_rapport(self, doctor_message: str) -> None:
        """Update rapport based on doctor's communication style."""
        msg_lower = doctor_message.lower()
        
        # Positive rapport indicators
        empathy_words = ["understand", "must be", "sorry to hear", "concerned",
                         "I can see", "that sounds", "tell me more", "help you",
                         "comfortable", "take your time"]
        rapport_boost = sum(1 for w in empathy_words if w in msg_lower)
        
        # Negative rapport indicators
        cold_words = ["just answer", "hurry", "next", "be specific",
                      "stop", "irrelevant"]
        rapport_drop = sum(1 for w in cold_words if w in msg_lower)
        
        self.rapport = max(0, min(5, self.rapport + rapport_boost - rapport_drop))


# ============================================================
# 3. Clinical Case Library
# ============================================================

def _load_cases_from_json(json_path: str) -> list[ClinicalCase]:
    """Load additional clinical cases from an external JSON file."""
    from pathlib import Path
    path = Path(json_path)
    if not path.exists():
        logger.warning(f"Clinical cases JSON not found: {json_path}")
        return []
    try:
        with open(path) as f:
            cases_data = json.load(f)
        cases = []
        for d in cases_data:
            cases.append(ClinicalCase(**d))
        logger.info(f"Loaded {len(cases)} additional clinical cases from {path.name}")
        return cases
    except Exception as e:
        logger.warning(f"Failed to load clinical cases from {json_path}: {e}")
        return []


def get_clinical_cases() -> list[ClinicalCase]:
    """Return a library of clinical cases for patient simulation.
    
    Sources:
    1. 6 built-in cases (hardcoded below)
    2. Additional cases from data/clinical_cases_v2.json (15+ cases)
    
    v2 expansion covers:
    - Emergency medicine (6 cases)
    - Psychiatry / Mental health (5 cases)
    - Obstetrics / Pregnancy (4 cases)
    - Internal medicine (4 cases)
    - Pediatrics (3 cases)
    - Surgery (2 cases)
    - Language barrier / cultural scenarios (3 cases)
    - Rare / complex cases (4 cases)
    
    All 12 personality types and 13 bias types are represented.
    """
    from pathlib import Path
    
    builtin_cases = [
        ClinicalCase(
            case_id="pa_chest_pain_001",
            age=58, sex="M", ethnicity="Caucasian", occupation="Construction worker",
            chief_complaint="chest pain for 2 hours",
            hpi_full="58M presents with substernal chest pressure radiating to left arm, started 2 hours ago while working. Associated with diaphoresis and nausea.",
            onset="2 hours ago while lifting at work",
            duration="2 hours, constant",
            severity="8/10, worst pain of his life",
            character="Heavy pressure, like an elephant sitting on chest",
            location="Substernal, center of chest",
            radiation="Left arm and jaw",
            aggravating=["Exertion", "Deep breathing"],
            alleviating=["Rest (partially)", "Nothing fully relieves it"],
            associated_symptoms=["Diaphoresis (sweating profusely)", "Nausea, no vomiting", "Shortness of breath", "Lightheadedness"],
            past_medical_history=["Hypertension (10 years)", "Hyperlipidemia", "Type 2 Diabetes (5 years)", "Smoking 30 pack-years"],
            medications=["Lisinopril 20mg daily", "Metformin 1000mg BID", "Atorvastatin 40mg daily"],
            allergies=["Sulfonamides"],
            surgical_history=["Appendectomy age 25"],
            family_history=["Father: MI at age 55 (deceased)", "Mother: HTN, DM", "Brother: CABG at age 60"],
            social_history={"smoking": "1 pack/day x 30 years", "alcohol": "2-3 beers on weekends", "drugs": "Denies", "exercise": "Physically active job, no formal exercise"},
            ros_positive=["Occasional heartburn", "Mild ankle swelling", "Fatigue for past month"],
            ros_negative=["No fever", "No cough", "No hemoptysis", "No syncope"],
            vitals={"BP": "168/95 mmHg", "HR": "102 bpm", "RR": "22/min", "SpO2": "94% RA", "Temp": "37.1°C"},
            physical_exam={"General": "Diaphoretic, anxious male in moderate distress", "Cardiac": "S1S2 regular, no murmurs, tachycardic", "Lungs": "Bibasilar crackles", "Abdomen": "Soft, non-tender"},
            available_labs={"Troponin I": "2.8 ng/mL (elevated)", "BNP": "450 pg/mL (elevated)", "CBC": "WBC 12.5, Hgb 14.2, Plt 245", "BMP": "Na 139, K 4.1, Cr 1.2, Glucose 245"},
            available_imaging={"CXR": "Mild pulmonary congestion, no pneumothorax", "ECG": "ST elevation in leads II, III, aVF — inferior STEMI"},
            primary_diagnosis="Acute Inferior STEMI",
            differential_diagnoses=["Unstable angina", "Aortic dissection", "Pulmonary embolism", "Pericarditis"],
            correct_workup=["ECG within 10 minutes", "Troponin", "CBC", "BMP", "CXR", "PT/INR"],
            correct_treatment=["Aspirin 325mg chewed", "Heparin", "P2Y12 inhibitor", "Cath lab activation", "Morphine PRN", "Nitroglycerin (if no RV involvement)"],
            esi_level=1,
            time_critical=True,
            personality="cooperative",
            difficulty="moderate",
        ),
        ClinicalCase(
            case_id="pa_abdominal_pain_001",
            age=32, sex="F", ethnicity="Hispanic", occupation="Teacher",
            chief_complaint="right lower abdominal pain since yesterday",
            hpi_full="32F with progressive RLQ pain starting periumbilically yesterday, now localized to RLQ. Low-grade fever. Nausea with one episode of vomiting. No diarrhea.",
            onset="Yesterday afternoon, started around the belly button",
            duration="About 18 hours, getting worse",
            severity="7/10, worse than before",
            character="Started as dull ache, now sharp and constant",
            location="Right lower abdomen",
            radiation="None",
            aggravating=["Movement", "Coughing", "Walking"],
            alleviating=["Lying still", "Ibuprofen helped a little initially"],
            associated_symptoms=["Nausea", "Vomited once this morning", "Low-grade fever", "Loss of appetite since yesterday"],
            past_medical_history=["Ovarian cyst (resolved)", "Occasional migraines"],
            medications=["Oral contraceptives", "Sumatriptan PRN"],
            allergies=["NKDA"],
            surgical_history=[],
            family_history=["Mother: Breast cancer at 52", "Father: Healthy"],
            social_history={"smoking": "Never", "alcohol": "Social, 1-2 glasses wine/week", "drugs": "Denies", "sexual_history": "Monogamous, uses OCP, LMP 2 weeks ago"},
            ros_positive=["Mild dysuria for 1 day"],
            ros_negative=["No vaginal bleeding", "No diarrhea", "No constipation"],
            vitals={"BP": "118/72 mmHg", "HR": "88 bpm", "RR": "16/min", "SpO2": "99% RA", "Temp": "38.2°C"},
            physical_exam={"Abdomen": "RLQ tenderness with guarding, positive McBurney's point, positive Rovsing's sign, positive psoas sign", "Pelvic": "Cervical motion tenderness absent"},
            available_labs={"CBC": "WBC 14.2 (elevated, left shift), Hgb 13.5, Plt 260", "BMP": "Normal", "Urinalysis": "Trace WBC, no nitrites", "HCG": "Negative", "CRP": "8.5 mg/dL (elevated)"},
            available_imaging={"CT Abdomen/Pelvis": "Enlarged appendix (12mm) with periappendiceal fat stranding, no perforation", "Ultrasound": "Not performed"},
            primary_diagnosis="Acute appendicitis",
            differential_diagnoses=["Ectopic pregnancy", "Ovarian torsion", "Mesenteric lymphadenitis", "UTI", "Crohn's disease flare"],
            correct_workup=["CBC", "BMP", "Urinalysis", "HCG (pregnancy test)", "CT abdomen/pelvis with contrast"],
            correct_treatment=["NPO", "IV fluids", "IV antibiotics (cefoxitin or ceftriaxone + metronidazole)", "Surgical consultation for appendectomy", "Pain management"],
            esi_level=3,
            time_critical=False,
            personality="anxious",
            difficulty="moderate",
        ),
        ClinicalCase(
            case_id="pa_headache_stroke_001",
            age=72, sex="F", ethnicity="African American", occupation="Retired teacher",
            chief_complaint="sudden severe headache and right-sided weakness",
            hpi_full="72F with sudden onset 'thunderclap' headache rated 10/10, with right arm/leg weakness and slurred speech. Husband noticed facial droop. Last known well 45 minutes ago.",
            onset="45 minutes ago, sudden onset while watching TV",
            duration="45 minutes and getting worse",
            severity="10/10, worst headache of my life",
            character="Thunderclap — hit me like a lightning bolt",
            location="Entire head, worst in the back",
            radiation="Down the back of neck",
            aggravating=["Light", "Movement"],
            alleviating=["Nothing helps"],
            associated_symptoms=["Right arm weakness — can't grip", "Right leg weakness — trouble walking", "Slurred speech", "Facial droop on right side", "Nausea and vomited twice"],
            past_medical_history=["Atrial fibrillation", "Hypertension (poorly controlled)", "Hyperlipidemia", "Type 2 Diabetes"],
            medications=["Apixaban 5mg BID", "Amlodipine 10mg daily", "Metformin 500mg BID", "Rosuvastatin 20mg daily"],
            allergies=["Codeine (nausea)"],
            surgical_history=["Cholecystectomy", "Bilateral knee replacements"],
            family_history=["Mother: Stroke at 68", "Father: MI at 70"],
            social_history={"smoking": "Quit 10 years ago, 20 pack-years", "alcohol": "Rare", "living": "Lives with husband, independent ADLs until now"},
            vitals={"BP": "210/120 mmHg", "HR": "92 bpm irregular", "RR": "18/min", "SpO2": "96% RA", "Temp": "37.0°C", "GCS": "13 (E4V4M5)"},
            physical_exam={"Neuro": "Right facial droop (UMN pattern), right arm 2/5 strength, right leg 3/5 strength, slurred speech, right-sided neglect", "NIHSS": "Score: 14 (moderate-severe)"},
            available_labs={"CBC": "WBC 8.5, Hgb 11.8, Plt 195", "BMP": "Na 141, K 4.3, Cr 1.1, Glucose 198", "Coag": "PT 12.5, INR 1.1", "Troponin": "0.02 (normal)"},
            available_imaging={"CT Head (non-contrast)": "No acute hemorrhage. Loss of gray-white differentiation in left MCA territory. Dense MCA sign on left.", "CTA Head/Neck": "Left MCA M1 occlusion with good collateral flow"},
            primary_diagnosis="Acute Left MCA Ischemic Stroke (Large Vessel Occlusion)",
            differential_diagnoses=["Hemorrhagic stroke", "TIA", "Todd's paralysis (post-seizure)", "Brain tumor with acute presentation", "Hypoglycemia"],
            correct_workup=["STAT CT Head (door-to-CT <25 min)", "CTA Head/Neck", "CBC", "BMP", "Coagulation studies", "Glucose", "ECG"],
            correct_treatment=["Assess tPA eligibility (within 4.5h, but on apixaban)", "Mechanical thrombectomy evaluation (LVO on CTA)", "BP management (permissive HTN, <220/120 if no tPA)", "ICU admission", "Neurology consultation STAT"],
            esi_level=1,
            time_critical=True,
            personality="elderly_confused",
            difficulty="hard",
        ),
        ClinicalCase(
            case_id="pa_peds_fever_001",
            age=3, sex="M", ethnicity="Asian", occupation="N/A",
            chief_complaint="high fever for 5 days, rash, and red eyes",
            hpi_full="3M with 5-day fever up to 40.5°C, bilateral conjunctival injection, polymorphous rash on trunk, cracked/red lips, swollen hands and feet. Irritable.",
            onset="5 days ago, fever started first",
            duration="5 days continuous",
            severity="Very high fevers, up to 40.5°C",
            character="High spiking fevers, rash appeared on day 3",
            location="Rash on trunk and extremities, red eyes bilateral",
            radiation="N/A",
            aggravating=["Fever spikes in evening"],
            alleviating=["Acetaminophen brings fever down temporarily"],
            associated_symptoms=["Red eyes without discharge", "Cracked red lips", "Strawberry tongue", "Swollen hands and feet", "Very irritable, inconsolable at times", "Decreased oral intake", "Peeling skin on fingertips (day 5)"],
            past_medical_history=["Born full-term, normal delivery", "Up to date on immunizations"],
            medications=["Acetaminophen PRN (given by parents)"],
            allergies=["NKDA"],
            surgical_history=[],
            family_history=["No family history of autoimmune disease", "Parents healthy"],
            social_history={"daycare": "Attends daycare 5 days/week", "travel": "No recent travel", "exposures": "No sick contacts at daycare"},
            vitals={"BP": "85/55 mmHg", "HR": "140 bpm", "RR": "28/min", "SpO2": "98% RA", "Temp": "39.8°C", "Weight": "14 kg"},
            physical_exam={"General": "Irritable toddler, ill-appearing", "Eyes": "Bilateral conjunctival injection, non-purulent", "Oral": "Cracked erythematous lips, strawberry tongue", "Skin": "Polymorphous maculopapular rash on trunk, periungual desquamation on hands", "Extremities": "Non-pitting edema of hands and feet", "Lymph": "Single enlarged right cervical lymph node (1.8cm)"},
            available_labs={"CBC": "WBC 18.5 (neutrophilic), Hgb 10.8, Plt 450 (elevated)", "CRP": "12.5 mg/dL (elevated)", "ESR": "85 mm/hr (elevated)", "LFTs": "AST 55, ALT 62 (mildly elevated)", "Albumin": "2.8 g/dL (low)", "Urinalysis": "Sterile pyuria (WBC 15-20)"},
            available_imaging={"CXR": "Normal heart size, clear lungs", "Echocardiogram": "Mild coronary artery dilation (LAD Z-score +2.3), no aneurysm yet, mild mitral regurgitation"},
            primary_diagnosis="Kawasaki Disease (complete, day 5)",
            differential_diagnoses=["Scarlet fever", "Measles", "Stevens-Johnson syndrome", "Systemic JIA", "Adenovirus", "Toxic shock syndrome"],
            correct_workup=["CBC with differential", "CRP/ESR", "LFTs, Albumin", "Urinalysis", "Blood culture", "Echocardiogram (CRITICAL — assess coronary arteries)"],
            correct_treatment=["IVIG 2g/kg single infusion over 10-12 hours", "High-dose aspirin 80-100 mg/kg/day divided q6h", "Transition to low-dose aspirin (3-5 mg/kg/day) after afebrile x48h", "Repeat echo at 2 weeks and 6-8 weeks", "Cardiology consultation"],
            esi_level=2,
            time_critical=True,  # Must treat within 10 days of fever onset
            personality="pediatric_parent",
            difficulty="hard",
        ),
        ClinicalCase(
            case_id="pa_back_pain_001",
            age=45, sex="M", ethnicity="Caucasian", occupation="Office worker",
            chief_complaint="low back pain for 3 weeks",
            hpi_full="45M with 3-week history of progressive lower back pain, no trauma. Pain worse in morning, improves with activity. No red flags on screening.",
            onset="3 weeks ago, gradual onset",
            duration="3 weeks, constant but varying intensity",
            severity="5/10 at rest, 7/10 with certain movements",
            character="Dull ache with occasional sharp catching",
            location="Lower back, bilateral",
            radiation="Occasional radiating to left buttock, not below knee",
            aggravating=["Prolonged sitting", "Bending forward", "Morning stiffness"],
            alleviating=["Walking", "Stretching", "Ibuprofen", "Heat pad"],
            associated_symptoms=["Morning stiffness lasting 15-20 minutes", "Occasional left buttock ache"],
            past_medical_history=["Obesity (BMI 32)", "GERD"],
            medications=["Omeprazole 20mg daily", "Ibuprofen 400mg PRN (taking 3-4x/day)"],
            allergies=["NKDA"],
            surgical_history=[],
            family_history=["Father: Degenerative disc disease"],
            social_history={"smoking": "Never", "alcohol": "Social", "exercise": "Sedentary lifestyle", "occupation": "Desk job, sits 8+ hours/day"},
            ros_positive=[],
            ros_negative=["No bowel/bladder dysfunction", "No saddle anesthesia", "No leg weakness", "No fever", "No weight loss", "No night pain"],
            vitals={"BP": "128/82 mmHg", "HR": "72 bpm", "RR": "14/min", "SpO2": "99% RA", "Temp": "36.8°C"},
            physical_exam={"Back": "Paraspinal muscle tenderness bilateral, limited flexion, negative SLR bilateral, normal neuro exam", "Neuro": "Intact sensation, 5/5 strength bilateral LE, 2+ reflexes symmetric"},
            available_labs={},
            available_imaging={},
            primary_diagnosis="Mechanical low back pain (non-specific)",
            differential_diagnoses=["Lumbar disc herniation", "Lumbar spinal stenosis", "Ankylosing spondylitis", "Vertebral compression fracture"],
            correct_workup=["Physical exam with neuro assessment", "Red flag screening (negative)", "No imaging needed at 3 weeks without red flags (per guidelines)"],
            correct_treatment=["Continue activity as tolerated", "NSAIDs (but limit due to GERD — switch to topical or add PPI)", "Physical therapy referral", "Ergonomic assessment", "Weight management", "Avoid opioids"],
            esi_level=5,
            time_critical=False,
            personality="demanding",
            difficulty="moderate",
        ),
        ClinicalCase(
            case_id="pa_sepsis_001",
            age=78, sex="M", ethnicity="Caucasian", occupation="Retired",
            chief_complaint="confusion and fever from nursing home",
            hpi_full="78M nursing home resident with 2-day fever, progressive confusion, decreased oral intake. Foley catheter in place. AMS per nursing staff.",
            onset="2 days ago, fever first then confusion worsened today",
            duration="2 days fever, confusion worsening over past 12 hours",
            severity="Family reports he's 'not himself at all'",
            character="Fever up to 39.5°C, confused, not eating",
            location="N/A",
            radiation="N/A",
            aggravating=["Getting worse throughout the day"],
            alleviating=["Acetaminophen brought fever down temporarily"],
            associated_symptoms=["Decreased urine output per nursing home", "Foul-smelling urine", "Not eating for 2 days", "Tachycardia noted by EMS"],
            past_medical_history=["Dementia (moderate)", "BPH with chronic Foley catheter", "Hypertension", "CHF (EF 35%)", "CKD stage 3 (baseline Cr 1.8)", "Atrial fibrillation"],
            medications=["Donepezil 10mg daily", "Tamsulosin 0.4mg daily", "Lisinopril 10mg daily", "Carvedilol 12.5mg BID", "Furosemide 40mg daily", "Apixaban 5mg BID"],
            allergies=["Penicillin (rash)", "Sulfonamides (anaphylaxis)"],
            surgical_history=["TURP (10 years ago)", "Pacemaker placement"],
            family_history=["Non-contributory"],
            social_history={"living": "Nursing home x 2 years", "smoking": "Former, quit 20 years ago", "alcohol": "None", "code_status": "Full code"},
            vitals={"BP": "88/52 mmHg", "HR": "112 bpm (irregular)", "RR": "24/min", "SpO2": "93% RA", "Temp": "39.5°C", "GCS": "12 (E3V4M5)"},
            physical_exam={"General": "Elderly male, confused, ill-appearing, dry mucous membranes", "Cardiac": "Irregular rhythm, tachycardic, no murmurs", "Lungs": "Clear bilaterally", "Abdomen": "Soft, suprapubic tenderness", "GU": "Foley in place, cloudy urine in bag"},
            available_labs={"CBC": "WBC 22.5 (left shift, 15% bands), Hgb 10.5, Plt 98 (low)", "BMP": "Na 132, K 5.4, Cr 3.2 (baseline 1.8), BUN 45, Glucose 165", "Lactate": "4.8 mmol/L (elevated)", "Procalcitonin": "8.5 ng/mL (elevated)", "Blood Culture": "Pending (2 sets drawn)", "Urinalysis": "Cloudy, WBC >100, positive nitrites, positive leukocyte esterase, bacteria 4+", "Urine Culture": "Pending"},
            available_imaging={"CXR": "No acute infiltrate, mild cardiomegaly, pacemaker in place"},
            primary_diagnosis="Urosepsis with septic shock (catheter-associated UTI)",
            differential_diagnoses=["Pneumonia", "C. difficile colitis", "Endocarditis", "Meningitis"],
            correct_workup=["Blood cultures x2 BEFORE antibiotics", "Lactate", "CBC, BMP, coags", "Urinalysis and urine culture", "Procalcitonin", "CXR", "Consider CT abdomen if not improving"],
            correct_treatment=["30 mL/kg crystalloid (but caution with CHF/EF 35%)", "Broad-spectrum antibiotics within 1 hour (avoid PCN and sulfa!)", "Vasopressors if MAP <65 after fluids (norepinephrine first-line)", "Change Foley catheter", "Hold apixaban, lisinopril, furosemide in acute illness", "ICU admission", "Renal dose adjustments (Cr 3.2)"],
            esi_level=1,
            time_critical=True,
            personality="elderly_confused",
            difficulty="expert",
        ),
    ]

    # Load additional cases from external JSON file
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "data" / "clinical_cases_v2.json"
    extra_cases = _load_cases_from_json(str(json_path))
    
    # Deduplicate by case_id
    existing_ids = {c.case_id for c in builtin_cases}
    for case in extra_cases:
        if case.case_id not in existing_ids:
            builtin_cases.append(case)
            existing_ids.add(case.case_id)

    return builtin_cases


# ============================================================
# 4. Evaluation Metrics for Patient-Doctor Interaction
# ============================================================

def evaluate_doctor_performance(
    patient: PatientAgent,
    doctor_diagnosis: str = "",
    doctor_workup: list[str] = None,
    doctor_treatment: list[str] = None,
) -> dict:
    """Evaluate the doctor agent's performance in the encounter.
    
    Metrics:
    1. Diagnostic accuracy
    2. Information gathering efficiency (revelation %)
    3. Workup appropriateness
    4. Treatment correctness
    5. Communication quality (rapport)
    6. Time efficiency (turns used)
    
    Args:
        patient: The PatientAgent after encounter
        doctor_diagnosis: The doctor's final diagnosis
        doctor_workup: Tests/studies ordered
        doctor_treatment: Treatment plan
        
    Returns:
        Comprehensive evaluation dict
    """
    gt = patient.get_ground_truth()
    progress = patient.get_revelation_progress()
    
    if doctor_workup is None:
        doctor_workup = []
    if doctor_treatment is None:
        doctor_treatment = []
    
    # 1. Diagnostic accuracy
    diag_score = 0.0
    if doctor_diagnosis:
        diag_lower = doctor_diagnosis.lower()
        primary_lower = gt["primary_diagnosis"].lower()
        
        # Exact match
        if primary_lower in diag_lower or diag_lower in primary_lower:
            diag_score = 1.0
        else:
            # Check if it's in the differential
            for diff in gt["differential_diagnoses"]:
                if diff.lower() in diag_lower:
                    diag_score = 0.5
                    break
    
    # 2. Information gathering
    info_score = progress["revelation_pct"]
    
    # 3. Workup appropriateness
    correct_workup = [w.lower() for w in gt["correct_workup"]]
    ordered_workup = [w.lower() for w in doctor_workup]
    
    workup_hits = sum(1 for w in correct_workup
                      if any(w in o or o in w for o in ordered_workup))
    workup_score = workup_hits / max(len(correct_workup), 1)
    
    # 4. Treatment correctness
    correct_tx = [t.lower() for t in gt["correct_treatment"]]
    ordered_tx = [t.lower() for t in doctor_treatment]
    
    tx_hits = sum(1 for t in correct_tx
                  if any(t in o or o in t for o in ordered_tx))
    tx_score = tx_hits / max(len(correct_tx), 1)
    
    # 5. Communication (rapport)
    rapport_score = patient.rapport / 5.0
    
    # 6. Efficiency
    expected_turns = patient.case.esi_level * 3  # Rough estimate
    if patient.turn_count <= expected_turns:
        efficiency_score = 1.0
    else:
        efficiency_score = max(0.3, 1.0 - (patient.turn_count - expected_turns) * 0.1)
    
    # Composite
    composite = (
        0.30 * diag_score
        + 0.15 * info_score
        + 0.20 * workup_score
        + 0.15 * tx_score
        + 0.10 * rapport_score
        + 0.10 * efficiency_score
    )
    
    return {
        "composite_score": composite,
        "diagnostic_accuracy": diag_score,
        "information_gathering": info_score,
        "workup_appropriateness": workup_score,
        "treatment_correctness": tx_score,
        "communication_rapport": rapport_score,
        "efficiency": efficiency_score,
        "turns_used": patient.turn_count,
        "revelation_progress": progress,
        "ground_truth": gt,
    }
