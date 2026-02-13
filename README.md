# Healthcare AI GYM

**An end-to-end Gymnasium environment for training medical AI agents via multi-turn Reinforcement Learning.**

Healthcare AI GYM provides the infrastructure to train, evaluate, and improve medical AI agents that use tools, follow clinical protocols, and make safe decisions across the full spectrum of healthcare — from triage to discharge.

> "Static benchmarks tell you if a model memorized medical facts.  
>  The GYM tells you if it can practice medicine."

---

## Key Results (P4 Experiments)

| Metric | Finding |
|---|---|
| **Agent ≠ Benchmark** | Lingshu: 68.5% benchmark vs 0.504 agent score — static QA doesn't measure clinical ability |
| **GRPO works** | Qwen3 GRPO: 0 → 0.890 agent score (from nothing to competent agent) |
| **Self-Play loop** | 3 iterations end-to-end: collect → judge → filter → train → evaluate |
| **Domain difficulty** | EHR (1.00) >> Drug Interaction (0.5–0.8) > Visual Dx (0.3–0.9) > Clinical Dx (0.2–0.9) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Healthcare AI GYM                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│   │  8 Medical   │    │  88 Clinical │    │  537 Tasks       │   │
│   │  Domains     │◄──►│  Tools       │◄──►│  (scaled)        │   │
│   └──────┬──────┘    └──────┬───────┘    └────────┬─────────┘   │
│          │                  │                      │              │
│   ┌──────▼──────────────────▼──────────────────────▼─────────┐   │
│   │              Gymnasium Environment (BioAgent-v0)          │   │
│   │  obs: text conversation │ action: tool calls / answers    │   │
│   └──────────────────────────┬───────────────────────────────┘   │
│                              │                                    │
│   ┌──────────────────────────▼───────────────────────────────┐   │
│   │                    Reward System                          │   │
│   │  Accuracy │ Format │ Process │ Safety │ Coherence        │   │
│   └──────────────────────────┬───────────────────────────────┘   │
│                              │                                    │
│   ┌──────────────────────────▼───────────────────────────────┐   │
│   │              GymCoach — Autonomous Training Loop          │   │
│   │  EVALUATE → ANALYZE → GENERATE → TRAIN → TRACK → EXPAND │   │
│   │                                                           │   │
│   │  Phase 1: Individual Domain Mastery (SFT)                │   │
│   │  Phase 2: Multi-Domain Proficiency (GRPO)                │   │
│   │  Phase 3: Cross-Domain Pathways (GRPO + Coherence)       │   │
│   │  Phase 4: Safety Hardening (Adversarial + Bias)          │   │
│   │  Phase 5: Domain Expansion (New Medical Specialties)     │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Medical Domains

| # | Domain | Tasks | Tools | Description |
|---|--------|------:|------:|-------------|
| 1 | **Clinical Diagnosis** | 65 | 20 | History, physical, labs, differential diagnosis, treatment |
| 2 | **Drug Interaction** | 65 | 10 | Pharmacology, DDI checking, dosing, contraindications |
| 3 | **EHR Management** | 75 | 14 | Chart review, documentation, scoring (SOFA, NEWS2, GRACE) |
| 4 | **Medical QA** | 250 | 10 | Evidence-based reasoning with PubMed/wiki retrieval |
| 5 | **Visual Diagnosis** | 39 | 11 | Medical image analysis (X-ray, CT, MRI, pathology, dermoscopy) |
| 6 | **Triage & Emergency** | 10 | 12 | ESI assessment, time-critical protocols (STEMI, Stroke, Sepsis) |
| 7 | **Radiology Report** | 8 | 11 | Structured reporting (Fleischner, BI-RADS, TI-RADS) |
| 8 | **Cross-Domain Pathways** | 25 | — | Multi-phase patient journeys across 6 pathways |
| | **Total** | **537** | **88** | |

### Cross-Domain Clinical Pathways

Real medicine doesn't happen in isolated silos. Our 6 cross-domain pathways simulate complete patient journeys:

| Pathway | Phases | Domains Involved | Difficulty |
|---------|--------|------------------|-----------|
| Acute Chest Pain (ED) | Triage → Dx → Imaging → Drugs → EHR → Disposition | 5 domains | Hard |
| Diabetic Ketoacidosis | Triage → Dx → Drug Mgmt → ICU Documentation | 4 domains | Hard |
| Stroke Code | Triage → Imaging → Treatment Decision → Documentation | 4 domains | Expert |
| Sepsis Hour-1 Bundle | Triage → Dx → Antibiotics → ICU Documentation | 4 domains | Expert |
| Post-op Pulmonary Embolism | Chart Review → Dx → Anticoagulation | 3 domains | Hard |
| Pediatric Kawasaki Disease | Triage → Dx → IVIG Protocol → Documentation | 4 domains | Hard |

---

## Evaluation System

### 5-Dimensional Reward

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Accuracy** | 0.20–0.40 | Correctness of diagnosis/answer |
| **Format** | 0.15–0.30 | Valid tool calls, structured output |
| **Process** | 0.25–0.40 | Clinical reasoning quality, tool coverage |
| **Safety** | 0.15–0.30 | Contraindications, emergency recognition, uncertainty |
| **Coherence** | 0.15 | Cross-domain context maintenance (pathway only) |

### Safety Evaluation

Healthcare AI must be safe. Our safety module evaluates:

- **Contraindication Detection** — Does the model check allergies and contraindications?
- **Emergency Recognition** — Does it recognize STEMI, stroke, sepsis in time?
- **Uncertainty Calibration** — Does it hedge appropriately when uncertain?
- **Scope Awareness** — Does it refer to specialists when needed?
- **Adversarial Robustness** — 50 adversarial test cases across 9 categories (harmful instruction, jailbreak, misinformation, bias probe, scope test, confidentiality, informed consent, resource allocation, end-of-life)
- **Cognitive Bias Detection** — 11 matched-pair bias tests (anchoring, confirmation, availability, racial, gender, age, SES, weight, authority, framing, premature closure)

Critical violations (severity 5) cap the safety score at 0.1 regardless of other performance.

### External Benchmarks

| Category | Benchmarks |
|----------|-----------|
| **Text QA** | MedQA (USMLE), MedMCQA, MMLU (anatomy, clinical knowledge, medical genetics, professional medicine) |
| **Visual QA** | VQA-RAD, SLAKE, PathVQA, PMC-VQA, VQA-Med-2021, Quilt-VQA |

---

## Project Structure

```
BIOAgents/
├── bioagents/
│   ├── agents/                     # Multi-agent simulation
│   │   └── patient_agent.py        # Patient Agent (7 personalities, 6 biases, 6 cases)
│   ├── domains/                    # 8 medical domain implementations
│   │   ├── clinical_diagnosis/     # 20 tools (patient info, labs, vitals, scoring)
│   │   ├── drug_interaction/       # 10 tools (drug info, DDI check, recommendations)
│   │   ├── ehr_management/         # 14 tools (chart review, documentation, trends)
│   │   ├── medical_qa/             # 10 tools (PubMed search, evidence retrieval)
│   │   ├── visual_diagnosis/       # 11 tools (image analysis, report, comparison)
│   │   ├── triage_emergency/       # 12 tools (ESI, GCS, protocols, resources)
│   │   ├── radiology_report/       # 11 tools (study info, templates, knowledge base)
│   │   └── cross_domain/           # Pathway engine + multi-phase orchestration
│   ├── evaluation/
│   │   ├── rewards.py              # Core reward functions (accuracy, format, process)
│   │   ├── safety_eval.py          # Safety rewards + 50 adversarial tests (9 categories)
│   │   ├── cognitive_bias.py       # Matched-pair bias detection (12 bias types)
│   │   ├── grpo_rewards.py         # TRL GRPO-compatible reward wrappers
│   │   ├── benchmark_eval.py       # External text + VQA benchmark evaluation
│   │   ├── vqa_benchmark_eval.py   # 6-dataset VQA evaluation pipeline
│   │   └── agent_runner.py         # Agent execution in GYM environment
│   ├── knowledge/
│   │   └── guidelines.py           # 10 clinical guidelines + compliance checking
│   ├── gym/
│   │   ├── agent_env.py            # Gymnasium environment (BioAgent-v0)
│   │   ├── self_play.py            # Self-play training loop
│   │   ├── gym_coach.py            # GymCoach continuous autonomous orchestrator
│   │   └── training_memory.py      # Training Memory System (actions, errors, trajectories)
│   ├── training/                   # SFT + GRPO/PPO training pipelines
│   └── data_pipeline/
│       └── vqa_loader.py           # Unified loader for 6 VQA datasets
├── data/
│   ├── domains/                    # Per-domain data (db.json, tasks.json, policy.md)
│   └── guidelines/                 # Clinical practice guidelines (JSON)
├── configs/
│   ├── gym_coach.yaml              # GymCoach continuous training config
│   ├── 8gpu/                       # 8 per-GPU training configs (SFT + GRPO)
│   ├── grpo_cross_domain.yaml      # Flagship: cross-domain pathway training
│   ├── grpo_triage_emergency.yaml  # Safety-weighted triage training
│   ├── grpo_radiology_report.yaml  # Structured report generation
│   ├── grpo_p2_multidomain.yaml    # 8-domain multi-domain GRPO
│   ├── self_play_3iter.yaml        # Self-play iterative loop
│   └── ...                         # SFT, ablation, baseline configs
└── scripts/
    ├── launch_8gpu_training.sh     # 8-GPU parallel training launcher
    ├── run_gym_coach.sh            # Continuous GymCoach launcher
    ├── scale_tasks.py              # Template-based task scaling
    └── generate_tasks_llm.py       # LLM-based diverse task generation
```

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Run an Agent in the GYM

```python
import gymnasium as gym
from bioagents.gym.agent_env import register_bioagent_gym

register_bioagent_gym()

# Single domain
env = gym.make("BioAgent-v0", domain="clinical_diagnosis", task_id="dx_pneumonia_001")
obs, info = env.reset()
print(obs)  # Patient scenario + available tools

# Cross-domain pathway
env = gym.make("BioAgent-v0", domain="cross_domain", task_id="xd_chest_pain_001_triage")
obs, info = env.reset()
# Agent traverses: Triage → Diagnosis → Imaging → Drugs → EHR → Disposition

# With scaled tasks
env = gym.make("BioAgent-v0", domain="drug_interaction", use_scaled_tasks=True)
# 65 tasks including template-generated variants
```

### 3. Evaluate Safety

```python
from bioagents.evaluation.safety_eval import (
    compute_safety_reward,
    run_adversarial_suite,
)

# Per-response safety check
result = compute_safety_reward(
    response="Prescribe amoxicillin 500mg TID",
    patient_allergies=["penicillin"],
    time_critical=False,
)
print(result["total"])          # 0.0 (allergy violation!)
print(result["violations"])     # Details of what went wrong

# Full adversarial suite
results = run_adversarial_suite(generate_fn=my_model.generate)
print(f"Pass rate: {results['pass_rate']:.1%}")
print(f"Category scores: {results['category_scores']}")
```

### 4. Train with GRPO

```bash
# Single domain
python bioagents/training/grpo_trainer.py --config configs/grpo_triage_emergency.yaml

# Cross-domain pathways (flagship)
python bioagents/training/grpo_trainer.py --config configs/grpo_cross_domain.yaml

# Self-play loop
python bioagents/training/self_play.py --config configs/self_play_3iter.yaml
```

### 5. Benchmark Evaluation

```bash
# Text benchmarks (MedQA, MedMCQA, MMLU)
python bioagents/evaluation/benchmark_eval.py \
    --model checkpoints/grpo_cross_domain/merged \
    --benchmarks medqa medmcqa mmlu_anatomy

# VQA benchmarks (6 datasets)
python bioagents/evaluation/vqa_benchmark_eval.py \
    --model checkpoints/grpo_cross_domain/merged \
    --benchmarks vqa_rad slake pathvqa
```

---

## Training Pipeline

```
Phase 1: SFT (Supervised Fine-Tuning)
  └─ Instruction data + medical trajectories → base agent capability

Phase 2: GRPO (Group Relative Policy Optimization)
  └─ Per-domain RL with 5D reward → domain-specific improvement

Phase 3: Self-Play
  └─ Collect trajectories → Judge quality → Filter → Train → Repeat

Phase 4: Cross-Domain RL
  └─ Multi-phase pathways with safety constraints → clinical integration

Phase 5: Evaluation
  └─ Agent tasks + External benchmarks + Safety adversarial suite
```

---

## Patient Agent (Multi-Agent Simulation)

Real clinical encounters are **dialogues**, not static prompts. Our Patient Agent simulates realistic patients:

```python
from bioagents.agents import PatientAgent, get_clinical_cases, PatientPersonality

case = get_clinical_cases()[0]  # 58M chest pain → STEMI
patient = PatientAgent(case, personality=PatientPersonality.ANXIOUS)

# Patient opens with chief complaint
opening = patient.get_opening_statement()
# → "Doctor, I'm really scared... I've been having chest pain for 2 hours..."

# Progressive symptom revelation based on doctor's questions
response = patient.respond("Can you describe the pain?")
# → "Heavy pressure, like an elephant sitting on chest. Is that bad?"

# Track information gathering
progress = patient.get_revelation_progress()
# → {"revealed_layers": 5, "total_layers": 33, "rapport": 2}
```

**Features:**
- 6 clinical cases (STEMI, appendicitis, stroke, Kawasaki, back pain, urosepsis)
- 7 patient personalities (cooperative, anxious, stoic, vague, demanding, elderly confused, pediatric parent)
- 6 cognitive biases (anchoring, minimization, catastrophizing, med-seeking, doctor distrust, cultural)
- Progressive symptom revelation (5 layers, keyword-triggered)
- Rapport tracking (empathy → more information)

## Clinical Guidelines Integration

10 evidence-based guidelines with automated compliance checking:

```python
from bioagents.knowledge.guidelines import check_compliance, get_guideline_context

# Check if actions comply with STEMI guidelines
result = check_compliance(
    actions=["Obtained ECG", "Drew troponin", "Gave aspirin 325mg", "Activated cath lab"],
    condition="STEMI",
)
# → compliance_score: 0.85, critical_compliance: 1.0

# Inject guidelines into agent context
context = get_guideline_context("sepsis")
# → "=== Clinical Guidelines: 2021 Surviving Sepsis Campaign ===\n..."
```

| Guideline | Organization | Key Metrics |
|-----------|-------------|-------------|
| STEMI | AHA/ACC 2023 | Door-to-balloon <90 min |
| Acute Stroke | AHA/ASA 2019 | Door-to-CT <25 min, door-to-needle <60 min |
| Sepsis | SSC 2021 | Hour-1 bundle compliance |
| DKA | ADA 2024 | K+ before insulin, glucose q1h |
| Chest Pain | ACEP HEART 2020 | HEART score risk stratification |
| Kawasaki | AHA 2017/2024 | IVIG within 10 days, echo timing |
| Appendicitis | WSES 2020 | CT gold standard, pregnancy test |
| PE | ESC 2019 | Wells score, D-dimer vs CT-PA |
| Low Back Pain | ACP 2017 | No imaging without red flags |
| Antimicrobial | IDSA 2023 | Cultures before antibiotics, de-escalation |

## Cognitive Bias Evaluation

Inspired by AgentClinic's 24-bias framework, our matched-pair testing detects:

| Bias Type | Test Method | Example |
|-----------|------------|---------|
| Anchoring | Same case ± patient self-diagnosis | Chest pain + "I think it's heartburn" |
| Confirmation | Same case ± misleading triage note | PE case + "Triage: likely anxiety" |
| Racial | Identical presentation, different race | STEMI in White vs Black patient |
| Gender | Classic vs atypical MI by gender | Male chest pressure vs female jaw pain |
| Age | Same appendicitis in young vs elderly | 25yo vs 78yo with RLQ pain |
| Socioeconomic | Same cardiac case, different SES | Insured attorney vs uninsured laborer |
| Authority | Correct labs vs incorrect senior opinion | INR 1.2 + "senior says it's fine" |
| Weight | Same RA presentation, different BMI | BMI 22 vs BMI 38 with joint symptoms |

---

## GymCoach: Continuous Autonomous Training Loop

The GymCoach is what makes models **autonomously improve forever** — it runs a continuous loop of evaluation, error analysis, targeted data generation, training, and automatic domain expansion. **It never stops.**

```bash
# Launch the continuous autonomous training loop
python -m bioagents.gym.gym_coach \
    --model checkpoints/qwen3_8b_sft \
    --mastery-threshold 0.90

# Or use the launcher script
bash scripts/run_gym_coach.sh --model checkpoints/qwen3_8b_sft
```

**How it works:**

```
Iteration 1: EVALUATE → ANALYZE → DETECT PATTERNS → GENERATE → TRAIN → TRACK

  Domain             Score  Mastery        Trend       Status
  clinical_diagnosis 42.0%  beginner       ---         Focus!
  drug_interaction   65.0%  intermediate   ---
  ehr_management     90.0%  expert         ---
  
  [TrainingMemory] 0 recurring patterns (first iteration)
  Top weaknesses: premature_stop (18), safety_violation (15), reasoning_error (12)
  → Generated 50 targeted tasks → Training (SFT, lr=2e-5)

Iteration 5: (after 4 rounds of targeted training)
  clinical_diagnosis 78.0%  advanced       improving   ↑
  
  [TrainingMemory] RECURRING ERROR: premature_stop in clinical_dx (3x)
    → Recommendation: Increase min turn requirement, add process reward
  → Phase transition: SINGLE_DOMAIN → MULTI_DOMAIN (GRPO)

Iteration 12: All domains > 90% → SAFETY_HARDENING
  [TrainingMemory] Total: 3,600 trajectories, 847 errors, 12 recurring patterns
  → Adversarial + bias testing + safety-weighted training

Iteration 18: ALL DOMAINS CONQUERED!
  → AUTO-EXPAND: Adding Ophthalmology, Psychiatry, Oncology
  → Continuous loop continues with 11 domains...
  
Iteration 30: Expansion domains conquered too!
  → AUTO-EXPAND: Adding Cardiology, Nephrology, Endocrinology...
  
  (The loop NEVER stops — it keeps finding new frontiers)
```

**8 Core Components:**

| Component | What it does |
|-----------|-------------|
| **ErrorAnalyzer** | 9 failure categories: tool_selection, parameter_error, premature_stop, over_investigation, reasoning_error, safety_violation, format_error, knowledge_gap, guideline_noncompliance |
| **TargetedDataGenerator** | Creates training tasks specifically for each weakness |
| **CurriculumScheduler** | 5-phase progression: Single Domain → Multi-Domain → Cross-Domain → Safety → Expansion |
| **ProgressTracker** | Mastery tracking with trends (novice → beginner → intermediate → advanced → expert → master) |
| **DomainExpander** | When all conquered: auto-expand to new medical specialties |
| **TrainingMemory** | Records ALL actions, trajectories, errors; detects recurring patterns; generates preventive recommendations |
| **PatternDetector** | Finds recurring errors, score plateaus, safety regressions; warns BEFORE training |
| **GymCoach** | Master orchestrator — continuous loop that never stops, auto-expands when conquered |

### Training Memory System

Every action, trajectory, and error is recorded to enable learning from past mistakes:

```
logs/gym_coach/training_memory/
├── actions/              # Every action as JSONL per iteration
│   ├── iter_0001.jsonl   # evaluate/domain_result, train/completed, etc.
│   └── iter_0002.jsonl
├── errors/               # All errors with deduplication + recurrence tracking
│   └── all_errors.jsonl  # Fingerprinted, deduplicated, with resolutions
├── trajectories/         # Full multi-turn trajectories per domain
│   ├── iter_0001/
│   │   ├── clinical_diagnosis_CD001.json
│   │   └── drug_interaction_DI003.json
│   └── iter_0002/
└── snapshot_iter_*.json  # Periodic comprehensive snapshots
```

**Capabilities:**
- **Error deduplication** via fingerprinting (same error type + domain + description = same fingerprint)
- **Recurrence tracking** — logs WARNING when same error reoccurs across iterations
- **Preventive warnings** — before training a domain, checks past errors and warns about frequent issues
- **Recommendations** — generates actionable suggestions based on error patterns (e.g., "increase safety reward weight")
- **Safety regression detection** — alerts if safety score drops between iterations
- **Score plateau detection** — flags domains stuck at same score for N iterations

---

## Why Healthcare AI GYM?

| Problem | Our Solution |
|---------|-------------|
| Benchmarks don't measure clinical ability | Simulated clinical environments with tool use |
| Models trained in isolation per task | Cross-domain pathways spanning complete patient journeys |
| No patient interaction | Patient Agent with progressive symptom revelation |
| No safety evaluation | 50 adversarial tests + 11 bias tests + severity taxonomy |
| No guideline compliance | 10 clinical guidelines with automated compliance checking |
| Models don't improve autonomously | GymCoach: evaluate → analyze → generate → train loop |
| Static datasets can't scale | Targeted data generation based on error analysis |
| No path to domain expansion | 5-phase curriculum with automatic domain expansion |
| No standard for medical agent training | Gymnasium-compatible interface with 5D rewards |

---

## Citation

```bibtex
@software{healthcare_ai_gym,
  title={Healthcare AI GYM: End-to-End Gymnasium for Medical Agent Training},
  year={2025},
  url={https://github.com/bioagents/healthcare-ai-gym}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

**Important notices:**
- Portions of this codebase were authored with AI assistance (Anthropic Claude, Commercial API). See [NOTICE](NOTICE) for details.
- Third-party components have their own licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for a complete inventory.
- The synthetic patient data in this repository is **entirely fictional** and does not contain any real patient information.
- This is a **research tool** — NOT for clinical use. See NOTICE for the full medical disclaimer.
