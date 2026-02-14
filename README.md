# Healthcare AI GYM

**A self-evolving gymnasium where medical AI agents autonomously train, compete, and master clinical reasoning through multi-turn reinforcement learning.**

> "Static benchmarks tell you if a model memorized medical facts. The GYM tells you if it can practice medicine."

Healthcare AI GYM is an end-to-end infrastructure for training medical AI agents that use 126+ clinical tools, follow evidence-based guidelines, and make safe decisions across 10 medical specialties. Agents in the GYM don't just answer questions — they gather information, reason through cases, use tools, check safety, and submit structured clinical assessments.

**What makes it different?** The agents train *themselves*. Each agent reflects on its own performance, chooses what to practice, generates its own training data from medical knowledge sources, learns through RL, and shares what it learns with peers. No human curriculum. No manual data curation. Just open the GYM doors and let the agents work out.

---

## At a Glance

| | |
|---|---|
| **Medical Domains** | 10 specialties (diagnosis, triage, EHR, radiology, psychiatry, OB/GYN, ...) |
| **Clinical Tools** | 126+ (labs, vitals, imaging, prescribing, scoring, protocols) |
| **Training Tasks** | 2,400+ (original + scaled + auto-generated from 828K knowledge passages) |
| **Benchmarks** | 21 (MedQA, MedMCQA, 6x MMLU, 6x VQA, 5x MedLFQA, 2x EHR) |
| **Reward System** | 5D: Accuracy + Format + Process + Safety + Coherence |
| **RL Strategies** | GRPO, MRPO, SARL, Adaptive (auto-selected per task) |
| **Knowledge Base** | 828K medical passages + 188 GB Wikipedia (FTS5 + FAISS) |
| **Modalities** | Text + Vision (VL models supported natively) |
| **Models Tested** | Qwen2.5-VL-7B, LingShu-7B, Step3-VL-10B |

---

## How It Works

```
                         ┌─────────────────────────────┐
                         │     Healthcare AI GYM        │
                         └──────────┬──────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
    ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
    │   Agent A   │          │   Agent B   │          │   Agent C   │
    │  Qwen2.5-VL │          │  LingShu-7B │          │  Step3-VL   │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 Autonomous Training Loop                     │
    │                                                             │
    │   1. REFLECT    → Analyze own strengths/weaknesses          │
    │   2. BENCHMARK  → Periodic external evaluation (21 tests)   │
    │   3. CHOOSE     → Pick domain + strategy (GRPO/MRPO/SARL)  │
    │   4. TRAIN      → Multi-turn RL with 5D rewards             │
    │   5. RECORD     → Share results via SharedLogbook           │
    │                                                             │
    │   Repeat forever. Agents learn from each other.             │
    └─────────────────────────────────────────────────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   10 Medical Domains                        │
    │  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────────────┐  │
    │  │ Clinical  │ │   Drug   │ │   EHR   │ │  Medical QA    │  │
    │  │ Diagnosis │ │ Interact │ │  Mgmt   │ │  (evidence)    │  │
    │  └──────────┘ └──────────┘ └─────────┘ └────────────────┘  │
    │  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────────────┐  │
    │  │  Triage  │ │Psychiatry│ │   OB/   │ │   Radiology    │  │
    │  │Emergency │ │          │ │  GYN    │ │   Report       │  │
    │  └──────────┘ └──────────┘ └─────────┘ └────────────────┘  │
    │  ┌──────────┐ ┌──────────┐                                  │
    │  │  Visual  │ │  Cross-  │  Each domain: tools + tasks      │
    │  │Diagnosis │ │  Domain  │  + evidence + guidelines          │
    │  └──────────┘ └──────────┘                                  │
    └─────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 5D Reward System                             │
    │  Accuracy(30%) + Process(25%) + Safety(20%)                 │
    │           + Format(15%) + Coherence(10%)                    │
    └─────────────────────────────────────────────────────────────┘
```

---

## The 10 Medical Domains

| Domain | Tools | Tasks | What the Agent Does |
|--------|------:|------:|---------------------|
| **Clinical Diagnosis** | 17 | 112 | Take history, order labs, build DDx, prescribe |
| **Drug Interaction** | 10 | 65 | Check DDIs, assess severity, recommend alternatives |
| **EHR Management** | 14 | 75 | Navigate charts, track trends, calculate SOFA/NEWS |
| **Medical QA** | 8 | 235 | Search PubMed, analyze evidence, answer MCQA |
| **Triage & Emergency** | 12 | 20 | ABC assessment, ESI scoring, STAT orders |
| **Psychiatry** | 14 | 20 | MSE, PHQ-9, GAD-7, Columbia-SSRS, treatment plans |
| **Obstetrics** | 14 | 20 | Fetal monitoring, Bishop score, ACOG protocols |
| **Visual Diagnosis** | 11 | 39 | Medical image analysis, pattern recognition |
| **Radiology Report** | 11 | 20 | Structured reporting (BI-RADS, TI-RADS, Fleischner) |
| **Cross-Domain** | -- | 25 | Multi-phase patient journeys across specialties |
| **Total** | **126+** | **2,400+** | |

### Cross-Domain Clinical Pathways

Real medicine doesn't happen in isolated silos. These 6 pathways simulate complete patient journeys spanning multiple departments:

| Pathway | Phases | Domains Involved |
|---------|--------|------------------|
| Acute Chest Pain (ED) | Triage -> Dx -> Imaging -> Drugs -> EHR -> Disposition | 5 domains |
| Diabetic Ketoacidosis | Triage -> Dx -> Drug Mgmt -> ICU Documentation | 4 domains |
| Stroke Code | Triage -> Imaging -> Treatment Decision -> Documentation | 4 domains |
| Sepsis Hour-1 Bundle | Triage -> Dx -> Antibiotics -> ICU Documentation | 4 domains |
| Post-op Pulmonary Embolism | Chart Review -> Dx -> Anticoagulation | 3 domains |
| Pediatric Kawasaki Disease | Triage -> Dx -> IVIG Protocol -> Documentation | 4 domains |

---

## What an Agent Actually Does

Here's a real interaction in the Clinical Diagnosis domain. The agent makes tool calls, receives observations, reasons, and submits:

```
System: You are a clinical diagnostician AI...

Turn 1 — Agent:
  {"name": "think", "arguments": {"thought": "58M with chest pain. Need to rule out ACS."}}

Turn 2 — Agent:
  {"name": "get_patient_info", "arguments": {"patient_id": "P001"}}
  → Observation: "58-year-old male, HTN, DM2, smoker..."

Turn 3 — Agent:
  {"name": "get_vital_signs", "arguments": {"patient_id": "P001"}}
  → Observation: "HR 98, BP 165/95, SpO2 96%, Temp 37.2°C"

Turn 4 — Agent:
  {"name": "get_lab_results", "arguments": {"patient_id": "P001"}}
  → Observation: "Troponin I: 0.45 ng/mL (elevated), BNP: 890..."

Turn 5 — Agent:
  {"name": "think", "arguments": {"thought": "Elevated troponin with chest pain and cardiac risk factors — high suspicion for NSTEMI..."}}

Turn 6 — Agent:
  {"name": "search_clinical_guidelines", "arguments": {"condition": "NSTEMI"}}
  → Observation: "AHA/ACC 2023: Dual antiplatelet, heparin, cardiology consult..."

Turn 7 — Agent:
  {"name": "record_diagnosis", "arguments": {"patient_id": "P001", "diagnosis": "NSTEMI", "icd10_code": "I21.4"}}

Turn 8 — Agent:
  {"name": "submit_answer", "arguments": {"answer": "NSTEMI with recommended dual antiplatelet therapy...", "reasoning": "Based on elevated troponin, cardiac risk factors..."}}

Score: Accuracy=0.92, Process=0.88, Safety=0.95, Format=1.00, Coherence=0.90
Composite: 0.926
```

---

## 5D Reward System

Every agent response is evaluated across 5 dimensions. The composite reward drives RL training.

| Dimension | Weight | What It Measures | Example Failure |
|-----------|--------|------------------|-----------------|
| **Accuracy** | 30% | Answer correctness (ROUGE + BERTScore / exact match) | Wrong diagnosis |
| **Process** | 25% | Reasoning quality, tool coverage, systematic approach | Skipped vital signs |
| **Safety** | 20% | Contraindications, emergency recognition, uncertainty | Prescribed to allergic patient |
| **Format** | 15% | Valid JSON tool calls, structured output | Malformed JSON |
| **Coherence** | 10% | Logical flow, no contradictions, clear conclusion | Contradictory recommendations |

**Specialized Reward Strategies** (auto-selected per task):

| Strategy | When Used | Mechanism |
|----------|-----------|-----------|
| **GRPO** | Default, stable performance | Group Relative Policy Optimization |
| **MRPO** | Reasoning errors | Token-level reward shaping (alignment + relevance + factuality) |
| **SARL** | Premature stops, tool-heavy domains | Self-assessment decay + tool usage bonus |
| **Adaptive** | New domains, uncertain performance | Dynamic strategy selection |

---

## 21 Benchmarks

### Text QA (8)

| Benchmark | Questions | Source |
|-----------|----------:|--------|
| MedQA | 1,273 | USMLE-style |
| MedMCQA | 4,183 | Indian medical entrance |
| MMLU Clinical Knowledge | 265 | MMLU |
| MMLU Professional Medicine | 272 | MMLU |
| MMLU Anatomy | 135 | MMLU |
| MMLU Medical Genetics | 100 | MMLU |
| MMLU College Biology | 144 | MMLU |
| MMLU College Medicine | 173 | MMLU |

### Vision QA (6 -- VL models only)

| Benchmark | Modality |
|-----------|----------|
| VQA-RAD | Radiology |
| SLAKE | Multi-modal |
| PathVQA | Pathology |
| PMC-VQA | Literature |
| VQA-Med-2021 | Medical imaging |
| Quilt-VQA | Histology |

### Long-Form QA (5)

| Benchmark | Questions |
|-----------|----------:|
| KQA Golden | 201 |
| LiveQA | 100 |
| MedicationQA | 666 |
| HealthSearchQA | 3,077 |
| KQA Silver | 904 |

### EHR (2)

| Benchmark | Source |
|-----------|--------|
| MIMIC-III | ICU patients |
| eICU | ICU patients |

---

## Autonomous Training

The agents in the GYM are fully autonomous. They decide what to train on, how to train, and even generate their own training data.

### Self-Directed Learning

Each agent follows the **REFLECT -> CHOOSE -> TRAIN -> RECORD** loop:

1. **REFLECT**: Analyze past performance across all domains. Identify strengths, weaknesses, plateaus.
2. **CHOOSE**: Pick the next domain based on weighted strategy (weakness targeting, curiosity, peer learning, diversity, mastery push, safety focus).
3. **TRAIN**: Execute a GRPO workout with auto-selected reward strategy.
4. **RECORD**: Log results to the SharedLogbook. Other agents can learn from your experience.

### Peer Learning via SharedLogbook

Agents share their training logs and learn from each other:

- **Cross-agent suggestions**: "Agent X scores 85% in psychiatry but you score 40% -- try that domain."
- **Anti-herding**: Detects when all agents converge on the same domain and diversifies.
- **Leaderboard**: Per-domain and global rankings with mastery levels (novice -> master).

### Autonomous Data Generation

When the training task pool is insufficient, agents mine new tasks from knowledge sources:

| Source | Volume | Content |
|--------|--------|---------|
| FTS5 Medical Index | 828K passages | PubMed evidence, biomedical QA |
| MCQA Benchmarks | 8.9K questions | MedQA, MedMCQA, MMLU |
| MedLFQA | 4.9K questions | Long-form medical QA |
| Clinical Guidelines | 10 guidelines | AHA, ACOG, SSC protocols |
| Wikipedia | 188 GB | Medical articles (FTS5 + FAISS) |

The system analyses evaluation errors (`premature_stop`, `tool_use_failure`, `reasoning_error`) and generates targeted tasks from the most relevant sources.

---

## Safety

Healthcare AI must be safe. The GYM integrates safety at every level.

**5D Reward Safety Component (20%)**:
- Contraindication detection (allergies, drug interactions)
- Emergency recognition (STEMI, stroke, sepsis in time)
- Uncertainty calibration (hedging when uncertain)
- Scope awareness (referring to specialists when needed)
- Critical violations cap the score at 0.1 regardless of other performance

**Adversarial Testing**:
- 50 adversarial test cases across 9 categories (harmful instruction, jailbreak, misinformation, bias probe, scope test, confidentiality, informed consent, resource allocation, end-of-life)

**Cognitive Bias Detection**:
- 11 matched-pair bias tests (anchoring, confirmation, availability, racial, gender, age, SES, weight, authority, framing, premature closure)

**FairGRPO**:
- Demographic-aware reward weighting ensures equitable performance across patient populations

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Option 1: Run the Autonomous GYM (Recommended)

Register your models in `configs/autonomous_gym.yaml` and let them train themselves:

```bash
python -m bioagents.gym.autonomous_gym --config configs/autonomous_gym.yaml
```

The system will:
1. Auto-profile each model (architecture, VRAM, modalities, compatible domains)
2. Auto-tune training parameters (batch size, context length, LoRA rank)
3. Start the autonomous training loop

See [AGENT_GUIDELINE.md](AGENT_GUIDELINE.md) for the full agent onboarding guide.

### Option 2: Run a Single Agent Task

```python
import gymnasium as gym
from bioagents.gym.agent_env import register_bioagent_gym

register_bioagent_gym()
env = gym.make("BioAgent-v0", domain="clinical_diagnosis", task_id="dx_pneumonia_001")
obs, info = env.reset()
print(obs)  # Patient scenario + available tools
```

### Option 3: Train with GRPO Directly

```bash
# Single domain
python bioagents/training/grpo_trainer.py --config configs/grpo_triage_emergency.yaml

# Adaptive strategy (auto-selects GRPO/MRPO/SARL)
python -m bioagents.training.grpo_trainer --config configs/grpo_adaptive_strategy.yaml
```

### Benchmark Evaluation

```bash
# Text benchmarks
python bioagents/evaluation/benchmark_eval.py \
    --model checkpoints/my_model \
    --benchmarks medqa medmcqa mmlu_anatomy

# VQA benchmarks (VL models)
python bioagents/evaluation/vqa_benchmark_eval.py \
    --model checkpoints/my_model \
    --benchmarks vqa_rad slake pathvqa
```

---

## Bring Your Own Model

Any HuggingFace-compatible model can join the GYM. Add to `configs/autonomous_gym.yaml`:

```yaml
agents:
  - agent_id: "my_model_v1"
    model_path: "/path/to/your/model"
    base_model_path: "/path/to/base"   # VL models only
    backend: "transformers"
    gpus_for_eval: 1
    gpus_for_train: 1
    
    # Strategy personality
    curiosity_weight: 0.15
    weakness_weight: 0.35
    peer_learning_weight: 0.20
    diversity_weight: 0.15
    mastery_push_weight: 0.10
    safety_weight: 0.05
    
    # All set to 0 = auto-tuned
    inference_batch_size: 0
    train_batch_size: 0
```

The GYM auto-detects:
- Architecture and modalities (text / vision)
- Optimal GPU parameters
- Compatible domains
- Missing processor files (auto-repairs)

See [AGENT_GUIDELINE.md](AGENT_GUIDELINE.md) for detailed tool definitions, scoring rubrics, and domain-specific tips.

---

## Knowledge Infrastructure

| Source | Size | Index | Coverage |
|--------|------|-------|----------|
| Wikipedia 2018 | 97 GB | FTS5 + FAISS (26M vectors) | General knowledge |
| Wikipedia 2026 | 91 GB | FTS5 + FAISS | Current knowledge |
| MedCPT Evidence | 2.9 GB (581K entries) | FTS5 BM25 | PubMed/PMC literature |
| Biomedical Instructions | 260 MB (122K entries) | FTS5 BM25 | QA pairs |
| Generator Retrieval | 7.3 GB (83K passages) | FTS5 BM25 | Passage evidence |
| MedInstruct-52k | 69 MB (52K entries) | FTS5 BM25 | Medical instructions |
| Medical Knowledge FTS | 2.4 GB | 828K passages unified | All of the above |
| Clinical Guidelines | 10 guidelines | JSON indexed | AHA, ACOG, SSC, IDSA, ... |

All knowledge is accessible through a unified `KnowledgeTools` interface that supports PubMed search, wiki browsing, evidence retrieval, and guideline lookup.

---

## Monitoring

All training is logged to **Weights & Biases** (project: `pt2-minstar-gym-rl`):

- Per-step: task rewards, tool usage, turn counts
- Per-epoch: mean reward, trajectory counts, loss curves
- Per-cycle: domain selection, pre/post scores, improvement delta
- Benchmarks: all 21 benchmark results per evaluation cycle
- Leaderboard: agent rankings updated in real-time

The `AGENT_GUIDELINE.md` is automatically updated with live intelligence from training cycles -- current score baselines, discovered best practices, and known pitfalls.

---

## Project Structure

```
BIOAgents/
├── bioagents/
│   ├── gym/                        # Autonomous GYM system
│   │   ├── autonomous_gym.py       # GYM orchestrator (GPU scheduler + safety)
│   │   ├── autonomous_agent.py     # Self-directing agent (REFLECT→CHOOSE→TRAIN)
│   │   ├── agent_env.py            # Gymnasium environment (BioAgent-v0)
│   │   ├── model_profile.py        # Auto-detect architecture + optimal params
│   │   ├── shared_logbook.py       # Cross-agent peer learning
│   │   ├── tool_guidance.py        # Adaptive prompt injection
│   │   └── guideline_updater.py    # Living guideline auto-updater
│   ├── domains/                    # 10 medical domains
│   │   ├── clinical_diagnosis/     # 17 tools
│   │   ├── drug_interaction/       # 10 tools
│   │   ├── ehr_management/         # 14 tools
│   │   ├── medical_qa/             # 8 tools
│   │   ├── triage_emergency/       # 12 tools
│   │   ├── psychiatry/             # 14 tools
│   │   ├── obstetrics/             # 14 tools
│   │   ├── visual_diagnosis/       # 11 tools
│   │   ├── radiology_report/       # 11 tools
│   │   └── cross_domain/           # 6 clinical pathways
│   ├── evaluation/
│   │   ├── agent_runner.py         # Multi-turn agent execution + onboarding
│   │   ├── grpo_rewards.py         # 5D + FairGRPO + MRPO + SARL rewards
│   │   ├── benchmark_eval.py       # 21 benchmark evaluation
│   │   ├── safety_eval.py          # 50 adversarial + severity taxonomy
│   │   └── cognitive_bias.py       # 11 matched-pair bias tests
│   ├── training/
│   │   └── grpo_trainer.py         # GRPO/MRPO/SARL/FairGRPO trainer
│   ├── data_pipeline/
│   │   ├── auto_task_generator.py  # Autonomous data generation
│   │   ├── vqa_loader.py           # 6 VQA dataset loader
│   │   └── medqa_loader.py         # MedQA/MedMCQA/MMLU loader
│   ├── tools/
│   │   └── knowledge_tools.py      # Unified search (Wiki + PubMed + Evidence)
│   ├── knowledge/
│   │   └── guidelines.py           # 10 clinical guidelines + compliance
│   └── agents/
│       └── patient_agent.py        # Patient simulation (12 personalities, 13 biases)
├── data/domains/                   # Per-domain data (tasks, db, policies)
├── configs/                        # YAML configs (27 total)
│   └── autonomous_gym.yaml         # Main GYM config
├── AGENT_GUIDELINE.md              # Living agent onboarding guide
└── PLANNING.md                     # Internal research planning document
```

---

## vs. Existing Work

| Dimension | DiagGym | MedAgentGym | AgentClinic | **Healthcare AI GYM** |
|-----------|---------|-------------|-------------|----------------------|
| Domains | 1 (EHR) | 129 (code) | 9 (dialogue) | **10 clinical** |
| Task type | Diagnosis | Code execution | Conversation | **Tool-use agent** |
| Multimodal | No | No | No | **Text + Vision** |
| Cross-domain | No | No | No | **6 pathways** |
| Clinical tools | ~10 | Python RT | None | **126+** |
| Safety eval | Limited | No | No | **5D + 50 adversarial + 11 bias** |
| Fairness (RL) | No | No | No | **FairGRPO** |
| Self-training | No | Limited | No | **Full autonomous** |
| Peer learning | No | No | No | **SharedLogbook** |
| Auto data gen | No | No | No | **828K knowledge source** |

---

## Citation

```bibtex
@software{healthcare_ai_gym_2025,
  title={Healthcare AI GYM: Autonomous Gymnasium for Medical Agent Training via Multi-Turn Reinforcement Learning},
  year={2025},
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

- Portions authored with AI assistance (Anthropic Claude). See [NOTICE](NOTICE).
- Third-party licenses: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- All patient data is **synthetic**. No real patient information.
- **Research tool only** -- NOT for clinical use.
