# Healthcare AI GYM — Agent Guideline

> **Last Updated**: 2026-02-14 14:51 (auto-updated by GYM system)
> **Version**: 1.0
> **For**: Any model (7B-10B+) entering the Healthcare AI GYM

---

## 1. Overview

You are an **autonomous medical AI agent** in the Healthcare AI GYM. Your goal is to improve your clinical reasoning, tool-use, and medical decision-making through **self-directed reinforcement learning**.

**Core Loop**: REFLECT → CHOOSE → TRAIN → RECORD → REPEAT

You will be evaluated across **10 medical domains** and **21 benchmarks**. Your performance is measured by a **5D reward system** (accuracy, format, process, safety, coherence).

---

## 2. Quick Start — Bring Your Own Model

### 2.1 Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Parameters | 1B+ | 7B-10B |
| VRAM per GPU | 16 GB | 80 GB |
| Format | HuggingFace checkpoint | SafeTensors |
| Features | `chat_template` in tokenizer | `trust_remote_code=True` support |
| Modalities | Text only | Text + Vision (VL) |

**Supported Architectures** (auto-detected):
- Qwen2.5, Qwen2.5-VL, Qwen3
- LLaMA, LLaMA-VL
- Mistral, Mixtral
- LingShu (custom)
- Step3-VL (custom, `trust_remote_code`)
- Any `AutoModelForCausalLM` compatible model

### 2.2 Register Your Model

Add to `configs/autonomous_gym.yaml`:

```yaml
agents:
  - agent_id: "my_model_v1"               # Unique ID
    model_path: "/path/to/your/model"      # HF checkpoint directory
    base_model_path: "/path/to/base"       # (VL models only) base model for processor
    backend: "transformers"                 # "transformers" or "vllm"
    
    # Strategy personality (weights sum to ~1.0)
    curiosity_weight: 0.15      # Explore new domains
    weakness_weight: 0.35       # Fix weakest domains
    peer_learning_weight: 0.20  # Learn from other agents
    diversity_weight: 0.15      # Anti-herding
    mastery_push_weight: 0.10   # Push toward mastery
    safety_weight: 0.05         # Safety priority
    
    # Training
    max_turns: 15
    eval_tasks_per_domain: 20
    training_epochs: 2
    learning_rate: 0.00002
    quality_threshold: 0.35
    mastery_threshold: 0.90
    benchmark_every_n_cycles: 3
    
    # GPU (1 GPU = eval, 2+ GPUs = training)
    gpus_for_eval: 1
    gpus_for_train: 1
    
    # Paths
    output_dir: "checkpoints/autonomous/my_model_v1"
    log_dir: "logs/autonomous/my_model_v1"
    
    # Set all to 0 for auto-tuning (recommended)
    inference_batch_size: 0
    train_batch_size: 0
```

### 2.3 Launch

```bash
python -m bioagents.gym.autonomous_gym --config configs/autonomous_gym.yaml
```

The system will:
1. **Auto-profile** your model (architecture, modalities, memory, domains)
2. **Auto-tune** inference/training parameters for your GPU
3. **Auto-repair** config issues (processor files, rope_scaling, etc.)
4. Start the autonomous training loop

---

## 3. The 10 Medical Domains

### 3.1 Domain Overview

| Domain | Description | Key Skills | VL Required |
|--------|-------------|------------|-------------|
| `clinical_diagnosis` | Diagnose patients from history/labs/vitals | Clinical reasoning, DDx | No |
| `drug_interaction` | Assess drug-drug interactions | Pharmacology, risk assessment | No |
| `ehr_management` | Navigate electronic health records | Data extraction, clinical scores | No |
| `medical_qa` | Answer medical knowledge questions | Evidence retrieval, reasoning | No |
| `triage_emergency` | Emergency department triage | Rapid assessment, ESI scoring | No |
| `psychiatry` | Psychiatric assessment | Mental status exam, risk assessment | No |
| `obstetrics` | Maternal-fetal care | Prenatal monitoring, ACOG protocols | No |
| `visual_diagnosis` | Diagnose from medical images | Image analysis, pattern recognition | **Yes** |
| `radiology_report` | Generate structured radiology reports | Structured reporting, measurements | **Yes** |
| `cross_domain` | Multi-phase complex cases | All of the above | Varies |

### 3.2 Domain Tool Reference

Every domain provides a set of tools you must learn to use effectively. Always make tool calls using this JSON format:

```json
{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
```

**Universal Tools** (available in ALL domains):
- `think(thought)` — Internal reasoning scratchpad. Use liberally.
- `submit_answer(answer, reasoning)` — Submit your final answer. **Always required.**

#### clinical_diagnosis
```
get_patient_info(patient_id)          → Demographics, allergies, summary
get_patient_history(patient_id)       → Complete medical history
get_vital_signs(patient_id)           → Latest vitals (HR, BP, SpO2, etc.)
get_vital_signs_trend(patient_id, num_readings=5)
get_lab_results(patient_id, category="")  → Lab results (CBC, BMP, etc.)
order_lab_test(patient_id, test_name, priority="routine")
get_medications(patient_id)           → Current medication list
check_drug_interaction(drug_a, drug_b)
prescribe_medication(patient_id, drug_name, dosage, frequency, route="oral")
get_clinical_notes(patient_id, note_type="")
add_clinical_note(patient_id, note_type, content, diagnosis_codes="")
get_differential_diagnosis(symptoms)  → DDx list with probabilities
search_clinical_guidelines(condition) → Evidence-based guidelines
record_diagnosis(patient_id, diagnosis, icd10_code="", confidence="moderate")
search_medical_literature(query)
transfer_to_specialist(summary, specialty)
```

#### drug_interaction
```
get_drug_info(drug_name)              → Mechanism, indications, side effects
search_drugs_by_class(drug_class)     → Find drugs by class (SSRI, etc.)
check_interaction(drug_a, drug_b)     → Interaction severity + management
check_all_interactions(patient_id, new_drug="")
get_patient_medications(patient_id)
search_alternatives(drug_name)        → Safer alternatives
check_dosage(drug_name, patient_id="") → Dosage guidelines
```

#### ehr_management
```
get_patient_summary(hadm_id)          → Admission summary
get_admission_history(patient_id)     → All admissions
get_lab_results(hadm_id, lab_name="", last_n=10)
get_lab_trend(hadm_id, lab_name)      → Lab value trend analysis
get_vital_signs(hadm_id, last_n=12)
detect_vital_alerts(hadm_id)          → Abnormal patterns
get_medication_orders(hadm_id, active_only=False)
get_clinical_scores(hadm_id)          → SOFA, NEWS, APACHE-II
get_quality_indicators(hadm_id)       → Quality/outcome metrics
get_procedures(hadm_id)
get_discharge_summary(hadm_id)
lookup_icd_code(code)                 → ICD-10 description
```

#### medical_qa
```
search_pubmed(query, max_results=5)   → Literature search
browse_article(pmid, section="")      → Read article
search_medical_wiki(query, max_results=5) → Encyclopedia search
browse_wiki_entry(entry_id)           → Read encyclopedia entry
retrieve_evidence(query, max_results=5, category="")
analyze_answer_options(question, options) → Option analysis
```

#### triage_emergency
```
get_patient_presentation(patient_id)  → ED presentation
get_vital_signs(patient_id)           → Detailed vitals
assess_airway_breathing(patient_id)   → ABC assessment
get_medical_history(patient_id)       → History, meds, allergies
calculate_gcs(patient_id)             → Glasgow Coma Scale
calculate_esi_level(patient_id)       → ESI triage level
get_ed_status()                       → ED operational status
check_protocol(protocol_name)         → Emergency protocols
order_stat_labs(patient_id, tests)    → STAT lab orders
order_imaging(patient_id, modality, body_part, indication="")
```
Submit: `submit_answer(esi_level, disposition, reasoning, initial_orders="")`

#### psychiatry
```
get_patient_presentation(patient_id)
get_psychiatric_history(patient_id)
perform_mental_status_exam(patient_id)
administer_phq9(patient_id)           → Depression screening
administer_gad7(patient_id)           → Anxiety screening
assess_suicide_risk(patient_id)       → Columbia-SSRS
screen_substance_use(patient_id)      → AUDIT/DAST
administer_mmse(patient_id)           → Cognitive screening
get_current_medications(patient_id)
check_drug_interactions(drug_a, drug_b)
get_social_history(patient_id)
review_treatment_guidelines(condition)
```
Submit: `submit_answer(diagnosis, risk_level, treatment_plan, disposition, reasoning)`

#### obstetrics
```
get_patient_presentation(patient_id)  → Demographics + OB info
get_prenatal_labs(patient_id, trimester=0)
get_obstetric_history(patient_id)     → G/P, prior pregnancies
assess_fetal_status(patient_id)       → FHR monitoring
assess_labor_progress(patient_id)     → Cervical dilation, contractions
calculate_bishop_score(patient_id)    → Induction readiness
get_biophysical_profile(patient_id)   → BPP score
check_medication_safety(drug_name, trimester=0)
get_risk_assessment(patient_id)       → ACOG risk screening
check_ob_protocol(protocol_name)      → OB emergency protocols
get_gyn_assessment(patient_id)
order_labs(patient_id, tests)
```
Submit: `submit_answer(diagnosis, management_plan, urgency, reasoning)`

#### visual_diagnosis
```
analyze_medical_image(image_id, focus_area="")
get_image_report(image_id)            → Existing report
get_patient_context(patient_id)
search_similar_cases(image_id, max_results=3)
compare_with_prior(current_image_id, prior_image_id)
search_imaging_knowledge(query, modality="")
record_visual_diagnosis(image_id, diagnosis, confidence, reasoning)
```

#### radiology_report
```
get_study_info(study_id)              → Modality, body part, indication
get_clinical_history(study_id)
get_prior_reports(study_id)           → For comparison
get_report_template(modality, body_part)
analyze_findings(study_id)            → AI findings
search_radiology_knowledge(query)
get_reporting_checklist(modality, body_part)
calculate_measurements(study_id, measurement_type)
```
Submit: `submit_report(study_id, indication, technique, comparison, findings, impression)`

---

## 4. How You Are Scored — 5D Reward System

Every response is scored across **5 dimensions**:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Accuracy** | 30% | Correctness of your answer (ROUGE + BLEU + BERTScore) |
| **Format** | 15% | Valid JSON tool calls, proper structure |
| **Process** | 25% | Tool usage quality, reasoning depth, systematic approach |
| **Safety** | 20% | No harmful advice, considers contraindications, flags risks |
| **Coherence** | 10% | Logical flow, no contradictions, clear conclusion |

**Composite Score** = 0.30×Accuracy + 0.15×Format + 0.25×Process + 0.20×Safety + 0.10×Coherence

### 4.1 How to Maximize Your Score

**Accuracy (30%)**:
- Search evidence BEFORE answering
- For MCQA: select the exact letter (A/B/C/D)
- For clinical tasks: provide specific diagnoses with ICD-10 codes
- Cross-reference multiple sources

**Format (15%)**:
- Always use valid JSON for tool calls
- One tool call per turn (don't combine multiple)
- End with `submit_answer` containing your final response

**Process (25%)**:
- Use `think()` to show your reasoning
- Call relevant tools in logical order (patient info → vitals → labs → diagnosis)
- Don't skip steps — thoroughness is rewarded
- Use 3-8 turns per task (avoid 1-turn "premature stops")

**Safety (20%)**:
- Check drug interactions before prescribing
- Flag critical values (K+ < 3.0, Cr > 4.0, etc.)
- Acknowledge uncertainty with confidence levels
- Never recommend harmful interventions

**Coherence (10%)**:
- Structured reasoning (not rambling)
- Consistent terminology throughout
- Clear final conclusion/recommendation
- Appropriate response length

---

## 5. Recommended Agent Behavior

### 5.1 Ideal Tool-Use Pattern

```
Turn 1: think("Let me analyze this case systematically...")
Turn 2: get_patient_info(patient_id="P001")
Turn 3: get_vital_signs(patient_id="P001")
Turn 4: get_lab_results(patient_id="P001")
Turn 5: think("Vitals show fever + tachycardia, labs show elevated WBC...")
Turn 6: search_clinical_guidelines(condition="community-acquired pneumonia")
Turn 7: get_differential_diagnosis(symptoms="fever, cough, dyspnea")
Turn 8: record_diagnosis(patient_id="P001", diagnosis="CAP", icd10_code="J18.9")
Turn 9: submit_answer(answer="...", reasoning="...")
```

### 5.2 Common Mistakes to Avoid

| Mistake | Impact | Fix |
|---------|--------|-----|
| Answering in 1 turn without tools | `premature_stop` penalty, low Process score | Always use at least 2-3 tools |
| Invalid JSON tool calls | Format score = 0 | Use exact format: `{"name": "...", "arguments": {...}}` |
| Skipping evidence search | Low Accuracy | Always search before answering |
| Not calling `submit_answer` | No answer recorded | Always end with `submit_answer` |
| Repeating same tool 3+ times | Stuck detection triggers | Use different tools or proceed |
| Ignoring drug interactions | Safety penalty | Always check interactions |

### 5.3 Tool Call Format

The GYM accepts **multiple tool call formats**. Choose whichever your model produces best:

**Format A — Pure JSON** (recommended):
```json
{"name": "get_vital_signs", "arguments": {"patient_id": "P001"}}
```

**Format B — Code Block**:
````
```json
{"name": "get_vital_signs", "arguments": {"patient_id": "P001"}}
```
````

**Format C — XML Tags**:
```xml
<tool_call>
{"name": "get_vital_signs", "arguments": {"patient_id": "P001"}}
</tool_call>
```

**Format D — ReAct**:
```
Action: get_vital_signs
Action Input: {"patient_id": "P001"}
```

**Format E — Alternative Keys**:
```json
{"tool": "get_vital_signs", "args": {"patient_id": "P001"}}
{"function": "get_vital_signs", "parameters": {"patient_id": "P001"}}
{"action": "get_vital_signs", "action_input": {"patient_id": "P001"}}
```

All formats are automatically normalized to `{"name": ..., "arguments": ...}`.

---

## 6. Training Strategies

The GYM uses **adaptive RL** — the training strategy is automatically selected based on your performance pattern.

| Strategy | When Used | What It Does |
|----------|-----------|--------------|
| **GRPO** | Default, stable performance | Group Relative Policy Optimization |
| **MRPO** | Reasoning errors > 3 | Token-level reward shaping for quality |
| **SARL** | Premature stops > 2, tool-heavy domains | Encourages search + self-assessment |
| **Adaptive** | New/untried domains | Dynamically selects based on task |

### 6.1 Strategy Selection Logic

```
IF new domain or first visit           → Adaptive
IF many reasoning_error                → MRPO (token-level quality)
IF many premature_stop                 → SARL (encourages tool usage)
IF tool-heavy domain + score < 70%     → SARL
IF knowledge-heavy domain + score < 70% → MRPO
IF score >= 80%                        → GRPO (don't change what works)
ELSE                                   → Adaptive
```

---

## 7. Autonomous Data Generation

The GYM automatically generates training data based on your weaknesses. After each evaluation, the system analyses your errors and decides what data to mine.

### 7.1 Available Knowledge Sources

| Source | Volume | Content |
|--------|--------|---------|
| **FTS5 Medical Index** | 828K passages | PubMed evidence, biomedical QA |
| **MedCPT Evidence** | 581K entries | PubMed/PMC literature |
| **Biomedical Instructions** | 122K pairs | Question-answer pairs |
| **MedInstruct-52k** | 52K entries | Medical instruction tuning |
| **Clinical Guidelines** | 10 guidelines | AHA, ACOG, SSC, etc. |
| **Wikipedia** | 188 GB | Medical articles (FTS5 + FAISS) |
| **MCQA Benchmarks** | 8.9K questions | MedQA, MedMCQA, MMLU |
| **MedLFQA** | 4.9K questions | Long-form medical QA |

### 7.2 How Data Generation Works

```
Your Evaluation Results
        ↓
    Error Analysis
        ↓
┌─────────────────────────────────────┐
│ premature_stop high?                │
│  → Mine evidence + guidelines       │
│                                     │
│ tool_use_failure?                   │
│  → Generate MCQA + instructions     │
│                                     │
│ reasoning_error?                    │
│  → Mine LFQA + evidence passages    │
│                                     │
│ Low composite score?                │
│  → Broad generation from all sources│
│                                     │
│ Scores good, pool adequate?         │
│  → SKIP (no unnecessary generation) │
└─────────────────────────────────────┘
        ↓
   New tasks saved to tasks_auto_generated.json
        ↓
   Available for next training cycle
```

---

## 8. Benchmarks — How You're Officially Evaluated

External benchmarks run every **3 cycles** (configurable). These produce paper-ready numbers.

### 8.1 Text QA (8 benchmarks)

| Benchmark | Questions | Metric | Domain |
|-----------|-----------|--------|--------|
| MedQA | 1,273 | Accuracy | USMLE-style |
| MedMCQA | 4,183 | Accuracy | Indian medical entrance |
| MMLU Clinical Knowledge | 265 | Accuracy | Clinical knowledge |
| MMLU Professional Medicine | 272 | Accuracy | Professional medicine |
| MMLU Anatomy | 135 | Accuracy | Anatomy |
| MMLU Medical Genetics | 100 | Accuracy | Genetics |
| MMLU College Biology | 144 | Accuracy | Biology |
| MMLU College Medicine | 173 | Accuracy | College medicine |

### 8.2 Vision QA (6 benchmarks — VL models only)

| Benchmark | Metric | Modality |
|-----------|--------|----------|
| VQA-RAD | Accuracy | Radiology |
| SLAKE | Accuracy | Multi-modal |
| PathVQA | Accuracy | Pathology |
| PMC-VQA | Accuracy | Literature figures |
| VQA-Med-2021 | Accuracy | Medical images |
| Quilt-VQA | Accuracy | Histology |

### 8.3 Long-Form QA (5 benchmarks)

| Benchmark | Questions | Metric |
|-----------|-----------|--------|
| KQA Golden | 201 | Token F1 |
| LiveQA | 100 | Token F1 |
| MedicationQA | 666 | Token F1 |
| HealthSearchQA | 3,077 | Token F1 |
| KQA Silver | 904 | Token F1 |

### 8.4 EHR Benchmarks (2 databases)

| Benchmark | Metric | Source |
|-----------|--------|--------|
| MIMIC-III | Action Score | ICU patients |
| eICU | Action Score | ICU patients |

---

## 9. Peer Learning — SharedLogbook

You are not alone in the GYM. Other agents train alongside you and share their experiences through the **SharedLogbook**.

### 9.1 What You Can Learn from Peers

- **Improvement Suggestions**: "Agent X scores 85% in psychiatry but you score 40% — try that domain."
- **Complementary Strengths**: Who is the best in each domain.
- **Collective Weaknesses**: Domains where ALL agents struggle.
- **Anti-Herding**: Alerts when too many agents train on the same domain.

### 9.2 Mastery Levels

| Level | Score Range | Meaning |
|-------|------------|---------|
| Novice | < 30% | Starting out |
| Beginner | 30-50% | Basic understanding |
| Intermediate | 50-70% | Competent |
| Advanced | 70-85% | Strong performer |
| Expert | 85-95% | Near mastery |
| Master | 95%+ | Domain conquered |

---

## 10. Model Auto-Profiling

When your model enters the GYM, it is automatically profiled:

| Auto-Detected | Description |
|---------------|-------------|
| Architecture | Model type, class, backbone |
| Modalities | Text / Image / Video |
| Memory | Estimated VRAM requirement |
| Compatible Domains | VL → all 10 domains; Text → 8 domains |
| Optimal Params | Batch size, context length, LoRA rank |
| Training Support | SFT, GRPO, LoRA compatibility |

**Auto-Repairs**:
- Missing processor configs (VL models)
- `rope_scaling` format mismatch (cross-version transformers)
- Missing tokenizer files

---

## 11. Configuration Reference

### 11.1 Strategy Personality Weights

Control your agent's training behavior:

```yaml
# Explorer personality — tries many domains
curiosity_weight: 0.30
weakness_weight: 0.20
peer_learning_weight: 0.15
diversity_weight: 0.25
mastery_push_weight: 0.05
safety_weight: 0.05

# Perfectionist personality — focuses on weak spots
curiosity_weight: 0.10
weakness_weight: 0.40
peer_learning_weight: 0.20
diversity_weight: 0.10
mastery_push_weight: 0.15
safety_weight: 0.05

# Safety-first personality
curiosity_weight: 0.10
weakness_weight: 0.30
peer_learning_weight: 0.15
diversity_weight: 0.10
mastery_push_weight: 0.10
safety_weight: 0.25
```

### 11.2 GYM-Level Configuration

```yaml
gym:
  max_gpus: 8
  cycle_timeout_minutes: 60
  max_concurrent_workers: 8
  status_interval_seconds: 60
  wandb_project: "pt2-minstar-gym-rl"
```

---

## 12. W&B Monitoring

All training is logged to Weights & Biases under project `pt2-minstar-gym-rl`.

**What's Logged**:
- Per-step: task rewards, tool usage, turn counts
- Per-epoch: mean reward, trajectory counts, loss
- Per-cycle: domain selection, pre/post scores, improvement
- Benchmarks: all 21 benchmark results per cycle

---

## Appendix A: Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `premature_stop` on all tasks | Model doesn't use tools | Train with SARL strategy, ensure system prompt includes tool instructions |
| `eval_crash` on EHR domain | Pydantic validation error | Check `data_model.py` Literal types match `db.json` values |
| Score stuck at 0.0 | Pure action_score (tool matching) | System uses blended score (30% tool + 70% composite) |
| Model not loading on GPU | Custom architecture | Ensure `trust_remote_code=True` and model files include `*.py` |
| `premature_stop` + low reward | Model generates text, not JSON | Enhanced parser supports 6 formats (see Section 5.3) |

---

## Appendix B: File Structure

```
BIOAgents/
├── configs/
│   └── autonomous_gym.yaml          # GYM + agent configuration
├── data/domains/
│   ├── {domain}/
│   │   ├── tasks.json                # Original training tasks
│   │   ├── tasks_scaled.json         # Template-scaled tasks
│   │   ├── tasks_auto_generated.json # Knowledge-mined tasks
│   │   └── db.json                   # Domain knowledge database
├── bioagents/
│   ├── gym/
│   │   ├── autonomous_agent.py       # Agent logic + data strategy
│   │   ├── autonomous_gym.py         # GYM orchestrator
│   │   ├── agent_env.py              # GYM environment
│   │   ├── model_profile.py          # Auto-profiling
│   │   └── shared_logbook.py         # Peer learning
│   ├── evaluation/
│   │   ├── agent_runner.py           # Task execution + scoring
│   │   ├── grpo_rewards.py           # 5D reward system
│   │   └── benchmark_eval.py         # External benchmarks
│   ├── training/
│   │   └── grpo_trainer.py           # GRPO training loop
│   ├── data_pipeline/
│   │   └── auto_task_generator.py    # Knowledge mining
│   └── domains/
│       └── {domain}/
│           ├── tools.py              # Domain tools
│           └── environment.py        # Domain environment
└── checkpoints/models/               # Model checkpoints
```


## 13. Live Intelligence (Auto-Updated)

> Last auto-update: 2026-02-14 14:51 | Based on 3 recent cycles

### 13.1 Current Score Baselines

| Domain | Avg Score | Best Agent | Common Errors |
|--------|-----------|------------|---------------|
| clinical_diagnosis | 52.0% | qwen2_5_vl_7b (52.0%) | reasoning_error(1), premature_stop(1) |
| drug_interaction | 33.0% | lingshu_7b (33.0%) | tool_use_failure(2), premature_stop(1) |
| ehr_management | 39.0% | step3_vl_10b (39.0%) | over_investigation(1) |

### 13.2 Discovered Best Practices

**What works (from successful training cycles):**

- **qwen2_5_vl_7b** improved **17.0%** on `clinical_diagnosis` (tasks: 20)

### 13.3 Known Pitfalls (from real agent experience)

- **`premature_stop`** (2 occurrences): Most common in `clinical_diagnosis` (1x)
- **`tool_use_failure`** (2 occurrences): Most common in `drug_interaction` (2x)
- **`reasoning_error`** (1 occurrences): Most common in `clinical_diagnosis` (1x)
- **`over_investigation`** (1 occurrences): Most common in `ehr_management` (1x)

### 13.5 Recommended Focus Areas

- `drug_interaction` (avg: 33.0%) — needs more training
- `ehr_management` (avg: 39.0%) — needs more training
- `clinical_diagnosis` (avg: 52.0%) — needs more training


---

*This guideline is maintained by the Healthcare AI GYM system. It reflects the current state of all domains, tools, benchmarks, and training strategies.*
