# BIOAgents - Medical/Biomedical Agent GYM: 전체 기획 문서

> **작성일**: 2026-02-12  
> **목표**: NeurIPS 2026 제출 (2026년 6월)  
> **가용자원**: A100 8대  

---

## 1. 프로젝트 개요 (Project Overview)

### 1.1 핵심 아이디어
Medical & Biomedical 도메인에 특화된 **Agent GYM** 프레임워크를 구축한다.  
AgentGym-RL과 τ²-bench의 아키텍처를 참고하되, **의료 도메인 특화 tool database**, **task 시나리오**, **Gymnasium-compatible 환경**을 새롭게 설계하여, multi-turn RL(GRPO/PPO)을 통해 medical agent를 학습시키는 end-to-end 시스템을 만든다.

### 1.2 기획 의도 (README.md 원문 기반)
1. medical & biomedical 관련 benchmark resource를 한 곳에 모아서 평가 가능하도록 한다.
2. τ²-bench처럼 medical domain 특화된 tool 기반 database와 task를 만들고 tool 실행이 simulation 기반으로 가능하게 한다.
3. agent가 gym 형태로 구성되어 medical 상황극에서의 모든 trajectory를 기록한다.
4. 스스로 기록한 방식을 보고 어떠한 trajectory가 있어야 realistic한지 판단한다.
5. realistic한 scenario 기반으로 스스로 학습(RL - GRPO)을 시키며 exploration & exploitation을 진행한다.
6. 특정 step마다 학습한 agent는 visual medical QA, text medical QA 등 평가를 해보며 본인의 상태를 파악한다.
7. 위와같은 과정을 지속, 반복하여 모든것을 기록한다.

### 1.3 핵심 차별점 (vs. 기존 연구) — 2025 경쟁 분석 반영

| 기존 연구 | 핵심 특징 | 한계 | Healthcare AI GYM 차별점 |
|---|---|---|---|
| **MedAgentBench** (Stanford, 2025) arXiv:2501.14654 | FHIR-compliant EHR, 100 patients, 700K data | Benchmark only (no RL training), single EHR domain | 8 domains + RL pipeline + multi-domain pathways |
| **AgentClinic** (2024) arXiv:2405.07960 | Patient agent, 24 cognitive biases, 9 specialties | No RL training, no cross-domain, no safety eval | Patient Agent + RL + cross-domain + safety evaluation |
| **Agent Hospital** (2024) arXiv:2405.02957 | Full hospital simulation, MedAgent-Zero, 10K patients | Chinese medical records, no tool-use, no safety | 88 clinical tools + safety rewards + English |
| **DoctorAgent-RL** (2025) arXiv:2505.19630 | Multi-agent RL, progressive symptoms | Dialogue-only, single domain | 8 domains, 88 tools, structured evaluation |
| **MedAgentSim** (MICCAI 2025) | Doctor+Patient+Measurement agents | Limited domains, no RL reward design | 5D reward + guideline compliance + adversarial testing |
| **CARES** (2025) Safety Benchmark | 18K adversarial prompts, AMA principles | Safety eval only, no training | 50 adversarial + 11 bias tests + RL safety reward |
| **MedSafetyBench** (2024) | AMA ethics-based safety benchmark | No agent evaluation, static only | Integrated safety in agent training loop |
| **LA-CDM** (2025) arXiv:2506.13474 | Hybrid SL+RL, MIMIC-CDM | Single domain (abdominal), no tools | Multi-domain, 88 tools, cross-domain pathways |
| **AgentGym-RL** (arXiv 2025) | 일반 환경 multi-environment RL | 의료 도메인 없음 | 의료 특화 환경/도구/시나리오 |
| τ²-bench | Tool-augmented agent benchmark | airline/retail/telecom만 | Medical domain with 88 clinical tools |

**Healthcare AI GYM의 고유 기여 (Unique Contributions):**
1. **Cross-Domain Clinical Pathways** — 6개 실제 임상 경로 (다른 어떤 논문에도 없음)
2. **5D Reward System** — Accuracy + Format + Process + Safety + Coherence
3. **Integrated Safety RL** — Safety reward가 RL 학습에 직접 통합
4. **Patient Agent + Tool-Use** — 동적 환자 상호작용 + 88개 임상 도구 동시 지원
5. **Clinical Guidelines Compliance** — 10개 가이드라인 자동 준수 평가
6. **Pure RL Training (No SFT)** — Pre-trained models learn directly via Multi-Turn GRPO with 5D adaptive rewards. Benchmark-guided reward weights dynamically adjust based on external evaluation results, enabling self-correcting RL without supervised fine-tuning.
7. **FairGRPO** — 인구통계학적 공정성 인식 RL 학습 (demographic-aware reward weighting)

### 1.4 심층 경쟁자 분석: DiagGym vs MedAgentGym vs Healthcare AI GYM

#### DiagGym (arXiv:2510.24654, Oct 2025)
- **핵심**: EHR 기반 world model로 진단 환경 시뮬레이션 + DiagAgent (LLM 기반 진단 에이전트)
- **특징**: DiagBench (2.2K physician-validated cases), multi-turn RL로 진단 정책 학습
- **성능**: 진단 정확도 +11.2%, 검사 추천 F1 +17.6% (vs SOTA LLM)
- **한계**:
  - 단일 도메인 (EHR 진단만) — cross-domain 경로 없음
  - Tool-use 프레임워크 없음 (검사 선택만, 일반 도구 호출 아님)
  - Safety 평가 프레임워크 명시적으로 없음
  - Patient Agent 시뮬레이션 없음 (데이터 기반만)

#### MedAgentGym (ICLR 2026 Oral, June 2025)
- **핵심**: 코드 중심 의생명 데이터 분석 에이전트 학습 환경 (Python 실행 기반)
- **특징**: 72,413 task instances / 129 categories / 12 real-world 시나리오
- **성능**: Med-Copilot offline RL +43%, online RL +45% (GPT-4o 수준)
- **한계**:
  - 코드 실행 기반 — 임상 의사결정이 아닌 데이터 분석 중심
  - Text-only (멀티모달 미지원)
  - 환자 상호작용 시뮬레이션 없음
  - Cross-domain 임상 경로 없음
  - PhysioNet 자격 요구 (일부 데이터)

#### Healthcare AI GYM (Ours) — 차별화 포인트

| 차원 | DiagGym | MedAgentGym | **Healthcare AI GYM** |
|------|---------|-------------|----------------------|
| **도메인 수** | 1 (EHR 진단) | 129 (데이터분석) | **10 임상 도메인** |
| **태스크 유형** | 진단/검사 선택 | 코드 실행 | **Tool-use 에이전트** |
| **태스크 수** | 2.2K | 72.4K | **550+ (확장 가능)** |
| **멀티모달** | ✗ | ✗ | **✓ (Text + Vision)** |
| **Cross-domain** | ✗ | ✗ | **✓ (6 clinical pathways)** |
| **Patient Agent** | ✗ | ✗ | **✓ (12 personalities, 13 biases)** |
| **Safety 평가** | 제한적 | ✗ | **✓ (5D reward + 50 adversarial + 11 bias tests)** |
| **공정성 (Fairness)** | ✗ | ✗ | **✓ (FairGRPO)** |
| **Tool 수** | ~10 검사 | Python runtime | **88+ 임상 도구** |
| **RL 방법** | Multi-turn RL | Offline/Online RL | **GRPO + Self-Play + GymCoach** |
| **자율 학습** | ✗ | ✗ | **✓ (GymCoach autonomous loop)** |
| **가이드라인 준수** | ✗ | ✗ | **✓ (10 guidelines)** |

**핵심 논쟁 포인트 (Rebuttal 준비):**
1. "MedAgentGym이 72K 태스크인데 Healthcare AI GYM은 550개뿐?" 
   → 우리는 tool-use 에이전트 환경이므로 태스크 하나가 multi-turn interaction + tool calling + reasoning으로 구성. LLM 기반 자동 태스크 생성(GymCoach)으로 무한 확장 가능.
2. "DiagGym이 physician-validated인데?" 
   → 우리도 evaluation_criteria에 nl_assertions (physician-level rubric)을 포함하며, 추가로 safety violation taxonomy까지 구현.
3. "스케일이 부족?" 
   → 우리의 기여는 스케일이 아닌 **통합 시스템** — 10 도메인 × 88 도구 × patient agent × safety × fairness × autonomous training. 어떤 단일 논문도 이 범위를 커버하지 못함.

### 1.5 FairGRPO 메커니즘 (arXiv:2510.19893)

**구현 완료 (2026-02-13):**

1. **Demographic Group Extraction** (`grpo_rewards.py`)
   - 환자 데이터에서 age_group / sex / ethnicity 자동 추출
   - 라벨 없을 때 unsupervised clustering으로 발견 가능 (추후 구현)

2. **FairnessTracker** (`grpo_rewards.py`)
   - 인구통계 그룹별 보상 통계 실시간 추적
   - Representation weight: 소수 그룹 상향 가중 (빈도 역수)
   - Performance weight: 저성과 그룹 상향 가중 (평균 역수)
   - Fairness gap 모니터링: max-min 그룹 간 격차 추적

3. **FairGRPO Reward Functions** (`grpo_rewards.py`)
   - `grpo_fairness_reward`: 기본 보상에 공정성 가중치 적용
   - `grpo_fair_composite_reward`: composite reward + fairness signal 통합

4. **FairGRPO Trainer** (`grpo_trainer.py`)
   - `FairGRPOConfig`: 공정성 파라미터 (weight, alpha_repr, alpha_perf, max_gap)
   - `train_fair_grpo()`: TRL GRPOTrainer 기반 공정성 인식 학습
   - 학습 완료 후 fairness_report.json 자동 저장

5. **GymCoach 통합** (`gym_coach.py`)
   - `_train_fair_grpo()`: 자율 학습 루프에서 FairGRPO 자동 활용
   - Training Memory에 fairness 결과 기록

---

## 2. 현재 리소스 현황 (Resource Inventory)

### 2.1 디렉토리 구조
```
BIOAgents/
├── README.md                    # 기획 의도 & 리소스 정리
├── PLANNING.md                  # 본 기획 문서
├── databases/                   # Tool DB & Knowledge Base
│   ├── critic/                  # Self-BioRAG critic 데이터 (8개 JSON)
│   ├── generator/               # Self-BioRAG generator 데이터
│   ├── instruction/             # 의료 instruction 데이터 (4개 JSON)
│   │   ├── all_biomedical_instruction.json
│   │   ├── MedInstruct-52k.json
│   │   ├── mol_instruction_qa.json
│   │   └── self_instruct_biomedical.json
│   ├── retriever/               # MedCPT top-10 evidence
│   ├── tau2-bench/              # τ²-bench 전체 코드 (참고용 도메인 구조)
│   ├── wiki2018_en/             # Wikipedia 2018 dump
│   └── wiki2026_en/             # Wikipedia 2026 dump
├── datasets/                    # (비어있음 - 학습/평가 데이터 큐레이션 예정)
├── evaluations/                 # 평가 벤치마크 코드
│   ├── mimic-code/              # MIMIC-III/IV EHR 코드 (benchmarks, SQL concepts)
│   ├── OLAPH/                   # Long-form Medical QA 평가 (MedLFQA)
│   ├── PathVQA/                 # PathVQA 베이스라인 & 평가
│   ├── PMC-VQA/                 # PMC-VQA + Slake1.0
│   ├── quilt-llava/             # Quilt-VQA (histopathology VQA)
│   ├── self-biorag/             # Self-BioRAG (MedQA, MedMCQA, MMLU 포함)
│   │   └── data/benchmark/      # med_qa, medmc_qa, mmlu (test/train .jsonl)
│   └── VQA-Med-2021/            # VQA-Med 2021 테스트셋
├── GYM_reference/               # GYM 구조 참고 코드
│   └── AgentGym-RL/             # AgentGym-RL 전체 (verl 기반 RL trainer)
│       ├── AgentGym/            # 원본 AgentGym (빈 디렉토리, 참고용)
│       ├── AgentGym-RL/         # verl 기반 agent trainer
│       │   └── verl/agent_trainer/  # PPO/GRPO trainer, 환경 설정
│       └── examples/train/      # 학습 스크립트 예시 (searchqa, webarena 등)
├── references/                  # 참고 논문 & 코드
│   ├── medical_agent/           # 의료 agent 관련 논문 4편
│   │   ├── 2024.findings-emnlp.510.pdf
│   │   ├── 2404.15155v3.pdf
│   │   ├── 2411.00248v2.pdf
│   │   └── 2505.16100v1.pdf
│   └── medical_qa/              # 의료 QA 관련 논문 & 코드
│       ├── grpo_vqa_Qwen3_token_shaping.py   # MRPO VQA 학습 코드
│       ├── run_grpo_MRPO_Qwen3.sh            # 실행 스크립트
│       ├── MRPO_ICML_submission.pdf           # MRPO 논문
│       ├── 2509.08755v1.pdf                   # AgentGym-RL 논문
│       └── ... (총 14개 파일)
├── tool_simulations/            # Tool Simulation 엔진
│   └── tool-dataset-generation/ # Tool 데이터셋 생성 파이프라인
│       ├── runner.py            # 메인 실행기
│       ├── generation.py        # 생성 로직
│       ├── utils/
│       │   ├── tool_generation/     # tool spec 자동 생성
│       │   ├── tool_simulation/     # tool 실행 시뮬레이션 (LLM 기반)
│       │   ├── task_generation/     # task 자동 생성
│       │   ├── user_simulation/     # user 시뮬레이션
│       │   ├── q_generation/        # question 생성
│       │   ├── response_generation/ # response 생성
│       │   └── validation/          # 검증
│       └── models/              # 모델 인터페이스 (OpenAI, Qwen, GLM 등)
└── trains/                      # 학습 프레임워크
    ├── oumi/                    # Oumi SFT 프레임워크
    │   ├── configs/             # 학습 설정 파일들
    │   ├── src/oumi/            # 코어 학습 코드
    │   └── scripts/             # 유틸리티 스크립트
    └── snapshot-po/             # Snapshot-PO RL 학습 프레임워크
        ├── configs/             # SARL 설정 파일들
        ├── run.py               # 메인 학습 실행기
        ├── reward_computation/  # 보상 함수 계산
        ├── generation/          # 생성 로직
        └── torchtitan_rl/       # TorchTitan RL 백엔드
```

### 2.2 보유 데이터셋 상세

#### Visual Medical QA (6개 소스)
| # | 데이터셋 | 소스 | 특징 | 상태 |
|---|---|---|---|---|
| 1 | VQA-RAD | HuggingFace (flaviagiammarino/vqa-rad) | 방사선학 VQA | 다운로드 필요 |
| 2 | SLAKE | HuggingFace (BoKelvin/SLAKE) + evaluations/PMC-VQA/Slake1.0 | 다국어 의료 VQA | 로컬 보유 |
| 3 | PathVQA | HuggingFace (flaviagiammarino/path-vqa) + evaluations/PathVQA | 병리학 VQA | 로컬 보유 |
| 4 | PMC-VQA | HuggingFace (RadGenome/PMC-VQA) + evaluations/PMC-VQA | 의학 논문 이미지 VQA | 로컬 보유 |
| 5 | VQA-Med-2021 | evaluations/VQA-Med-2021 | 의료 VQA 챌린지 | 로컬 보유 (zip) |
| 6 | Quilt-VQA | HuggingFace (wisdomik/Quilt_VQA) + evaluations/quilt-llava | 조직병리학 VQA | 로컬 보유 |

#### Text Medical QA (3개 소스)
| # | 데이터셋 | 소스 | 특징 | 상태 |
|---|---|---|---|---|
| 1 | MedLFQA | HuggingFace (dmis-lab/MedLFQA) + evaluations/OLAPH | Long-form 의료 QA | 로컬 보유 |
| 2 | MedQA/MedMCQA/MMLU | evaluations/self-biorag/data/benchmark/ | 객관식 의료 시험 문제 | 로컬 보유 |
| 3 | Biomedical Instructions | databases/instruction/ | SFT용 instruction 데이터 (52k+) | 로컬 보유 |

#### EHR Record (1개 소스)
| # | 데이터셋 | 소스 | 특징 | 상태 |
|---|---|---|---|---|
| 1 | MIMIC-III/IV | evaluations/mimic-code | EHR 코드, SQL concepts, 벤치마크 | 코드 보유 (데이터는 별도 접근 필요) |

#### Knowledge Base
| # | 리소스 | 경로 | 용도 |
|---|---|---|---|
| 1 | Wikipedia 2018 dump | databases/wiki2018_en/ | 검색 시뮬레이션용 |
| 2 | Wikipedia 2026 dump | databases/wiki2026_en/ | 검색 시뮬레이션용 |
| 3 | MedCPT evidence | databases/retriever/ | top-10 의료 근거 검색 |
| 4 | Critic 데이터 | databases/critic/ | relevance/utility/groundness 평가 |
| 5 | Generator 데이터 | databases/generator/ | retrieval token 기반 생성 |

### 2.3 모델 후보군
| # | 모델 | 크기 | 특징 | 용도 |
|---|---|---|---|---|
| 1 | Lingshu-7B | 7B | 의료 MLLM, multi-modality | 주 학습 대상 후보 |
| 2 | Qwen2.5-VL-7B-Instruct | 7B | 범용 VLM, tool-use 지원 | 주 학습 대상 후보 |
| 3 | Step3-VL-10B | 10B | VLM, 고성능 | 비교 실험용 |

### 2.4 학습 프레임워크 현황
| 프레임워크 | 경로 | 용도 | 비고 |
|---|---|---|---|
| Oumi | trains/oumi/ | SFT (Supervised Fine-Tuning) | 이미 agent SFT config 존재 |
| Snapshot-PO | trains/snapshot-po/ | SARL (Search Agent RL) | GRPO 기반, 실행 로그 존재 (260209~260212) |
| AgentGym-RL | GYM_reference/AgentGym-RL/ | Multi-turn RL (PPO/GRPO) | verl 기반, 환경 서버 아키텍처 |
| MRPO (참고) | references/medical_qa/ | VQA GRPO with token shaping | ICML 제출 코드, BERTScore/ROUGE reward |

### 2.5 참고 시스템 아키텍처 분석

#### τ²-bench 도메인 구조 (databases/tau2-bench)
```
도메인 1개 구성 요소:
├── src/tau2/domains/{domain}/
│   ├── data_model.py    # DB 스키마 (Pydantic BaseModel)
│   ├── tools.py         # ToolKitBase 상속, @is_tool 데코레이터
│   ├── environment.py   # get_environment(), get_tasks() 함수
│   └── utils.py         # 경로 설정 등
├── data/tau2/domains/{domain}/
│   ├── db.json          # 시뮬레이션용 데이터베이스
│   ├── policy.md        # 에이전트 행동 정책
│   ├── tasks.json       # 평가용 task 시나리오
│   └── split_tasks.json # train/test 분리
└── Gymnasium-compatible gym interface (gym_agent.py)
    ├── AgentGymEnv  - reset() → observation, step(action) → obs, reward, done
    └── UserGymEnv   - 사용자 역할 플레이
```

#### AgentGym-RL 아키텍처 (GYM_reference/AgentGym-RL)
```
3개 모듈:
1. Environment Module: HTTP 서버 기반 환경, 병렬 요청 지원
2. Agent Module: 추론/의사결정, 장기 계획, self-reflection
3. Training Module: verl 기반 PPO/GRPO/RLOO/REINFORCE++
   - RolloutHandler: attention mask, loss mask, position ids 처리
   - EnvClient: observation(), available_actions(), step(), reset()
   - RoundScheduler: fixed / scaling_inter_stepwise (ScalingInter-RL)
```

#### Tool Simulation 파이프라인 (tool_simulations/tool-dataset-generation)
```
파이프라인 단계:
1. tool_generation/   → 질문에서 tool spec 자동 생성 (LLM 기반)
2. task_generation/   → 시나리오/대화 생성 (initial + continual)
3. tool_simulation/   → LLM으로 tool 실행 결과 시뮬레이션
4. user_simulation/   → 사용자 행동 시뮬레이션
5. response_generation/ → 응답 생성
6. validation/        → 품질 검증
```

---

## 3. 기술 설계 (Technical Design)

### 3.1 BIOAgents GYM 아키텍처 (설계안)

```
┌────────────────────────────────────────────────────────┐
│                    BIOAgents GYM                        │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Environment  │  │    Agent     │  │   Training   │ │
│  │    Module     │  │    Module    │  │    Module    │ │
│  │              │  │              │  │              │ │
│  │ ·Medical     │  │ ·Reasoning   │  │ ·SFT (Oumi) │ │
│  │  Domains     │  │ ·Tool Use    │  │ ·GRPO       │ │
│  │ ·Tool DB     │  │ ·Planning    │  │ ·PPO        │ │
│  │ ·Simulation  │  │ ·Reflection  │  │ ·ScalingRL  │ │
│  │ ·EHR System  │  │ ·Multi-modal │  │ ·Logging    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Evaluation Suite                     │  │
│  │  Text QA │ Visual QA │ EHR Tasks │ Agent Tasks   │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 3.2 Medical Domain 설계 (τ²-bench 스타일)

**도메인 목록 (계획):**

| # | Domain | 설명 | Tools | Tasks (orig+scaled) | 상태 |
|---|---|---|---|---|---|
| 1 | `clinical_diagnosis` | 환자 증상 → 진단 | 20 | 5 + 60 scaled | 52/13 | ✅ |
| 2 | `medical_qa` | 의료 질문 응답 | 10 | 50 + 200 scaled | 35/15 | ✅ |
| 3 | `visual_diagnosis` | 의료 이미지 분석 | 11 | 8 + 31 scaled | 31/8 | ✅ |
| 4 | `drug_interaction` | 약물 상호작용 검증 | 10 | 5 + 60 scaled | 52/13 | ✅ |
| 5 | `ehr_management` | EHR 조회/분석 | 14 | 15 + 60 scaled | 58/17 | ✅ |
| 6 | `triage_emergency` | 응급실 트리아지 | 12 | 20 | 14/6 | ✅ 확장 |
| 7 | `radiology_report` | 영상 판독문 생성 | 11 | 20 | 14/6 | ✅ 확장 |
| 8 | `psychiatry` | 정신건강 평가 & 치료 | 14 | 20 | 13/7 | ✅ **NEW** |
| 9 | `obstetrics` | 산과 진료 & 분만 관리 | 14 | 20 | 13/7 | ✅ **NEW** |
| 10 | `cross_domain` | 다단계 임상 경로 (6 pathways) | multi | 25 | 17/8 | ✅ |
| | **총계** | | **126+ unique** | **~600** | 299/107 | |

### 3.3 Medical Tool Database 설계 (상세)

```python
# 계획된 Tool 카테고리 (총 ~25개 tool)

# Category 1: Medical Knowledge Search
- search_pubmed(queries: list[str]) → list[{title, abstract, pmid, url}]
- browse_article(pmid: str, query: str) → str
- search_medical_wiki(queries: list[str]) → list[{title, url, snippet}]
- browse_medical_wiki(url: str, query: str) → str
- search_clinical_guidelines(condition: str) → list[{guideline, source}]

# Category 2: Patient Record (EHR) Tools
- get_patient_info(patient_id: str) → {demographics, conditions, allergies}
- get_lab_results(patient_id: str, test_type: str) → list[{test, value, unit, date}]
- get_medication_list(patient_id: str) → list[{drug, dose, frequency, start_date}]
- get_vital_signs(patient_id: str) → {bp, hr, temp, spo2, rr}
- get_clinical_notes(patient_id: str, note_type: str) → list[{date, content}]

# Category 3: Diagnostic Tools
- check_drug_interaction(drug_a: str, drug_b: str) → {severity, description}
- calculate_clinical_score(score_type: str, params: dict) → {score, interpretation}
- get_differential_diagnosis(symptoms: list[str]) → list[{condition, probability}]
- order_lab_test(patient_id: str, test_type: str) → {order_id, status}

# Category 4: Medical Image Analysis (시뮬레이션)
- analyze_medical_image(image_path: str, modality: str) → {findings, confidence}
- get_image_report(image_id: str) → {report, impression}
- compare_with_prior(current_id: str, prior_id: str) → {changes, assessment}

# Category 5: Communication & Workflow
- transfer_to_specialist(summary: str, specialty: str) → str
- schedule_followup(patient_id: str, reason: str) → {appointment_id, date}
- send_patient_message(patient_id: str, message: str) → str
```

### 3.4 Reward Function 설계 (✅ 구현 완료)

**Core Rewards** (`bioagents/evaluation/rewards.py`):
```python
# 1. Accuracy Reward
accuracy_reward_exact_match()  # MC 정답 exact match
accuracy_reward_soft()         # ROUGE-1 proxy (token overlap F1)
accuracy_reward_bertscore()    # BiomedBERT 기반 semantic similarity

# 2. Format Reward
format_reward_tool_call()      # JSON tool call 형식 검증
format_reward_think_answer()   # <think>/<answer> 태그 검증
format_reward_composite()      # 턴 컨텍스트 적응형 (intermediate vs final)

# 3. Process Reward
process_reward_tool_usage()    # 기대 tool vs 실제 tool (coverage 70% + efficiency 30%)
process_reward_reasoning_quality()  # 의료 용어, 구조적 추론, 길이 휴리스틱

# 4. Composite Reward
compute_composite_reward()     # 가중 합산 (default: acc=0.4, fmt=0.2, proc=0.4)
# + Registry: get_reward_function(), register_reward_function()
```

**GRPO-Compatible Wrappers** (`bioagents/evaluation/grpo_rewards.py`):
```python
# TRL GRPOTrainer 호환 시그니처: fn(completions, **kwargs) -> list[float]
grpo_accuracy_reward()   # MC exact match + open-ended (ROUGE/BLEU/BERTScore)
grpo_format_reward()     # tool call & answer tag 검증
grpo_process_reward()    # reasoning quality 휴리스틱
grpo_tool_use_reward()   # 기대 vs 실제 tool calls
grpo_composite_reward()  # 가중 합산
get_grpo_reward_functions(["accuracy", "format", "process"])  # Registry
```

**테스트**: 61 tests in `tests/test_rewards.py` (all passing)

---

## 4. 타임라인 & 로드맵

### Phase 1: 기반 구축 (2026.02.12 ~ 2026.03.15) [4주]

| 주차 | 작업 | 산출물 | 상태 |
|---|---|---|---|
| W1 (02/12~02/18) | 프로젝트 구조 설계 & 기획 문서 | PLANNING.md, 디렉토리 구조 | ✅ 완료 |
| W1 | Medical Tool Database 스키마 설계 | tools.py, data_model.py 초안 | ⬜ 대기 |
| W2 (02/19~02/25) | Medical Domain 환경 구현 (clinical_diagnosis) | environment.py, policy.md | ⬜ 대기 |
| W2 | Tool Simulation 엔진 의료 도메인 적용 | tool simulation prompts | ⬜ 대기 |
| W3 (02/26~03/04) | Task 시나리오 생성 (50+ tasks) | tasks.json, db.json | ⬜ 대기 |
| W3 | GYM 인터페이스 구현 (Gymnasium-compatible) | gym_agent.py | ⬜ 대기 |
| W4 (03/05~03/15) | 데이터셋 전처리 파이프라인 | datasets/ 구성 | ⬜ 대기 |
| W4 | 기본 평가 파이프라인 구축 | eval scripts | ⬜ 대기 |

### Phase 2: 학습 파이프라인 (2026.03.15 ~ 2026.04.15) [4주]

| 주차 | 작업 | 산출물 | 상태 |
|---|---|---|---|
| W5 (03/15~03/22) | SFT 데이터 구성 (instruction + tool-use) | SFT jsonl 데이터 | ⬜ 대기 |
| W5 | 모델 선정 & baseline 평가 | baseline 결과 로그 | ⬜ 대기 |
| W6 (03/22~03/29) | SFT 학습 (Oumi) | SFT 체크포인트 | ⬜ 대기 |
| W6 | additional Medical Domain 구현 (medical_qa, visual_diagnosis) | 추가 도메인 코드 | ⬜ 대기 |
| W7 (03/29~04/05) | RL 학습 시작 (GRPO, GYM 환경 연동) | RL 체크포인트 | ⬜ 대기 |
| W7 | Trajectory 로깅 시스템 구축 | trajectory 파일들 | ⬜ 대기 |
| W8 (04/05~04/15) | ScalingInter-RL 적용 실험 | 학습 곡선, 비교 결과 | ⬜ 대기 |
| W8 | 중간 평가 (Text QA + Visual QA) | 중간 결과 리포트 | ⬜ 대기 |

### Phase 3: 반복 개선 (2026.04.15 ~ 2026.05.15) [4주]

| 주차 | 작업 | 산출물 | 상태 |
|---|---|---|---|
| W9 | EHR Domain 구현 & 학습 | EHR 도메인 코드 | ⬜ 대기 |
| W10 | Reward function 개선 & 실험 | ablation 결과 | ⬜ 대기 |
| W11 | Multi-domain 통합 학습 | 통합 체크포인트 | ⬜ 대기 |
| W12 | 전체 벤치마크 평가 | 최종 결과 테이블 | ⬜ 대기 |

### Phase 4: 논문 작성 (2026.05.15 ~ 2026.06.01) [2주]

| 주차 | 작업 | 산출물 | 상태 |
|---|---|---|---|
| W13 | 논문 초안 작성 | paper draft | ⬜ 대기 |
| W14 | 추가 실험 + 논문 완성 | final paper | ⬜ 대기 |

---

## 5. Related Work 분석

### 5.1 가장 관련된 기존 연구

#### (1) AgentClinic (EMNLP 2024 Findings)
- **논문**: `references/medical_agent/2024.findings-emnlp.510.pdf`
- **핵심**: 환자/의사/측정/조정 에이전트로 구성된 임상 시뮬레이션
- **한계**: 진단 시나리오에 한정, tool 다양성 부족, RL 학습 없음
- **참고점**: 다중 역할 에이전트 구조, 편향 시뮬레이션

#### (2) AgentGym-RL (arXiv:2509.08755)
- **논문**: `references/medical_qa/2509.08755v1.pdf`
- **핵심**: verl 기반 multi-turn RL, ScalingInter-RL 알고리즘
- **한계**: 의료 도메인 환경 없음
- **참고점**: 아키텍처, 학습 파이프라인, RoundScheduler

#### (3) τ²-bench (arXiv:2506.07982)
- **코드**: `databases/tau2-bench/`
- **핵심**: 도메인별 tool+DB+policy+task 구조, Gymnasium 인터페이스
- **한계**: airline/retail/telecom만
- **참고점**: 도메인 구조 패턴, 평가 체계

#### (4) Self-BioRAG (arXiv:2305.10415)
- **코드**: `evaluations/self-biorag/`
- **핵심**: 의료 RAG, retrieval critic, MedQA/MMLU 평가
- **참고점**: 의료 지식 검색 구조, 벤치마크 데이터

#### (5) MRPO (ICML submission)
- **코드**: `references/medical_qa/grpo_vqa_Qwen3_token_shaping.py`
- **핵심**: Medical VQA에 GRPO + token shaping 적용
- **참고점**: process reward (Alignment/Relevance/Factuality), BERTScore reward

#### (6) Lingshu
- **URL**: https://huggingface.co/lingshu-medical-mllm/Lingshu-7B
- **핵심**: 의료 특화 MLLM, MedEvalKit
- **참고점**: 모델 후보, 평가 프레임워크

### 5.2 추가 참고 논문 (references/ 내)
| 파일명 | 추정 내용 |
|---|---|
| `medical_agent/2404.15155v3.pdf` | 의료 agent 관련 (2024) |
| `medical_agent/2411.00248v2.pdf` | 의료 agent 관련 (2024) |
| `medical_agent/2505.16100v1.pdf` | 의료 agent 관련 (2025) |
| `medical_qa/2003.10286v1.pdf` | 의료 QA 관련 (2020) |
| `medical_qa/2009.13081v1.pdf` | 의료 QA 관련 (2020) |
| `medical_qa/2309.11080v1.pdf` | 의료 QA 관련 (2023) |
| `medical_qa/2405.12701v3.pdf` | 의료 QA 관련 (2024) |
| `medical_qa/2506.09513v3.pdf` | 의료 QA 관련 (2025) |
| `medical_qa/2508.19096v1.pdf` | 의료 QA 관련 (2025) |
| `medical_qa/sdata2018251.pdf` | 의료 데이터 관련 |
| `medical_qa/SLAKE.pdf` | SLAKE 데이터셋 논문 |

---

## 6. 핵심 기술 결정 사항

### 6.1 결정된 사항
- [x] GYM 구조: τ²-bench 스타일 도메인 구조 + Gymnasium 인터페이스
- [x] RL 알고리즘: GRPO (주), PPO (비교), ScalingInter-RL (실험)
- [x] Tool Simulation: LLM 기반 시뮬레이션 (tool-dataset-generation 참고)
- [x] 평가 벤치마크: MedQA, MedMCQA, MMLU, VQA-RAD, SLAKE, PathVQA, PMC-VQA
- [x] 가용 자원: A100 8대

### 6.2 결정 사항 (2026-02-12 확정)
- [x] **주 모델 선택**: Lingshu-7B (의료 특화) 우선 → 이후 Qwen2.5-VL-7B로 확장
- [ ] **EHR 데이터 접근**: MIMIC-III/IV 데이터 실제 접근 가능 여부 (확인 필요)
- [x] **도메인 우선순위**: clinical_diagnosis → medical_qa → visual_diagnosis → drug_interaction → ehr_management → triage_emergency → radiology_report → cross_domain
- [x] **도메인 구현 현황**: 8개 도메인 + cross_domain 전체 완료 ✅
- [ ] **Tool Simulation vs Real API**: 어디까지 시뮬레이션, 어디부터 실제 API?
- [x] **논문 포지셔닝**: Framework paper (BIOAgents GYM 자체가 contribution)
- [x] **논문 작성**: 사용자가 직접 작성, AI는 모든 실험/구현/분석 수행 및 기록
- [x] **VQA 평가**: 6개 VQA 벤치마크 통합 (VQA-RAD, SLAKE, PathVQA, PMC-VQA, VQA-Med-2021, Quilt-VQA)
- [x] **Task Scalability**: Scaled tasks GYM 통합 + LLM-based generation pipeline
- [x] **스키마 통일**: 모든 도메인 evaluation_criteria 표준 포맷 자동 변환
- [x] **Safety Evaluation**: 5개 safety reward + 12 adversarial tests + severity taxonomy
- [x] **Cross-Domain Pathways**: 6개 임상 경로 (25 phase tasks across 5 domains)
- [x] **DB 정합성**: visual_diagnosis + drug_interaction 모든 참조 무결성 해소
- [x] **split_tasks.json**: 모든 8개 도메인 + cross_domain 100% 커버리지

---

## 7. 실험 로그 (Experiment Log)

### [2026-02-12] 프로젝트 시작
- 프로젝트 구조 분석 완료
- 기획 문서 초안 작성
- 보유 리소스 전수 조사 완료
- Related work 서베이 시작

### [2026-02-12] Phase 1: GYM 환경 구축 완료
- **작업 내용**:
  1. **Dataset Pipeline 강화**: MedQA(1,273) + MedMCQA(4,183) + MMLU(1,089) = 6,545문제를 자동 변환하는 파이프라인 구축
     - `bioagents/data_pipeline/medqa_loader.py`: JSONL → unified task format 변환기
     - `scripts/generate_gym_data.py`: 벤치마크 데이터 → tasks.json + db.json + split_tasks.json 자동 생성
     - 50 tasks (balanced) + 200 tasks (large) 데이터셋 생성 완료
     - Evidence 데이터 연동: 10,584 articles + 21,810 evidence passages
  2. **Visual Diagnosis 도메인 구축**: 10 images, 8 reports, 10 questions, 8 tasks
     - `bioagents/domains/visual_diagnosis/` — data_model.py, tools.py, environment.py
     - Tools: analyze_medical_image, get_image_report, compare_with_prior, search_similar_cases, search_imaging_knowledge, submit_answer, think 등 9개
     - Tasks: chest X-ray, CT stroke, pathology, dermoscopy, fundus, MRI, breast (easy~hard)
  3. **Drug Interaction 도메인 구축**: 12 drugs, 10 interactions, 4 patient profiles, 5 tasks
     - `bioagents/domains/drug_interaction/` — data_model.py, tools.py, environment.py
     - Tools: get_drug_info, check_interaction, check_all_interactions, get_patient_medications, search_alternatives, check_dosage, search_drugs_by_class, submit_answer, think 등 9개
     - 약물: warfarin, aspirin, fluoxetine, tramadol, metformin, lisinopril, spironolactone, phenytoin, simvastatin, amiodarone, clopidogrel, omeprazole
     - 시나리오: warfarin+aspirin 출혈 위험, serotonin syndrome, 다약제 polypharmacy, clopidogrel+PPI 상호작용, 안전한 조합 확인
  4. **GRPO Training Pipeline**: TRL GRPOTrainer 연동 완료
     - `bioagents/training/grpo_trainer.py`: YAML 설정 → dataset 빌드 → reward function 연결 → TRL GRPOTrainer 실행
     - `configs/grpo_medical_qa.yaml`: Medical QA GRPO 설정 (Qwen3-1.7B + LoRA r=16)
     - `configs/grpo_drug_interaction.yaml`: Drug Interaction GRPO 설정
     - Reward functions: accuracy(0.4) + format(0.2) + process(0.4) composite
     - Dry-run 검증 완료: 35 train tasks, 3 reward functions 정상 동작
  5. **SFT Training Pipeline**: TRL SFTTrainer 연동 완료
     - `bioagents/training/sft_trainer.py`: trajectory-based SFT + direct QA SFT + instruction SFT
     - `configs/sft_medical_qa.yaml`: SFT 설정
     - sft_generator.py 옵션 포맷 호환성 개선 (dict/list 양쪽 지원)
     - Dry-run 검증 완료: 45 train + 5 eval, 7-turn tool-use demonstration
  6. **GYM 통합**: 4개 도메인 Gymnasium 등록 완료
     - clinical_diagnosis (17 tools), medical_qa (8 tools), visual_diagnosis (9 tools), drug_interaction (9 tools)
     - `bioagents/gym/agent_env.py`: 도메인별 초기 관측(observation) 커스터마이징
  7. **통합 테스트**: 4개 테스트 스위트 전체 통과
     - `tests/test_drug_interaction.py`: DB 로딩, 9개 도구 실행, 환경, GYM 인터페이스 (Final reward: 1.0)
     - `tests/test_visual_diagnosis.py`: DB 로딩, 도구 실행, 환경, GYM 인터페이스 (Final reward: 0.667)
     - `tests/test_training_pipeline.py`: GRPO/SFT 설정, 데이터셋, 보상 함수, cross-domain GYM
     - `tests/test_clinical_diagnosis.py`, `tests/test_medical_qa.py`, `tests/test_rewards.py` (기존)
- **결과 요약**:
  - 총 4개 의료 도메인, 43개 도구, 6,545+ 문제 규모의 GYM 환경 구축
  - GRPO/SFT 학습 파이프라인 TRL 연동 완료 (dry-run 검증)
  - 전체 테스트 통과율: 100%
- **다음 단계**:
  - Phase 2: 실제 GRPO 학습 실행 (Qwen3-1.7B → 7B)
  - SFT warmup → GRPO fine-tuning 파이프라인 실행
  - Agent evaluation: 학습된 에이전트 벤치마크 평가
  - EHR Management 도메인 추가 (MIMIC 데이터 접근 확인 후)
- **관련 파일**:
  - `bioagents/domains/drug_interaction/` (data_model, tools, environment)
  - `bioagents/domains/visual_diagnosis/` (data_model, tools, environment)
  - `bioagents/training/grpo_trainer.py`, `bioagents/training/sft_trainer.py`
  - `configs/grpo_medical_qa.yaml`, `configs/grpo_drug_interaction.yaml`, `configs/sft_medical_qa.yaml`
  - `scripts/generate_gym_data.py`
  - `tests/test_drug_interaction.py`, `tests/test_visual_diagnosis.py`, `tests/test_training_pipeline.py`

### [2026-02-12] Phase 2 시작: EHR Management 도메인 구축 완료
- **작업 내용**:
  1. **EHR Management 도메인 구축**: MIMIC-III/IV 스타일 합성 EHR 데이터 기반 5번째 도메인 완성
     - `bioagents/domains/ehr_management/` — data_model.py, tools.py, environment.py, __init__.py
     - **Data Model (MIMIC 호환)**: Demographics, Admission, ICUStay, LabEvent, VitalEvent, MedicationOrder, Procedure, DischargeSummary, ClinicalScore, QualityIndicator → EHRRecord → EHRDB
     - **합성 데이터**: 3명 환자, 4 admissions (1 readmission, 1 active ICU, 1 STEMI post-PCI)
       - P2001 Robert Chen: HFrEF (LVEF 25%), HTN, DM2, CKD3 — 재입원 환자
       - P2002 Maria Santos: Septic shock (E. coli UTI/bacteremia), AKI Stage 3 — 현재 MICU 입원중
       - P2003 James Williams: Acute anterior STEMI, primary PCI with LAD stenting — 퇴원 완료
     - **Tools 14개**: get_patient_summary, get_admission_history, get_lab_results, get_lab_trend, get_vital_signs, detect_vital_alerts, get_medication_orders, get_clinical_scores, get_quality_indicators, get_procedures, get_discharge_summary, lookup_icd_code, think, submit_answer
     - **Tasks 15개** (8 train / 7 test): chart_review, critical_value_identification, medication_reconciliation, readmission_risk, clinical_scoring, discharge_planning, antibiotic_stewardship, quality_measure, icu_assessment, multi_patient_triage, drug_interaction, procedure_interpretation, aki_management, icu_to_floor_transfer, longitudinal_analysis
     - **Clinical Scores**: SOFA, qSOFA, NEWS2, GRACE — 각 점수 components 및 interpretation 포함
     - **Quality Indicators**: readmission_risk, mortality_risk, expected_los, sepsis_flag, aki_stage
  2. **Gymnasium 통합**: `agent_env.py`에 ehr_management 도메인 등록, EHR-specific observation builder 추가
  3. **테스트 스위트**: `tests/test_ehr_management.py` — DB 로딩, 14개 도구 실행, 환경, Task split 필터링, GYM 인터페이스 (5/5 passing)
- **결과 요약**:
  - 총 **5개 의료 도메인**, **57개 도구**, **6,560+ 문제** 규모의 GYM 환경 구축 완료
  - EHR 도메인: MIMIC-IV 스키마 호환, 시간열 lab/vital 데이터, 임상 점수, 품질 지표 포함
  - 테스트 전체 통과: EHR 도메인 5/5, 기존 도메인 정상 유지
- **관련 파일**:
  - `bioagents/domains/ehr_management/` (data_model.py, tools.py, environment.py, __init__.py)
  - `data/domains/ehr_management/` (db.json, policy.md, tasks.json)
  - `tests/test_ehr_management.py`
  - `bioagents/gym/agent_env.py` (ehr_management 등록 추가)

### [2026-02-12] Phase 2: Multi-Model Training and Ablation (8 GPU Parallel)
- **P2 Agent Task Action Score**: SFT=0.504, Ablation-Attn=0.564, Ablation-r32=0.476, **Qwen3 GRPO=0.865**
- **External Benchmarks**: Lingshu MedQA=64.5%, MMLU=75.5%; P2-SFT MMLU=76.8%; Qwen3 MedQA=44.0%
- **Self-Play Loop**: 25 trajectories collected, 23 filtered, SFT trained, eval completed
- **New Files**: self_play.py, benchmark_eval.py, scale_tasks.py, 211 scaled tasks

### [2026-02-12] Healthcare AI GYM 대규모 확장
- **작업 내용**:
  1. **Visual QA Pipeline 완전 구축** (Priority 1 — 가장 큰 gap 해소)
     - `bioagents/data_pipeline/vqa_loader.py`: 6개 VQA 데이터셋 통합 로더
       - VQA-RAD (HuggingFace), SLAKE (local + HF), PathVQA (local + HF)
       - PMC-VQA (HF), VQA-Med-2021 (local), Quilt-VQA (local + HF)
       - 통합 포맷: {image_path, question, answer, answer_type, modality, category}
       - VQA_DATASET_REGISTRY: 6개 데이터셋 메타데이터 + 로더 레지스트리
     - `bioagents/evaluation/vqa_benchmark_eval.py`: VL 모델 지원 VQA 평가기
       - Qwen2.5-VL 네이티브 멀티모달 추론 지원
       - Text-only 폴백 (이미지 없는 모델용)
       - 4종 VQA 메트릭: Exact Match, Token F1, BLEU-1, Contains Match + BERTScore
       - per-dataset & per-answer-type 리포팅
     - `benchmark_eval.py` 통합: CLI에서 --benchmarks vqa_rad slake pathvqa 등으로 VQA 평가 가능
  2. **Scaled Tasks GYM 통합** (Priority 2)
     - `agent_env.py` 대폭 업그레이드:
       - `use_scaled_tasks=True` 옵션으로 tasks_scaled.json 자동 로드
       - 7개 도메인 자동 등록 (lazy loading, importlib 기반)
       - `get_gym_stats()`: 전체 GYM 통계 (도메인별 task/tool 수)
       - `get_registered_domains()`: 등록된 도메인 목록 조회
     - EHR 스키마 정규화: `_normalize_task_schema()` 함수
       - expected_actions/rubric → evaluation_criteria 자동 변환
       - 모든 도메인 동일 평가 스키마 사용 가능
  3. **Triage & Emergency 도메인 신규 구축** (Domain 6)
     - `bioagents/domains/triage_emergency/` — data_model, tools, environment
     - Data Model: EmergencyPatient, EDResource, EmergencyProtocol, TriageDecision, EDStatus
     - Tools 12개: get_patient_presentation, get_vital_signs, assess_airway_breathing,
       get_medical_history, calculate_gcs, calculate_esi_level, get_ed_status,
       check_protocol, order_stat_labs, order_imaging, think, submit_answer
     - DB: 10명 환자 (STEMI, SAH, stroke, sepsis, fracture, CHF, ankle, acute abdomen, med refill, anaphylaxis)
     - Protocols: STEMI Alert, Stroke Alert, Sepsis 1-Hour Bundle, Anaphylaxis
     - Tasks: 10개 (ESI 1~5 모든 레벨 커버)
  4. **Radiology Report Generation 도메인 신규 구축** (Domain 7)
     - `bioagents/domains/radiology_report/` — data_model, tools, environment
     - Data Model: RadiologyStudy, ReportTemplate, PriorReport, RadiologyKnowledge
     - Tools 11개: get_study_info, get_clinical_history, get_prior_reports,
       get_report_template, analyze_findings, search_radiology_knowledge,
       get_reporting_checklist, calculate_measurements, think, submit_report, submit_answer
     - DB: 8개 study (CXR pneumonia, CT stroke, CT abdomen, MRI brain tumor,
       CT lung nodule, CXR normal, Mammogram BI-RADS 5, US thyroid TI-RADS 5)
     - Knowledge Base: Fleischner, BI-RADS, TI-RADS, Acute Stroke CT, Pneumonia CXR
     - Tasks: 8개 (radiology reporting across modalities)
  5. **LLM-based Task Generation Pipeline**
     - `scripts/generate_tasks_llm.py`: OpenAI/Anthropic API 기반 task 자동 생성
     - 도메인별 프롬프트 템플릿: clinical_diagnosis, triage_emergency, drug_interaction, radiology_report
     - 생성 → 검증 → 저장 → split 자동화
     - `scripts/scale_tasks.py` 업데이트: triage_emergency, radiology_report 템플릿 추가
- **결과 요약**:
  - 총 **7개 의료 도메인**, **~80개 도구**, 확장 가능한 GYM 환경
  - 6개 VQA 벤치마크 평가 파이프라인 완성 (기존 0개 → 6개)
  - Scaled tasks가 GYM에 통합 (use_scaled_tasks=True)
  - 모든 도메인 통일된 evaluation_criteria 스키마
  - LLM 기반 task 자동 생성 파이프라인
- **관련 파일**:
  - `bioagents/data_pipeline/vqa_loader.py` (6개 VQA 통합 로더)
  - `bioagents/evaluation/vqa_benchmark_eval.py` (VQA 평가기)
  - `bioagents/evaluation/benchmark_eval.py` (VQA 통합 업데이트)
  - `bioagents/gym/agent_env.py` (scaled tasks + 7 도메인 + 스키마 정규화)
  - `bioagents/domains/triage_emergency/` (전체 도메인)
  - `bioagents/domains/radiology_report/` (전체 도메인)
  - `data/domains/triage_emergency/` (db, policy, tasks, split)
  - `data/domains/radiology_report/` (db, policy, tasks, split)
  - `scripts/generate_tasks_llm.py` (LLM task 생성)
  - `scripts/scale_tasks.py` (새 도메인 템플릿 추가)

### [2026-02-12] Healthcare AI GYM v2: Safety, Cross-Domain, Complete System
- **작업 내용**:
  1. **DB 정합성 완전 수정**
     - `visual_diagnosis`: IMG_XXX → IMG0XX 포맷 통일, 21개 이미지(IMG011-IMG031) + 리포트 + 질문 + 환자 컨텍스트 추가
     - `drug_interaction`: 9종 누락 약물 추가 (methotrexate, ciprofloxacin, theophylline, digoxin, lithium, rifampin, oral_contraceptives, contrast_dye, NSAIDs) + 6개 상호작용 추가
     - 4개 도메인 `split_tasks.json` 생성 (clinical_diagnosis, drug_interaction, ehr_management, visual_diagnosis)
  2. **Safety Evaluation 모듈 구축** (`bioagents/evaluation/safety_eval.py`)
     - 5개 Safety Reward Functions: contraindication, emergency_recognition, uncertainty, scope, composite
     - SafetyViolation 분류 체계 (severity 1-5, 13 categories)
     - 12개 Adversarial Test Cases: harmful_instruction(4), jailbreak(2), misinformation(2), bias_probe(2), scope_test(2)
     - GRPO-compatible safety reward wrapper (`grpo_safety_reward`)
     - 약물 교차반응성(cross-reactivity) 체크, 임신 Category D/X 약물 검증
     - rewards.py 레지스트리에 safety 함수 등록
  3. **Cross-Domain Clinical Pathway 시스템** (`bioagents/domains/cross_domain/`)
     - Pathway Engine: 다단계 환자 여정 오케스트레이션
     - 6개 Clinical Pathways: Chest Pain(ED), DKA, Stroke Code, Sepsis Bundle, Post-op PE, Pediatric Kawasaki
     - 25개 phase-level tasks spanning 5 domains
     - 각 pathway에 patient_data, time_pressure, safety_critical 메타데이터
     - PathwayResult 평가 체계: phase scores + safety + coherence + time compliance
     - GYM 환경 등록 (8번째 도메인)
  4. **Training Configs 확장**
     - `configs/grpo_triage_emergency.yaml`: safety 30%, 낮은 temperature, 높은 KL penalty
     - `configs/grpo_radiology_report.yaml`: format 30%, 긴 completion, 구조화된 출력
     - `configs/grpo_cross_domain.yaml`: flagship config, 5D reward (accuracy+format+process+safety+coherence)
     - `configs/grpo_p2_multidomain.yaml`: 8 도메인으로 업데이트
  5. **README.md 전면 재작성**: 프로젝트 포탈 수준으로 완전히 재구성
- **결과 요약**:
  - 총 **8개 의료 도메인 + 1 cross-domain**, **88개 도구**, **537개 tasks**
  - **5차원 보상 체계**: Accuracy + Format + Process + Safety + Coherence
  - **12개 adversarial test cases** 포함한 safety evaluation 완성
  - **6개 cross-domain pathways** (25 phase tasks): 실제 임상 환자 여정 시뮬레이션
  - 모든 도메인 DB 정합성 100%, split_tasks.json 100% 커버리지
  - **17개 training configs** (SFT, GRPO, Self-Play, Cross-Domain, 도메인별)
- **관련 파일**:
  - `bioagents/evaluation/safety_eval.py` (Safety 모듈)
  - `bioagents/domains/cross_domain/` (pathway_engine, environment)
  - `data/domains/cross_domain/` (tasks.json, split_tasks.json, policy.md)
  - `configs/grpo_triage_emergency.yaml`, `configs/grpo_radiology_report.yaml`, `configs/grpo_cross_domain.yaml`
  - `README.md` (전면 재작성)
  - `data/domains/visual_diagnosis/db.json` (21개 이미지 추가)
  - `data/domains/drug_interaction/db.json` (9개 약물 + 6개 상호작용 추가)

### [2026-02-13] Session 2: 시스템 완성도 대폭 강화
- **작업 내용**:
  1. **FairGRPO 메커니즘 구현** (`grpo_rewards.py`, `grpo_trainer.py`, `gym_coach.py`)
     - FairnessTracker: 인구통계 그룹별 보상 추적 (age_group/sex/ethnicity)
     - Representation-aware + Performance-aware 적응형 가중치
     - `grpo_fairness_reward`, `grpo_fair_composite_reward` 함수
     - `FairGRPOConfig` + `train_fair_grpo()` — TRL 기반 공정성 인식 학습
     - GymCoach `_train_fair_grpo()` 통합
  2. **Multi-turn GRPO 완전 구현** (`grpo_trainer.py`)
     - Placeholder → 환경-인-더-루프 실제 학습 루프 구현 (~300줄)
     - `_run_single_rollout()`: GYM 환경에서 다회전 에이전트-환경 상호작용
     - `_grpo_policy_update()`: Group-relative advantage 계산 + 정책 경사 업데이트
     - `_save_trajectories()`: 궤적 저장 + 분석
     - `MultiTurnGRPOConfig`: rollouts_per_task, max_turns, trajectory_epochs 등
  3. **Agent Runner 도메인별 프롬프트** (`agent_runner.py`)
     - 10개 도메인 전문 시스템 프롬프트 추가 (기존: medical_qa만)
     - 도메인별 역할, 도구 사용 가이드, 최종 응답 형식 커스터마이징
  4. **태스크 도메인 확장**
     - triage_emergency: 10 → 20 (DKA, 긴장성기흉, 수막염, 충수염, 과량복용, 화상, 자궁외임신, 간질중첩, 급성사지허혈, 정신과응급)
     - radiology_report: 8 → 20 (무릎MRI, 경추CT, COVID CT, 담낭US, PE CTA, 간MRI, V/Q, 신장CT, 어깨MRI, 뇌MS MRI, 골반US, 심장CTA)
     - psychiatry: 12 → 20 (섭식장애, OCD, 양극성혼합, BPD, 성인자폐, 불면증, 복잡사별, 법정신의학)
     - obstetrics: 12 → 20 (조기진통, 쌍둥이, 전치태반, HELLP, 제대탈출, GBS, IUGR, 양수색전)
  5. **누락 GRPO configs 추가** (5개)
     - grpo_clinical_diagnosis.yaml (safety 0.2)
     - grpo_visual_diagnosis.yaml (format 0.3)
     - grpo_ehr_management.yaml (process 0.5, max_prompt 3072)
     - grpo_psychiatry.yaml (safety 0.25)
     - grpo_obstetrics.yaml (safety 0.2)
  6. **경쟁자 심층 분석** — DiagGym vs MedAgentGym 상세 비교표 + rebuttal 준비
  7. **환자 데이터 25건 추가** — 5개 도메인 (clinical_diagnosis, drug_interaction, triage, ehr, radiology)
  8. **라이선스 체계 수립** — Apache-2.0 + NOTICE + THIRD_PARTY_LICENSES.md (40+ 컴포넌트)
  9. **Git Submodule 연결** — AgentGym-RL, tau2-bench
- **결과 요약**:
  - 총 **10개 의료 도메인**, **126+ 도구**, **~600 tasks** (scaled 포함)
  - **모든 도메인 GRPO config 완비** (10/10)
  - **Multi-turn GRPO 완전 구현** — 환경 루프 + GRPO 정책 업데이트
  - **FairGRPO** — 세계 최초 의료 AI 공정성 인식 RL 학습
  - **Agent Runner** 10개 도메인 전문 프롬프트 완성
  - Apache-2.0 라이선스 + AI 생성 코드 공시 + 써드파티 라이선스 완전 정리
- **다음 단계**:
  - 실제 GPU 학습 실행 (SFT warmup → Multi-turn GRPO)
  - 전체 벤치마크 baseline 평가 (10개 도메인 + external benchmarks)
  - 결과 테이블 작성 → 논문 초안

### 향후 기록 형식
```
### [YYYY-MM-DD] 작업 제목
- **작업 내용**: 수행한 작업 상세
- **사용 모델/데이터**: 
- **결과 요약**: 
- **다음 단계**: 
- **관련 파일**: 경로 목록
```

---

## 8. 리스크 & 대응 전략

| 리스크 | 영향 | 확률 | 대응 |
|---|---|---|---|
| MIMIC 데이터 접근 불가 | EHR 도메인 구현 불가 | 중 | 합성 EHR 데이터로 대체 |
| RL 학습 불안정 | 성능 저하 | 고 | SFT warmup + KL penalty + ScalingInter |
| A100 8대 리소스 부족 | 대형 모델 학습 불가 | 중 | 7B 모델 집중, LoRA/QLoRA 적용 |
| NeurIPS 마감 (6월) | 시간 부족 | 중 | Phase 1-2 엄격 관리, MVP 우선 |
| Tool simulation 품질 | 비현실적 결과 | 중 | GPT-5/Claude로 고품질 simulation |

---

## 9. 코드 컨벤션 & 로깅 규칙

### 9.1 디렉토리 규칙
- 모든 실험 결과는 `logs/` 디렉토리에 날짜별 저장
- 체크포인트는 `checkpoints/` 디렉토리에 실험명_날짜로 저장
- 학습 설정은 `configs/` 디렉토리에 YAML로 관리

### 9.2 로깅 규칙
- 모든 학습은 W&B (Weights & Biases)에 기록
- Trajectory는 JSON 형식으로 전체 저장
- 평가 결과는 표준 JSON 형식으로 저장
- 코드 변경은 Git commit으로 추적

### 9.3 파일 명명 규칙
- 데이터: `{domain}_{split}_{version}.json`
- 설정: `{model}_{method}_{date}.yaml`
- 로그: `{experiment_name}_{date}_log.txt`
- 체크포인트: `{model}_{method}_{step}/`

---

## 10. Autonomous GYM Architecture (v3)

> **추가일**: 2026-02-13

### 10.1 핵심 변화: Coach-Driven → Agent-Driven

기존 GymCoach는 **탑다운** 구조였다:
- GymCoach가 모든 커리큘럼을 결정
- 모든 에이전트가 같은 루프를 따름
- GPU는 순차적으로 사용

새로운 Autonomous GYM은 **바텀업** 구조:
- 에이전트가 **스스로** 약점을 인지하고 학습 방향을 결정
- 여러 에이전트가 **동시에** GPU를 활용하여 비동기 학습
- SharedLogbook을 통해 **서로의 기록을 참조**하며 상호 학습
- GYM은 자원 관리만 담당, 학습 방향은 에이전트가 결정

### 10.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AutonomousGym (Shared Space)                   │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Station 1    │  │  Station 2    │  │  Station N            │  │
│  │ Clinical Dx   │  │ Drug Safety   │  │ Emergency             │  │
│  │  (GPU 0-1)    │  │  (GPU 2-3)    │  │  (GPU 4-5)            │  │
│  └──────┬────────┘  └──────┬────────┘  └──────┬─────────────┘   │
│         │                  │                   │                  │
│  ┌──────▼──────────────────▼───────────────────▼──────────────┐  │
│  │                  SharedLogbook                               │  │
│  │  - Agent A: "drug interaction에서 3번 실패"                  │  │
│  │  - Agent B: "emergency triage 정확도 92%"                   │  │
│  │  - Agent C: "obstetrics에서 safety 위반"                    │  │
│  │  - Leaderboard + InsightEngine + Herding Detection          │  │
│  └──────┬──────────────────┬───────────────────┬──────────────┘  │
│         │                  │                   │                  │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌────────▼──────────┐     │
│  │  Agent A      │  │  Agent B      │  │  Agent C           │    │
│  │ (Qwen3-8B)   │  │(LingShu-8B)  │  │ (Qwen3-8B         │    │
│  │              │  │              │  │  safety variant)   │    │
│  │ REFLECT →    │  │ REFLECT →    │  │ REFLECT →          │    │
│  │ CHOOSE →     │  │ CHOOSE →     │  │ CHOOSE →           │    │
│  │ TRAIN →      │  │ TRAIN →      │  │ TRAIN →            │    │
│  │ RECORD       │  │ RECORD       │  │ RECORD             │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            GymScheduler + SafetyGuardrail                    │  │
│  │  - GPU 할당/해제 (자원 관리만)                               │  │
│  │  - Safety score floor 모니터링                               │  │
│  │  - Consecutive failure 감지 → cooldown                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 New Modules

| 모듈 | 파일 | 역할 |
|------|------|------|
| **SharedLogbook** | `bioagents/gym/shared_logbook.py` | 모든 에이전트의 기록을 공유. Workout logging, Profile 관리, Leaderboard, Herding 감지, Cross-agent 인사이트 |
| **AutonomousAgent** | `bioagents/gym/autonomous_agent.py` | 자기 인식(SelfAwareness) + 전략 선택(StrategySelector) + 운동 실행(WorkoutExecutor). 6가지 동기(Motivation): curiosity, weakness, peer_learning, diversity, mastery_push, safety |
| **AutonomousGym** | `bioagents/gym/autonomous_gym.py` | GymScheduler(GPU 관리) + SafetyGuardrail + AgentWorker pool. 에이전트들이 비동기적으로 출입하는 공유 공간 |

### 10.4 Agent Decision Flow

```
AutonomousAgent.run_one_cycle():
  1. REFLECT
     └─ SelfAwareness.reflect()
        ├─ 내 최근 기록 분석 (strengths, weaknesses)
        ├─ Plateau 감지
        └─ Improvement rate 계산

  2. CHOOSE
     └─ StrategySelector.choose_next_action()
        ├─ 각 도메인에 6가지 motivation factor로 점수 계산
        │   ├─ weakness_weight × (내 약점 도메인인가?)
        │   ├─ curiosity_weight × (안 해본 도메인인가?)
        │   ├─ peer_learning_weight × (다른 에이전트가 잘하는데 나는 못하나?)
        │   ├─ diversity_weight × (herding 방지 - 덜 방문된 도메인인가?)
        │   ├─ mastery_push_weight × (거의 정복 직전인가?)
        │   └─ safety_weight × (safety 위반 이력이 있나?)
        ├─ ε-greedy: 10% 확률로 차선책 선택 (exploration)
        └─ Plateau 도메인에는 점수 감소 (diversity 유도)

  3. TRAIN
     └─ WorkoutExecutor.execute_workout()
        ├─ Evaluate → Analyze errors → Generate data → Train
        └─ 결과를 SharedLogbook에 기록

  4. RECORD
     └─ SharedLogbook.record_workout()
        └─ 다른 에이전트가 이 기록을 읽고 자기 전략에 반영
```

### 10.5 Config

```yaml
# configs/autonomous_gym.yaml
gym:
  num_gpus: 8
  safety_score_floor: 0.30
  max_consecutive_failures: 5

agents:
  - agent_id: "qwen3_8b_v1"
    weakness_weight: 0.35      # 약점 집중
    curiosity_weight: 0.15

  - agent_id: "qwen3_8b_explorer"
    weakness_weight: 0.20
    curiosity_weight: 0.30     # 탐험 집중

  - agent_id: "lingshu_8b_v1"
    peer_learning_weight: 0.25 # 동료 학습 집중

  - agent_id: "qwen3_8b_safety"
    safety_weight: 0.35        # Safety 전문가
```

### 10.6 기존 GymCoach와의 관계

Autonomous GYM은 GymCoach를 **대체**하는 것이 아니라 **발전**시킨 것:
- GymCoach의 ErrorAnalyzer, TargetedDataGenerator, CurriculumScheduler는 그대로 재사용
- SelfPlayLoop, TrainingMemory도 AutonomousAgent 내부에서 활용
- 차이점: 오케스트레이션이 GymCoach(중앙 통제) → AutonomousGym(자율 분산)

---

*이 문서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
