# BIOAgents Bug Log & Prevention Registry

> **목적**: 발견된 모든 버그를 기록하고, 근본 원인과 재발 방지 대책을 명시하여 같은 실수를 반복하지 않도록 한다.
> **규칙**: 버그 수정 시 반드시 이 파일에 기록할 것. PR 리뷰 시 이 로그와 대조할 것.

---

## Bug Index

| ID | 심각도 | 카테고리 | 요약 | 상태 |
|----|--------|----------|------|------|
| [BUG-001](#bug-001) | CRITICAL | Data Contamination | 외부 벤치마크 TEST 파일로 GYM task 생성 | FIXED |
| [BUG-002](#bug-002) | CRITICAL | Data Contamination | SFT 데이터에 test split task 포함 | FIXED |
| [BUG-003](#bug-003) | CRITICAL | Data Contamination | Trajectory에서 test task 미필터링 | FIXED |
| [BUG-004](#bug-004) | HIGH | Data Contamination | medical_qa_200 오염 데이터 사용 | FIXED |
| [BUG-005](#bug-005) | MEDIUM | Data Pipeline | Contamination checker가 nested JSON 미파싱 | FIXED |
| [BUG-006](#bug-006) | HIGH | Model Loading | VL 모델을 AutoModelForCausalLM으로 로딩 시도 | FIXED |
| [BUG-007](#bug-007) | MEDIUM | Model Loading | model_type "qwen2_5_vl_text" 미인식 | FIXED |
| [BUG-008](#bug-008) | HIGH | GYM Environment | env.reset() 반환값 타입 불일치 (str vs dict) | FIXED |
| [BUG-009](#bug-009) | HIGH | GYM Environment | Gymnasium 환경 ID 불일치 | FIXED |
| [BUG-010](#bug-010) | MEDIUM | Dependencies | prometheus_client 미설치 | FIXED |
| [BUG-011](#bug-011) | MEDIUM | Config | SelfPlayConfig 인자명 불일치 | FIXED |
| [BUG-012](#bug-012) | LOW | Config | SFT config 모델 경로 오류 | FIXED |
| [BUG-013](#bug-013) | MEDIUM | Data Pipeline | MedMCQA train 파일 nested list 포맷 | FIXED |

---

## Bug Details

### BUG-001
**외부 벤치마크 TEST 파일로 GYM task 생성** `CRITICAL` `Data Contamination`

- **발견일**: 2026-02-12
- **파일**: `scripts/generate_gym_data.py`
- **증상**: MedQA/MedMCQA/MMLU 평가 점수가 비정상적으로 높을 수 있음
- **근본 원인**: `generate_gym_data.py`가 `med_qa_test.jsonl`, `medmc_qa_test.jsonl`, `mmlu_test.jsonl` (외부 벤치마크 **테스트** 파일)에서 GYM task를 생성. 이 task들로 SFT/GRPO 학습하면 evaluation에서 이미 본 문제를 풀게 됨.
- **수정**:
  - `med_qa_train_gpt4.jsonl`, `medmc_qa_train_gpt4.jsonl`에서만 task 생성
  - MMLU는 train split이 없으므로 GYM에서 완전 제외 (eval-only)
  - 재생성 후 cross-reference 검증: **0/50 overlap**
- **재발 방지**:
  - `generate_gym_data.py`에 `# CONTAMINATION GUARD` 주석 + assertion 추가
  - 새 벤치마크 추가 시 반드시 train/test 분리 확인
  - **절대 `*_test.jsonl` 파일에서 학습/GYM 데이터를 만들지 말 것**

```python
# CONTAMINATION GUARD: Never load from test files!
assert "test" not in str(benchmark_path).lower(), f"CONTAMINATION: {benchmark_path} is a test file!"
```

---

### BUG-002
**SFT 데이터에 test split task 포함** `CRITICAL` `Data Contamination`

- **발견일**: 2026-02-12
- **파일**: `scripts/generate_p2_sft_data.py`, `scripts/generate_multidomain_sft.py`
- **증상**: SFT 학습 데이터에 test split task ID가 포함됨 (p2: 228/701 = 32.5% 오염!)
- **근본 원인**: `load_domain_tasks()` 함수가 `split_tasks.json`의 train/test 구분 없이 전체 task를 로딩
- **수정**:
  - `load_domain_tasks(domain, split="train")` — 기본값을 train으로 설정
  - Post-processing filter: 모든 SFT 파일에서 test ID를 가진 example 제거
  - 최종 검증: 4개 SFT 파일 모두 **test_leaks=0**
- **재발 방지**:
  - SFT 데이터 생성 스크립트에서 `split="train"` 명시적 호출 필수
  - 생성 후 `scripts/check_contamination.py` 자동 실행

---

### BUG-003
**Trajectory에서 test task 미필터링** `CRITICAL` `Data Contamination`

- **발견일**: 2026-02-13
- **파일**: `scripts/generate_p2_sft_data.py` (`extract_expert_trajectories`)
- **증상**: Baseline 실행 로그에서 추출한 trajectory 중 test task로 실행된 것이 SFT에 포함
- **근본 원인**: `extract_expert_trajectories()` 함수가 trajectory의 task_id를 test split과 대조하지 않음
- **수정**:
  - `_load_all_test_ids()` 함수 추가 — 모든 도메인의 test ID 수집
  - Trajectory 추출 시 test ID와 일치하면 skip
- **재발 방지**:
  - 모든 trajectory 추출 코드에 test ID 필터 의무화
  - `extract_expert_trajectories()`에 필터링 로깅 추가

---

### BUG-004
**medical_qa_200 오염 데이터 사용** `HIGH` `Data Contamination`

- **발견일**: 2026-02-13
- **파일**: `scripts/generate_p2_sft_data.py`
- **증상**: 오염 제거 후에도 p2 SFT에 MMLU/MedMCQA test 데이터 포함
- **근본 원인**: `medical_qa_200/tasks.json`이 `generate_gym_data.py` (BUG-001)으로 생성된 오염 데이터. p2 스크립트가 이 경로에서 추가 task를 로딩.
- **수정**:
  - `medical_qa_200` 로딩 경로 제거
  - `medical_qa/tasks_scaled.json`만 사용하되 train split 필터 적용
- **재발 방지**:
  - 새 데이터 디렉토리 추가 시 출처(train/test) 명시
  - `medical_qa_200` 디렉토리에 `DEPRECATED_CONTAMINATED.txt` 표시

---

### BUG-005
**Contamination checker가 nested JSON 미파싱** `MEDIUM` `Data Pipeline`

- **발견일**: 2026-02-12
- **파일**: `scripts/check_contamination.py`
- **증상**: 오염된 데이터가 있는데도 "ALL CLEAN" 보고
- **근본 원인**: 외부 벤치마크 test 파일의 질문이 `instances.input` 안에 중첩되어 있는데, checker가 top-level field만 확인
- **수정**:
  - `get_external_test_questions()`에서 `d.get("instances",{}).get("input","")` 경로 추가
  - SFT 레코드 검사에서도 동일하게 nested field 탐색
- **재발 방지**:
  - 새 벤치마크 추가 시 데이터 포맷 문서화
  - Checker에 포맷 감지 로직 추가 (자동 nested/flat 구분)

---

### BUG-006
**VL 모델을 AutoModelForCausalLM으로 로딩 시도** `HIGH` `Model Loading`

- **발견일**: 2026-02-13
- **파일**: `bioagents/training/grpo_trainer.py`, `bioagents/training/sft_trainer.py`
- **증상**: `ValueError: Unrecognized configuration class Qwen2_5_VLConfig for AutoModelForCausalLM`
- **근본 원인**: Lingshu-7B와 Qwen2.5-VL-7B는 VL(Vision-Language) 모델인데, `AutoModelForCausalLM`으로 로딩 시도. VL 모델은 `Qwen2_5_VLForConditionalGeneration` 필요.
- **수정**:
  - `AutoConfig.from_pretrained()`로 model_type 감지
  - `qwen2_5_vl*` 타입이면 `Qwen2_5_VLForConditionalGeneration` 사용
- **재발 방지**:
  - 모든 모델 로딩 코드에 VL 모델 분기 필수
  - **패턴**: `_detect_and_load_model()` 유틸 함수 만들어 중앙화 (아래 참조)

```python
def load_model_auto(model_path, dtype=torch.bfloat16):
    """VL/CausalLM 자동 감지 모델 로딩 — 모든 코드에서 이것만 사용할 것."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    is_vl = "qwen2" in model_type.lower() and "vl" in model_type.lower()
    
    if is_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    else:
        return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
```

---

### BUG-007
**model_type "qwen2_5_vl_text" 미인식** `MEDIUM` `Model Loading`

- **발견일**: 2026-02-13
- **파일**: `bioagents/training/grpo_trainer.py`, `sft_trainer.py`, `run_full_benchmark_suite.py`
- **증상**: Lingshu-7B SFT 체크포인트의 model_type이 `qwen2_5_vl_text`인데, 정확 매칭 `("qwen2_5_vl", "qwen2_vl")`에 걸리지 않음
- **근본 원인**: SFT 후 저장된 체크포인트가 `qwen2_5_vl_text`라는 변형 model_type을 가짐. 기존 코드는 정확한 문자열 매칭만 사용.
- **수정**: substring 매칭으로 변경:
  ```python
  is_vl = "qwen2" in model_type.lower() and "vl" in model_type.lower()
  ```
- **재발 방지**:
  - model_type 체크는 항상 **substring 매칭** 사용
  - 정확한 문자열 리스트 매칭 금지 (모델 변형이 생길 수 있음)

---

### BUG-008
**env.reset() 반환값 타입 불일치 (str vs dict)** `HIGH` `GYM Environment`

- **발견일**: 2026-02-13
- **파일**: `bioagents/training/grpo_trainer.py` (`_run_single_rollout`)
- **증상**: `AttributeError: 'str' object has no attribute 'get'`
- **근본 원인**: `BioAgent-v0` 환경의 `reset()`이 `(str, dict)` 튜플을 반환하는데, GRPO trainer가 `obs.get("system_prompt", ...)` 처럼 dict로 접근
- **수정**:
  ```python
  if isinstance(obs, dict):
      observation_text = obs.get("ticket", str(obs))
  else:
      observation_text = str(obs)
  ```
- **재발 방지**:
  - GYM 환경 인터페이스 문서화: `reset()` → `(str, dict)`, `step()` → `(str, float, bool, bool, dict)`
  - 환경 사용 코드에서 항상 타입 체크

---

### BUG-009
**Gymnasium 환경 ID 불일치** `HIGH` `GYM Environment`

- **발견일**: 2026-02-12
- **파일**: `bioagents/training/grpo_trainer.py`
- **증상**: `gymnasium.error.NameNotFound: Environment 'bioagent-gym' doesn't exist`
- **근본 원인**: `gym.make("bioagent-gym-v0")` 사용했지만, 실제 등록된 ID는 `"BioAgent-v0"`
- **수정**: `gym.make("BioAgent-v0")`로 변경
- **재발 방지**:
  - 환경 ID를 상수로 정의: `GYM_ENV_ID = "BioAgent-v0"`
  - 모든 코드에서 상수 참조

---

### BUG-010
**prometheus_client 미설치** `MEDIUM` `Dependencies`

- **발견일**: 2026-02-12
- **파일**: TRL 라이브러리 내부 (GRPOTrainer 의존)
- **증상**: `ModuleNotFoundError: No module named 'prometheus_client'`
- **근본 원인**: TRL의 GRPOTrainer가 메트릭 수집을 위해 `prometheus_client`를 필요로 하지만, conda 환경에 미설치
- **수정**: `pip install prometheus_client`
- **재발 방지**:
  - `requirements.txt`에 `prometheus_client` 추가
  - 환경 셋업 스크립트에 의존성 검증 단계 추가

---

### BUG-011
**SelfPlayConfig 인자명 불일치** `MEDIUM` `Config`

- **발견일**: 2026-02-12
- **파일**: `scripts/gpu_gym_scheduler.py`
- **증상**: `TypeError: SelfPlayConfig.__init__() got an unexpected keyword argument 'num_iterations'`
- **근본 원인**: 스케줄러에서 `num_iterations`/`trajectories_per_iter` 사용했지만, 실제 `SelfPlayConfig`는 `max_iterations`/`num_trajectories_per_task`
- **수정**: 인자명 일치시킴
- **재발 방지**:
  - 새 Config 클래스 사용 시 반드시 소스 코드 확인
  - Config dataclass에 명확한 docstring 유지

---

### BUG-012
**SFT config 모델 경로 오류** `LOW` `Config`

- **발견일**: 2026-02-12
- **파일**: `configs/8gpu/sft_qwen3_8b_gpu0.yaml`
- **증상**: 모델 파일을 찾을 수 없음
- **근본 원인**: `model.name_or_path`가 `/checkpoints/Qwen3-8B-Base`로 설정되어 있었지만, 실제 경로는 `/models/Qwen3-8B-Base`
- **수정**: 경로 수정
- **재발 방지**:
  - Config 파일에서 모델 경로는 `MODELS` 레지스트리의 값 사용
  - 스케줄러에서 config 생성 시 자동 경로 검증 추가

---

### BUG-013
**MedMCQA train 파일 nested list 포맷** `MEDIUM` `Data Pipeline`

- **발견일**: 2026-02-13
- **파일**: `scripts/generate_gym_data.py`
- **증상**: `AttributeError: 'list' object has no attribute 'get'`
- **근본 원인**: `medmc_qa_train_gpt4.jsonl`의 각 줄이 단일 dict가 아니라 1000개 레코드의 list. `load_jsonl()`이 이 포맷을 처리 못함.
- **수정**:
  ```python
  for r in raw_records:
      if isinstance(r, list):
          records.extend(r)
      else:
          records.append(r)
  ```
- **재발 방지**:
  - `load_jsonl()` 함수에 자동 list-flatten 로직 내장
  - 새 데이터 파일 추가 시 첫 줄 포맷 확인 필수

---

## Prevention Checklist

새로운 코드 작성 시 반드시 확인:

### Data Contamination 방지
- [ ] 학습 데이터 생성 시 `*_test.jsonl` 파일 사용 금지
- [ ] `split_tasks.json`에서 반드시 `train` split만 사용
- [ ] 생성 후 `scripts/check_contamination.py` 실행
- [ ] Trajectory 추출 시 test task ID 필터링
- [ ] 새 벤치마크 추가 시 train/test 분리 문서화

### Model Loading 방지
- [ ] 모델 로딩 전 `AutoConfig`로 model_type 확인
- [ ] VL 모델 (qwen2*vl*) 감지 시 `Qwen2_5_VLForConditionalGeneration` 사용
- [ ] model_type 비교는 substring 매칭 (`"vl" in model_type.lower()`)
- [ ] 새 모델 추가 시 `MODELS` 레지스트리에 type 명시

### GYM Environment 방지
- [ ] 환경 ID는 상수 `BioAgent-v0` 사용
- [ ] `env.reset()` 반환값은 `(str, dict)` — str에 `.get()` 호출 금지
- [ ] `env.step()` 반환값 5-tuple 확인

### Config & Dependencies 방지
- [ ] Config 클래스 인자명 확인 후 사용
- [ ] YAML config의 모델 경로 실제 존재 여부 검증
- [ ] 새 라이브러리 의존 시 `requirements.txt` 업데이트
- [ ] JSONL 파일 로딩 시 list/dict 포맷 자동 감지

---

## History

| 날짜 | 작업 | 수정 버그 |
|------|------|-----------|
| 2026-02-12 | 초기 GPU 가동 | BUG-009, BUG-010, BUG-011, BUG-012 |
| 2026-02-12 | Data contamination 감사 | BUG-001, BUG-002 |
| 2026-02-13 | Contamination 완전 해결 | BUG-003, BUG-004, BUG-005, BUG-013 |
| 2026-02-13 | GPU 8대 풀가동 | BUG-006, BUG-007, BUG-008 |
