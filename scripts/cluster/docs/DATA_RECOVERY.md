# Healthcare_GYM — Data Recovery Report

**Repo:** `/data/project/private/minstar/workspace/minstar/Healthcare_GYM`
**Script:** `./restore_data.sh` (written, **not executed**)
**Date:** 2026-07-28

Every source below was verified against the live HuggingFace / GitHub API from this
machine. Row counts and JSON schemas were checked by actually downloading and parsing
the files (or by streaming the tar headers), not inferred.

**Current state on disk:** `databases/`, `datasets/`, `evaluations/`, and
`data/ehr_benchmarks/` **do not exist at all**. Only `data/domains/` survives. All four
are in `.gitignore`, so restoring them does not dirty the repo's git status.

---

## Headline findings

1. **The DMIS lab data server is unreachable from this cluster.** The canonical
   Self-BioRAG download host `nlp.dmis.korea.edu` resolves (163.152.163.168) but TCP/80
   and TCP/443 both time out. Every `http://nlp.dmis.korea.edu/projects/selfbiorag-.../data/*.tar.gz`
   link in the Self-BioRAG README is dead from here.
2. **A complete HuggingFace mirror of those tarballs exists**: `awinml/biorag_data`
   carries all four (`instruction`, `retriever`, `generator`, `critic`) `.tar.gz` files,
   ungated. Streamed tar listings confirm they contain the exact filenames the repo wants.
   **Section A is fully recoverable.**
3. **The deleted download scripts were recovered from git history.**
   Commit `6ac472e` ("refactor: clean up repo") deleted `scripts/download_medical_images.py`
   and `scripts/download_medical_sft_data.py`. Recovered via
   `git show 6ac472e^:scripts/download_medical_images.py`. This gave the *exact*
   filename convention for the 520 lost training images, making them
   **bit-for-index reconstructable** (see D-train).
4. **Only two things are not recoverable:** MIMIC-III / eICU (credentialed PhysioNet),
   and Quilt-VQA (gated; this HF account is authenticated but **not approved** — HTTP 403).
5. **No schema conversion is needed for A, B, or C.** The upstream files are already in
   the exact shapes the repo's loaders expect. Verified field-by-field.

---

## A. 828K-passage medical knowledge base (`databases/`)

Consumed by `scripts/build_medical_fts_index.py` → `databases/medical_knowledge_fts.sqlite`.

| Target path | Source | Exists | Gated | Download | Extracted |
|---|---|---|---|---|---|
| `databases/retriever/medcpt_top10_evidence_createret.json` | `awinml/biorag_data` → `retriever.tar.gz` | **YES** | no | 440,244,685 B | ~1.5–2 GB (2 files) |
| `databases/instruction/all_biomedical_instruction.json` | `awinml/biorag_data` → `instruction.tar.gz` | **YES** | no | 74,483,809 B | 136,874,535 B |
| `databases/instruction/MedInstruct-52k.json` | same tarball | **YES** | no | (same) | 68,740,963 B |
| `databases/generator/bio_generator_train.json` | `awinml/biorag_data` → `generator.tar.gz` | **YES** | no | 526,310,647 B | multi-GB (incl. `batch_output/`) |

**Original (dead) source:** `http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/{instruction,retriever,critic,generator}.tar.gz`
— listed in the `dmis-lab/self-biorag` GitHub README. **UNREACHABLE (TCP timeout).**
The mirror is the only working route.

**Tarball internal layout matches the repo's expected paths exactly** (verified by
streaming `curl … | tar -tzf - | head`):

```
instruction/  mol_instruction_qa.json  MedInstruct-52k.json
              all_biomedical_instruction.json  self_instruct_biomedical.json
retriever/    medcpt_top10_evidence_createret.json  medcpt_top10_evidence.json
generator/    bio_generator_train.json  bio_generator_train.jsonl  batch_output/…
critic/       bio_critic_train.json  critic_5k_{retrieval,utility}.json  …
```

so `tar -xzf retriever.tar.gz -C databases/` lands the file at precisely
`databases/retriever/medcpt_top10_evidence_createret.json`. **No renaming, no conversion.**

### Schema verification (this is the load-bearing part)

`build_medical_fts_index.py` reads the four files with **two different parsers** and
would silently index 0 rows if either guess were wrong. Both were checked against the
real bytes:

- `medcpt_top10_evidence_createret.json` — read **line-by-line as JSONL**
  (`index_medcpt_evidence`, line 93-99). Confirmed: the file does **not** start with `[`;
  line 1 parses as a standalone object with keys
  `{q_id, question, evidence, dataset_name, instruction, output, answers, target_output, preceding_sentences, sent_idx}`.
  The builder needs `q_id` / `question` / `evidence` / `dataset_name` — all present. ✅
- `all_biomedical_instruction.json` — `json.load` **array**. Confirmed: list of
  **122,349** objects, keys `{id, instruction, input, output, topic, dataset_name, metadata}`. ✅ (matches "~122K")
- `MedInstruct-52k.json` — `json.load` **array**. Confirmed: list of **52,002** objects,
  keys `{instruction, input, output}`. ✅
- `bio_generator_train.json` — `json.load` **array**; builder regexes
  `<paragraph>(.*?)</paragraph>` out of `output`. Confirmed: starts with `[{"instruction": …,
  "output": "[Retrieval]<paragraph>…</paragraph>…"}`. ✅

Section A of `restore_data.sh` re-runs all four of these assertions after extraction and
exits non-zero on mismatch.

**Note on "828K":** the builder's own docstring says 581K evidence + 122K instructions +
84K generator + 52K MedInstruct → `passages_fts` ≈ 838K, `evidence_fts` ≈ 581K. The
restore script prints the true `medcpt` line count so the paper number can be
re-derived rather than assumed.

`critic.tar.gz` (41,804,006 B) is **not** indexed by `build_medical_fts_index.py` despite
being named in its docstring; it is downloaded only under `WITH_CRITIC=1`.

---

## B. Text QA benchmarks

Expected paths come from `scripts/eval_benchmark_multiturn.py` `BENCHMARK_FILES`
(L79-107) and `bioagents/evaluation/benchmark_eval.py` `BENCHMARK_PATHS` (L58-67) —
both point at `evaluations/self-biorag/data/benchmark/*.jsonl`.

**Critical:** the repo does **not** want the raw HF datasets. It wants the
*Self-BioRAG-preprocessed* JSONL, which is committed directly in the
`dmis-lab/self-biorag` GitHub repo. Fetching from `cais/mmlu` / `GBaker/MedQA-USMLE-4-options`
would require writing a converter; fetching from GitHub requires none.

| Target file | Source (raw.githubusercontent.com/dmis-lab/self-biorag/main/data/benchmark/) | Exists | Gated | Size | Rows (verified) | Expected |
|---|---|---|---|---|---|---|
| `med_qa_test.jsonl` | ✔ | **YES** | no | 1,466,910 B | **1273** | 1273 ✅ |
| `medmc_qa_test.jsonl` | ✔ | **YES** | no | 2,566,627 B | **4183** | 4183 ✅ |
| `mmlu_clinical_knowledge_test.jsonl` | ✔ | **YES** | no | 180,934 B | **265** | 265 ✅ |
| `mmlu_professional_medicine_test.jsonl` | ✔ | **YES** | no | 337,971 B | **272** | 272 ✅ |
| `mmlu_anatomy_test.jsonl` | ✔ | **YES** | no | 91,265 B | **135** | 135 ✅ |
| `mmlu_medical_genetics_test.jsonl` | ✔ | **YES** | no | 64,148 B | **100** | 100 ✅ |
| `mmlu_college_biology_test.jsonl` | ✔ | **YES** | no | 113,143 B | **144** | 144 ✅ |
| `mmlu_college_medicine_test.jsonl` | ✔ | **YES** | no | 159,235 B | **173** | 173 ✅ |
| `mmlu_test.jsonl` (union, `"mmlu"` key) | ✔ | **YES** | no | 946,696 B | **1089** | = 265+272+135+100+144+173 ✅ |

Total ≈ **5.9 MB**. All nine downloaded and parsed during this investigation; every
count matches the spec exactly.

**Schema — no conversion needed.** Each line is:

```json
{"id":"seed_task_0","name":"med_qa","is_classification":true,
 "instruction":"Given four answer candidates, A, B, C and D, choose the best answer choice.",
 "instances":{"input":"QUESTION: …\nOption A: …\nOption B: …","output":"…"}}
```

`load_textqa_benchmark()` reads `item["instances"]["input"]` / `["output"]` and regexes
`Option [A-E]:` out of the question text — which is exactly the format above. ✅

*Upstream HF equivalents (verified to exist, in case a raw rebuild is ever wanted, but
each would need a converter):* `GBaker/MedQA-USMLE-4-options` (test 1273),
`openlifescienceai/medmcqa` (validation 4183 — note the repo's "dev" split is HF's
`validation`), `cais/mmlu` (subset test counts 265/272/135/100/144/173 — all confirmed
identical). None are gated.

---

## C. MedLFQA long-form QA

| Target file (`evaluations/OLAPH/MedLFQA/`) | Source | Exists | Gated | Size | Rows (verified) | Expected |
|---|---|---|---|---|---|---|
| `kqa_golden_test_MedLFQA.jsonl` | `dmis-lab/MedLFQA` | **YES** | no | 296,961 B | **201** | 201 ✅ |
| `live_qa_test_MedLFQA.jsonl` | `dmis-lab/MedLFQA` | **YES** | no | 146,367 B | **100** | 100 ✅ |
| `medication_qa_test_MedLFQA.jsonl` | `dmis-lab/MedLFQA` | **YES** | no | 701,862 B | **666** | 666 ✅ |
| `healthsearch_qa_test_MedLFQA.jsonl` | `dmis-lab/MedLFQA` | **YES** | no | 4,467,928 B | **3077** | 3077 ✅ |
| `kqa_silver_wogold_test_MedLFQA.jsonl` | `dmis-lab/MedLFQA` | **YES** | no | 1,163,931 B | **904** | 904 ✅ |

Total ≈ **6.8 MB**; sum = 4948 rows, which matches the HF dataset-viewer's reported
`test` split size of 4948 exactly.

**Schema — no conversion needed.** Keys are `Question`, `Free_form_answer`, `Must_have`,
`Nice_to_have`; `Must_have` / `Nice_to_have` are **real JSON arrays** (checked
`isinstance(..., list)`), not stringified lists. `load_textqa_benchmark()` reads exactly
these four names for the `MEDLFQA_BENCHMARKS` set. ✅

*Alternative source:* `github.com/dmis-lab/OLAPH` `MedLFQA/` has the same five filenames.
`healthsearch_qa`, `live_qa`, `medication_qa` are byte-identical to the HF copies;
`kqa_golden` (381,210 B) and `kqa_silver_wogold` (1,212,734 B) differ slightly in
formatting. **Prefer the HF copy** — that is the one whose counts were verified.

---

## D. Visual QA

Target layout read by `load_vqa_benchmark()`:
`datasets/vqa/<name>/test.json` = list of `{"question","answer","image_path"}`, plus
`datasets/vqa/<name>/images/`.

| Benchmark | Source | Exists | Gated | Size | Rows | Expected |
|---|---|---|---|---|---|---|
| VQA-RAD | `flaviagiammarino/vqa-rad` split `test` | **YES** | no | 10,312,735 B | **451** | 451 ✅ |
| SLAKE | `BoKelvin/SLAKE` `test.json` + `imgs.zip` | **YES** | no | 635,829 + 212,343,373 B | 2094 → **1061 en** | 1061 ✅ |
| PathVQA | `flaviagiammarino/path-vqa` split `test` | **YES** | no | 156,396,827 B | **6719** | 6719 ✅ |
| PMC-VQA | `jspetrisko/PMC-VQA-Test-Clean` split `train` | **YES** | no | 222,255,843 B | **1996** | ~2000 ✅ |
| VQA-Med-2021 | GitHub `abachaa/VQA-Med-2021` (official) | **YES** | no | 24,147,698 B zip + 353 KB json | **500** | 500 ✅ |
| Quilt-VQA | `wisdomik/Quilt_VQA` | **YES** | **YES (auto)** | 1,290,121 B json + 305,489,536 B zip | unverifiable | ~985 ⚠️ |

### Per-dataset notes / conversions

- **SLAKE — needs a filter, and images ship separately.** The parquet is annotations-only
  (175 KB for 9835 train rows); images are in `imgs.zip` (212 MB). `test.json` has
  **2094** rows: `Counter({'en': 1061, 'zh': 1033})`. The repo's own
  `bioagents/data_pipeline/vqa_loader.py` filters `q_lang != "en"`, giving exactly the
  expected **1061**. `img_name` is `"xmlab102/source.jpg"` — i.e. already relative to the
  image root — so the restore script unzips `imgs.zip` into `images/` (stripping a
  possible extra `imgs/` wrapper) and writes `image_path = img_name` verbatim.
- **PMC-VQA — deliberate substitute, already the repo's own choice.** The official
  `RadGenome/PMC-VQA` exists but its `images.zip` is **18.9 GB** (+ `images_2.zip` 2.2 GB).
  The deleted `scripts/download_vqa_benchmarks.py` used `jspetrisko/PMC-VQA-Test-Clean`
  (1996 rows, images embedded, 222 MB) instead. The restore script keeps that choice so
  the numbers stay comparable to the existing results. **This is a substitute for the
  official release, not the official release.**
- **VQA-Med-2021 — HF mirror is WRONG size; use GitHub.** `bangthe2222/vqa_med` (named in
  the deleted script) has only **425** test rows, but the repo's benchmark is **500**.
  The official ImageCLEF release `github.com/abachaa/VQA-Med-2021` has exactly **500**
  questions (`Task1-VQAnswering2021-Test-ReferenceQuestions_mscoco_format_vqa.json`,
  verified: `len(d["questions"]) == 500`) plus reference answers and
  `Task1-VQA-2021-TestSet-w-GroundTruth.zip` for the images. This also matches
  `vqa_loader.load_vqa_med_2021()`, which reads those exact filenames. The restore script
  joins questions→answers by `question_id` and images by filename stem.
- **Quilt-VQA — ⚠️ BLOCKED pending one click.** `wisdomik/Quilt_VQA` is `gated: "auto"`.
  Verified from this machine: the local HF token is valid (`whoami` → `Minbyul`), but the
  gated file returns **HTTP 403** → terms not yet accepted. Files present in the repo:
  `quiltvqa_test_w_ans.json` (1.29 MB — the answered test set, presumably the 985),
  `quilt_vqa.zip` (305 MB, images), and `data/train-…parquet`. **Note:** the parquet has
  only a `train` split, so the deleted script's `load_dataset("wisdomik/Quilt_VQA", split="test")`
  would have failed — the restore script uses the raw
  `quiltvqa_test_w_ans.json` + `quilt_vqa.zip` instead. The **985** count could not be
  verified without access. **Fix: accept terms at
  <https://huggingface.co/datasets/wisdomik/Quilt_VQA>, then `bash restore_data.sh D_quilt`.**

### ⚠️ Latent path bug being corrected

`load_vqa_benchmark()` (`scripts/eval_benchmark_multiturn.py`) does:

```python
image_path = item.get("image_path", item.get("image", ""))
if image_path and not Path(image_path).is_absolute():
    image_path = str(data_dir / "images" / image_path)
```

but the deleted `scripts/download_vqa_benchmarks.py` wrote
`"image_path": f"images/{img_filename}"` — producing
`datasets/vqa/<name>/images/images/000001.jpg`, which does not exist. The restore script
writes `image_path` **relative to `images/`** (`"000001.jpg"`, or `"xmlab102/source.jpg"`
for SLAKE) so the join resolves correctly. This is the one intentional deviation from the
original scripts; it does not change any data, only the stored path string.

---

## D-train. The 520 lost training images — FULLY RECONSTRUCTABLE

`data/domains/full_4modality_combined/tasks.json` (5459 tasks) has **800 `_image_path`
references → 520 distinct files**:

- `vqarad_train_XXXXX.jpg` — **220 distinct**, indices 0…480
- `pathvqa_train_XXXXX.jpg` — **300 distinct**, indices 0…299

The naming convention was recovered from the deleted
`scripts/download_medical_images.py` (`git show 6ac472e^:scripts/download_medical_images.py`,
780 lines, saved to the scratchpad). It walks each HF split **in native order** and names
by split index:

```python
# VQA-RAD
ds = load_dataset("flaviagiammarino/vqa-rad", split=split)     # "train", then "test"
for idx, item in enumerate(ds):
    if len(metadata) >= max_samples: break                     # max_samples = 500
    img_name = f"vqarad_{split}_{idx:05d}.jpg"

# PathVQA
ds = load_dataset("flaviagiammarino/path-vqa", split=split)
n = min(len(ds), max_samples - len(metadata))                  # 500
for idx in range(n):
    img_name = f"pathvqa_{split}_{idx:05d}.jpg"
```

No image in either split is `None`, so `idx == len(metadata)` throughout and the cap of
500 stops both at `train[0:500]` — the `test` split never contributed. Since the highest
referenced indices are 480 and 299, **regenerating `train[0:500]` for both datasets
reproduces every referenced file at the correct index.** `restore_data.sh D_train` does
this and then re-scans `tasks.json` to report any still-missing reference (should be 0).

**⚠️ The stored paths are absolute and point at a dead mount:**

```
/mnt/aiplatform/csi-volumes/pvc-e668fe31-e015-4e4e-a3f4-35f18e2ad53f-bd5321b06ddb2b68ae682cc934af2027aeea25db/private/minstar/workspace/BIOAgents/datasets/medical_images/vqa_rad/images/vqarad_train_00314.jpg
```

`/mnt/aiplatform` no longer exists on this machine. The restore script writes images to
`<repo>/datasets/medical_images/{vqa_rad,pathvqa}/images/` and emits
`image_path_remap.json` here (**not** in the repo) with the old→new prefix. Either
rewrite `_image_path` in a copy of `tasks.json`, or recreate the old mount as a symlink:

```bash
sudo mkdir -p /mnt/aiplatform/csi-volumes/pvc-e668fe31-…/private/minstar/workspace/BIOAgents/datasets
sudo ln -s <repo>/datasets/medical_images  /mnt/aiplatform/…/datasets/medical_images
```

Download cost: `flaviagiammarino/vqa-rad` train 24.2 MB + `flaviagiammarino/path-vqa`
train 477 MB (the whole train parquet must be pulled to reach indices 0-499).

---

## E. EHR (MIMIC-III, eICU) — **UNRECOVERABLE without credentials**

**Confirmed: `scripts/build_ehr_benchmark.py` requires raw PhysioNet gzipped CSVs.**
It reads them directly with `gzip.open(...); csv.DictReader(...)` from:

```
data/ehr_benchmarks/mimic_iii/mimic-iii-clinical-database-1.4/
    PATIENTS.csv.gz  ADMISSIONS.csv.gz  ICUSTAYS.csv.gz  DIAGNOSES_ICD.csv.gz
    D_ICD_DIAGNOSES.csv.gz  D_LABITEMS.csv.gz  LABEVENTS.csv.gz
    PRESCRIPTIONS.csv.gz  DRGCODES.csv.gz  PROCEDURES_ICD.csv.gz  D_ICD_PROCEDURES.csv.gz
data/ehr_benchmarks/eicu/eicu-collaborative-research-database-2.0/
```

Both datasets are behind **PhysioNet credentialed access** (CITI "Data or Specimens Only
Research" training + signed DUA) and **redistribution is prohibited**, so no mirror can
exist. **Marked UNRECOVERABLE programmatically.** No download was attempted.
`restore_data.sh E` only prints the manual procedure.

Approximate manual cost if credentials are obtained: MIMIC-III v1.4 ≈ 6.2 GB gz,
eICU-CRD v2.0 ≈ 2.5 GB gz; build time is dominated by `LABEVENTS.csv.gz` (~28 M rows).

**Extra bug found:** `build_ehr_benchmark.py` **writes** `mimic_iii_bench.json.gz` /
`eicu_bench.json.gz`, but `eval_benchmark_multiturn.py` `BENCHMARK_FILES` **reads**
`data/ehr_benchmarks/mimic_iii_bench.json` / `eicu_bench.json` (uncompressed) via a plain
`json.load`. Whoever restores EHR must `gunzip` afterwards or the EHR benchmarks load 0 tasks.

---

## Summary table

| Item | Verified exists | Gated | Recoverable | Conversion needed |
|---|---|---|---|---|
| A. medcpt / instructions / generator / MedInstruct | ✅ (HF mirror) | no | **YES** | none (tar layout == target layout) |
| B. MedQA / MedMCQA / MMLU ×6 (+union) | ✅ | no | **YES** | none |
| C. MedLFQA ×5 | ✅ | no | **YES** | none |
| D. VQA-RAD / PathVQA | ✅ | no | **YES** | HF split → `test.json` + JPEGs |
| D. SLAKE | ✅ | no | **YES** | filter `q_lang=="en"`; unzip `imgs.zip` |
| D. PMC-VQA | ✅ | no | **YES (substitute)** | `jspetrisko` clean-test, not the 19 GB official |
| D. VQA-Med-2021 | ✅ | no | **YES** | GitHub, not the 425-row HF mirror; join qid→answer |
| D. Quilt-VQA | ✅ | **YES** | **BLOCKED** — accept terms, then re-run | raw json + zip (no `test` split in parquet) |
| D-train. 520 images | ✅ | no | **YES** | regenerate `train[0:500]`; remap dead abs paths |
| E. MIMIC-III / eICU | n/a | credentialed | **UNRECOVERABLE** | build from raw CSVs, then gunzip output |

**Total download if everything runs:** ≈ 2.1 GB (section A) + 6 MB (B) + 7 MB (C) +
≈ 1.1 GB (D test sets & images) + ≈ 0.5 GB (D-train) ≈ **3.7 GB**, plus ~307 MB more once
Quilt-VQA access is granted.

## Running it

```bash
bash /data/project/private/minstar/workspace/hcgym_rebuttal/restore_data.sh          # all
bash /data/project/private/minstar/workspace/hcgym_rebuttal/restore_data.sh A B C    # subset
bash /data/project/private/minstar/workspace/hcgym_rebuttal/restore_data.sh D_train  # just the 520 images
```

Idempotent and resumable: each step short-circuits on its final artifact,
`hf_hub_download` resumes via the HF cache, `curl -C -` resumes byte ranges, and JSON
outputs are written to `.part` then renamed so an interrupted run never leaves a
half-written `test.json`. Sections A/B/C self-verify row counts and schemas and exit
non-zero on mismatch. Override targets with `REPO=`, `STAGE=`, `PY=`.
