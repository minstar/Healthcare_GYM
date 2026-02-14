#!/usr/bin/env python3
"""Build EHR Benchmark Datasets from MIMIC-III and eICU.

Reads raw PhysioNet CSVs (gzipped), processes ALL qualifying patient cohorts,
converts to BIOAgents EHR format, and generates evaluation tasks with
clinically grounded rubrics.

Output (gzipped JSON for efficient storage):
    data/ehr_benchmarks/mimic_iii_bench.json.gz  — all qualifying ICU patients + tasks
    data/ehr_benchmarks/eicu_bench.json.gz        — all qualifying ICU patients + tasks

Qualifying criteria:
    MIMIC-III: admission has ICU stay AND ≥3 lab events
    eICU:      has APACHE admission dx, age, unit type AND ≥3 lab results

Usage:
    python scripts/build_ehr_benchmark.py                        # full dataset (all patients)
    python scripts/build_ehr_benchmark.py --n-patients 100       # limit to 100 per dataset
    python scripts/build_ehr_benchmark.py --labs-per-patient 200  # more labs per patient
"""

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_MIMIC_DIR = PROJECT_ROOT / "data/ehr_benchmarks/mimic_iii/mimic-iii-clinical-database-1.4"
DEFAULT_EICU_DIR = PROJECT_ROOT / "data/ehr_benchmarks/eicu/eicu-collaborative-research-database-2.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/ehr_benchmarks"
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def read_gz_csv(path: Path, max_rows: int = 0):
    """Read a gzipped CSV and yield dicts. If max_rows > 0, stop early."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, quotechar='"')
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            yield row


def read_gz_csv_progress(path: Path, desc: str = "", report_every: int = 5_000_000):
    """Read a gzipped CSV with progress reporting (row count every N rows)."""
    t0 = time.time()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, quotechar='"')
        for i, row in enumerate(reader):
            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    {desc}: {(i+1):,} rows ({rate:,.0f} rows/s)", flush=True)
            yield row
    elapsed = time.time() - t0
    print(f"    {desc}: done — {(i+1):,} rows in {elapsed:.1f}s", flush=True)


def safe_float(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=None):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def anon_id(prefix, original_id):
    """Deterministic anonymized ID."""
    h = hashlib.md5(str(original_id).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def write_gzip_json(data: dict, path: Path):
    """Write dict to gzip-compressed JSON (compact format)."""
    t0 = time.time()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str, separators=(",", ":"))
    elapsed = time.time() - t0
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {path} ({size_mb:.1f} MB, {elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# MIMIC-III Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_mimic_benchmark(mimic_dir: Path, n_patients: int = 0,
                          labs_per_patient: int = 100,
                          meds_per_patient: int = 30,
                          procs_per_patient: int = 20):
    """Build benchmark from MIMIC-III data.

    Args:
        mimic_dir: Path to MIMIC-III extracted directory.
        n_patients: Max patients to include. 0 = all qualifying patients.
        labs_per_patient: Max lab events to keep per patient admission.
        meds_per_patient: Max medications per admission.
        procs_per_patient: Max procedures per admission.
    """
    print(f"\n{'='*60}")
    limit_str = f"{n_patients}" if n_patients > 0 else "ALL qualifying"
    print(f"  Building MIMIC-III Benchmark ({limit_str} patients)")
    print(f"{'='*60}")

    build_t0 = time.time()

    # ── Step 1: Load patients ─────────────────────────────────────────────
    print("  [1/9] Loading PATIENTS...", flush=True)
    patients = {}
    for row in read_gz_csv(mimic_dir / "PATIENTS.csv.gz"):
        patients[row["SUBJECT_ID"]] = row
    print(f"    → {len(patients):,} patients")

    # ── Step 2: Load admissions ───────────────────────────────────────────
    print("  [2/9] Loading ADMISSIONS...", flush=True)
    admissions = defaultdict(list)
    all_hadm_ids = set()
    for row in read_gz_csv(mimic_dir / "ADMISSIONS.csv.gz"):
        admissions[row["SUBJECT_ID"]].append(row)
        all_hadm_ids.add(row["HADM_ID"])
    print(f"    → {len(all_hadm_ids):,} admissions")

    # ── Step 3: Load ICU stays ────────────────────────────────────────────
    print("  [3/9] Loading ICUSTAYS...", flush=True)
    icustays = defaultdict(list)
    for row in read_gz_csv(mimic_dir / "ICUSTAYS.csv.gz"):
        icustays[row["HADM_ID"]].append(row)
    icu_hadm_ids = set(icustays.keys())
    print(f"    → {sum(len(v) for v in icustays.values()):,} ICU stays across {len(icu_hadm_ids):,} admissions")

    # ── Step 4: Load diagnoses ────────────────────────────────────────────
    print("  [4/9] Loading DIAGNOSES_ICD...", flush=True)
    diagnoses = defaultdict(list)
    for row in read_gz_csv(mimic_dir / "DIAGNOSES_ICD.csv.gz"):
        diagnoses[row["HADM_ID"]].append(row)
    print(f"    → {sum(len(v) for v in diagnoses.values()):,} diagnosis entries")

    # ── Step 5: Load ICD descriptions ─────────────────────────────────────
    print("  [5/9] Loading D_ICD_DIAGNOSES...", flush=True)
    icd_desc = {}
    for row in read_gz_csv(mimic_dir / "D_ICD_DIAGNOSES.csv.gz"):
        icd_desc[row["ICD9_CODE"]] = row["LONG_TITLE"]
    print(f"    → {len(icd_desc):,} ICD descriptions")

    # ── Step 6: Load D_LABITEMS ───────────────────────────────────────────
    print("  [6/9] Loading D_LABITEMS...", flush=True)
    lab_items = {}
    for row in read_gz_csv(mimic_dir / "D_LABITEMS.csv.gz"):
        lab_items[row["ITEMID"]] = row["LABEL"]
    print(f"    → {len(lab_items):,} lab item definitions")

    # ── Step 7: Identify ALL candidate admissions ─────────────────────────
    print("  [7/9] Identifying candidate admissions (ICU stay required)...", flush=True)
    candidates = []
    for subj_id, adm_list in admissions.items():
        for adm in adm_list:
            hadm = adm["HADM_ID"]
            if hadm in icu_hadm_ids:
                candidates.append((subj_id, hadm))

    random.seed(SEED)
    random.shuffle(candidates)
    if n_patients > 0:
        # Over-sample 3x to account for filtering
        candidates = candidates[:n_patients * 3]

    candidate_hadm_ids = {h for _, h in candidates}
    print(f"    → {len(candidates):,} candidate admissions with ICU stays")

    # ── Step 8: Load labs, meds, procedures for candidates ────────────────
    print("  [8/9] Loading LABEVENTS for candidates (full scan)...", flush=True)
    lab_events = defaultdict(list)
    for row in read_gz_csv_progress(mimic_dir / "LABEVENTS.csv.gz", desc="LABEVENTS"):
        hadm = row.get("HADM_ID")
        if hadm in candidate_hadm_ids and len(lab_events[hadm]) < labs_per_patient:
            lab_events[hadm].append(row)

    hadms_with_labs = {h for h, labs in lab_events.items() if len(labs) >= 3}
    print(f"    → Labs collected for {len(lab_events):,} admissions "
          f"({len(hadms_with_labs):,} with ≥3 labs)")

    print("  [8/9] Loading PRESCRIPTIONS for candidates...", flush=True)
    prescriptions = defaultdict(list)
    for row in read_gz_csv_progress(mimic_dir / "PRESCRIPTIONS.csv.gz", desc="PRESCRIPTIONS"):
        hadm = row.get("HADM_ID")
        if hadm in candidate_hadm_ids and len(prescriptions[hadm]) < meds_per_patient:
            prescriptions[hadm].append(row)
    print(f"    → Prescriptions for {len(prescriptions):,} admissions")

    print("  [8/9] Loading DRGCODES...", flush=True)
    drg_codes = {}
    for row in read_gz_csv(mimic_dir / "DRGCODES.csv.gz"):
        if row["HADM_ID"] in candidate_hadm_ids:
            drg_codes[row["HADM_ID"]] = row

    print("  [8/9] Loading PROCEDURES_ICD...", flush=True)
    procedures = defaultdict(list)
    for row in read_gz_csv(mimic_dir / "PROCEDURES_ICD.csv.gz"):
        hadm = row["HADM_ID"]
        if hadm in candidate_hadm_ids and len(procedures[hadm]) < procs_per_patient:
            procedures[hadm].append(row)

    print("  [8/9] Loading D_ICD_PROCEDURES...", flush=True)
    icd_proc_desc = {}
    for row in read_gz_csv(mimic_dir / "D_ICD_PROCEDURES.csv.gz"):
        icd_proc_desc[row["ICD9_CODE"]] = row["LONG_TITLE"]

    # ── Step 9: Build EHR records ─────────────────────────────────────────
    print("  [9/9] Building EHR records...", flush=True)
    records = {}
    patient_index = defaultdict(list)
    final_count = 0
    max_count = n_patients if n_patients > 0 else len(candidates)

    for subj_id, hadm_id in candidates:
        if final_count >= max_count:
            break

        if hadm_id not in hadms_with_labs:
            continue  # skip if insufficient labs

        pat = patients.get(subj_id, {})
        adm = next((a for a in admissions[subj_id] if a["HADM_ID"] == hadm_id), None)
        if not adm:
            continue

        # Anonymize
        anon_patient_id = anon_id("MPAT", subj_id)
        anon_hadm_id = anon_id("MHADM", hadm_id)

        # Demographics
        dob = pat.get("DOB", "")
        gender = pat.get("GENDER", "O")
        sex_map = {"M": "M", "F": "F"}
        sex = sex_map.get(gender, "O")

        # Age approximation from admission time
        admit_time = adm.get("ADMITTIME", "")
        try:
            admit_year = int(admit_time[:4])
            dob_year = int(dob[:4])
            age = admit_year - dob_year
            if age > 120:
                age = random.randint(65, 89)  # MIMIC uses shifted dates
        except (ValueError, IndexError):
            age = 70

        # Map admission type
        admit_type_map = {
            "EMERGENCY": "emergency",
            "URGENT": "urgent",
            "ELECTIVE": "elective",
            "NEWBORN": "newborn",
        }
        admit_type = admit_type_map.get(adm.get("ADMISSION_TYPE", ""), "emergency")

        # ICD codes
        diag_list = diagnoses.get(hadm_id, [])
        icd_codes = [d["ICD9_CODE"] for d in diag_list[:10]]
        primary_dx = icd_desc.get(icd_codes[0], adm.get("DIAGNOSIS", "")) if icd_codes else adm.get("DIAGNOSIS", "Unknown")

        # LOS
        try:
            at = datetime.strptime(admit_time, "%Y-%m-%d %H:%M:%S")
            dt = datetime.strptime(adm.get("DISCHTIME", admit_time), "%Y-%m-%d %H:%M:%S")
            los_days = round((dt - at).total_seconds() / 86400, 2)
        except (ValueError, TypeError):
            los_days = None

        # ICU stays
        icu_records = []
        for icu in icustays.get(hadm_id, []):
            anon_icu_id = anon_id("MICU", icu["ICUSTAY_ID"])
            icu_records.append({
                "icustay_id": anon_icu_id,
                "hadm_id": anon_hadm_id,
                "patient_id": anon_patient_id,
                "icu_type": icu.get("FIRST_CAREUNIT", "MICU"),
                "intime": icu.get("INTIME", ""),
                "outtime": icu.get("OUTTIME"),
                "los_icu_hours": round(safe_float(icu.get("LOS", 0), 0) * 24, 1),
            })

        # Lab events
        lab_records = []
        for lab in lab_events.get(hadm_id, []):
            item_id = lab.get("ITEMID", "")
            label = lab_items.get(item_id, f"Lab_{item_id}")
            val = safe_float(lab.get("VALUENUM"))
            if val is None:
                continue
            flag = "abnormal" if lab.get("FLAG") == "abnormal" else "normal"
            lab_records.append({
                "itemid": item_id,
                "label": label,
                "value": val,
                "valueuom": lab.get("VALUEUOM", ""),
                "flag": flag,
                "ref_range_lower": None,
                "ref_range_upper": None,
                "charttime": lab.get("CHARTTIME", ""),
            })

        # Medications
        med_records = []
        for i, rx in enumerate(prescriptions.get(hadm_id, [])):
            drug_type_map = {"MAIN": "MAIN", "BASE": "BASE", "ADDITIVE": "ADDITIVE"}
            med_records.append({
                "order_id": anon_id("MORD", f"{hadm_id}_{i}"),
                "drug": rx.get("DRUG", ""),
                "drug_type": drug_type_map.get(rx.get("DRUG_TYPE", "MAIN"), "MAIN"),
                "dose_val": rx.get("DOSE_VAL_RX", ""),
                "dose_unit": rx.get("DOSE_UNIT_RX", ""),
                "route": rx.get("ROUTE", ""),
                "frequency": "",
                "start_time": rx.get("STARTDATE", ""),
                "end_time": rx.get("ENDDATE"),
                "status": "completed",
            })

        # Procedures
        proc_records = []
        for p in procedures.get(hadm_id, []):
            code = p.get("ICD9_CODE", "")
            proc_records.append({
                "procedure_id": anon_id("MPROC", f"{hadm_id}_{code}"),
                "procedure_name": icd_proc_desc.get(code, f"Procedure ICD9 {code}"),
                "icd_procedure_code": code,
                "procedure_time": admit_time,
                "performed_by": "",
                "notes": "",
                "outcome": "",
            })

        # DRG
        drg = drg_codes.get(hadm_id, {})

        # Readmission flag
        patient_adms = admissions.get(subj_id, [])
        is_readmission = len(patient_adms) > 1

        # Build record
        record = {
            "demographics": {
                "patient_id": anon_patient_id,
                "name": f"MIMIC Patient {final_count + 1}",
                "age": min(max(age, 18), 99),
                "sex": sex,
                "date_of_birth": "REDACTED",
                "ethnicity": adm.get("ETHNICITY", ""),
                "language": adm.get("LANGUAGE", "English") or "English",
                "insurance": adm.get("INSURANCE", ""),
                "marital_status": adm.get("MARITAL_STATUS", ""),
            },
            "admission": {
                "hadm_id": anon_hadm_id,
                "patient_id": anon_patient_id,
                "admit_time": admit_time,
                "discharge_time": adm.get("DISCHTIME"),
                "admit_type": admit_type,
                "admit_location": adm.get("ADMISSION_LOCATION", ""),
                "discharge_location": adm.get("DISCHARGE_LOCATION"),
                "diagnosis_at_admission": primary_dx,
                "icd_codes": icd_codes,
                "drg_code": drg.get("DRG_CODE"),
                "los_days": los_days,
                "icu_stays": [ic["icustay_id"] for ic in icu_records],
                "is_readmission": is_readmission,
            },
            "icu_stays": icu_records,
            "lab_events": lab_records,
            "vital_events": [],  # MIMIC vitals are in CHARTEVENTS (huge, skipped)
            "medication_orders": med_records,
            "procedures": proc_records,
            "discharge_summary": None,
            "clinical_scores": [],
            "quality_indicators": {
                "readmission_risk": 0.3 if is_readmission else 0.1,
                "mortality_risk": 0.1 if adm.get("HOSPITAL_EXPIRE_FLAG") == "0" else 0.8,
                "expected_los": los_days or 5.0,
                "sepsis_flag": any("sepsis" in icd_desc.get(c, "").lower() for c in icd_codes),
                "aki_stage": None,
                "notes": "",
            },
            "prior_admissions": [],
        }

        records[anon_hadm_id] = record
        patient_index[anon_patient_id].append(anon_hadm_id)
        final_count += 1

        if final_count % 5000 == 0:
            print(f"    Built {final_count:,} records...", flush=True)

    print(f"  Built {len(records):,} MIMIC-III patient records")

    # ── Build lab reference ranges ────────────────────────────────────────
    lab_reference_ranges = {
        "Creatinine": {"lower": 0.7, "upper": 1.3, "unit": "mg/dL"},
        "Potassium": {"lower": 3.5, "upper": 5.0, "unit": "mEq/L"},
        "Sodium": {"lower": 136, "upper": 145, "unit": "mEq/L"},
        "Hemoglobin": {"lower": 12.0, "upper": 17.5, "unit": "g/dL"},
        "White Blood Cells": {"lower": 4.5, "upper": 11.0, "unit": "K/uL"},
        "Platelet Count": {"lower": 150, "upper": 400, "unit": "K/uL"},
        "Glucose": {"lower": 70, "upper": 100, "unit": "mg/dL"},
        "BUN": {"lower": 7, "upper": 20, "unit": "mg/dL"},
        "Bicarbonate": {"lower": 22, "upper": 29, "unit": "mEq/L"},
        "Chloride": {"lower": 96, "upper": 106, "unit": "mEq/L"},
        "Lactate": {"lower": 0.5, "upper": 2.0, "unit": "mmol/L"},
        "Troponin T": {"lower": 0, "upper": 0.01, "unit": "ng/mL"},
        "INR(PT)": {"lower": 0.8, "upper": 1.2, "unit": ""},
        "Albumin": {"lower": 3.5, "upper": 5.5, "unit": "g/dL"},
        "Bilirubin, Total": {"lower": 0.1, "upper": 1.2, "unit": "mg/dL"},
        "Anion Gap": {"lower": 8, "upper": 12, "unit": "mEq/L"},
    }

    # ── Build ICD descriptions map (subset) ───────────────────────────────
    used_icds = set()
    for rec in records.values():
        used_icds.update(rec["admission"]["icd_codes"])
    icd_descriptions = {c: icd_desc.get(c, f"ICD9 {c}") for c in used_icds}

    # ── Build database ────────────────────────────────────────────────────
    db = {
        "records": records,
        "patient_index": dict(patient_index),
        "lab_reference_ranges": lab_reference_ranges,
        "icd_descriptions": icd_descriptions,
        "query_log": [],
    }

    # ── Build tasks ───────────────────────────────────────────────────────
    tasks = _generate_mimic_tasks(records)

    elapsed = time.time() - build_t0
    print(f"  MIMIC-III build complete: {len(records):,} records, "
          f"{len(tasks):,} tasks in {elapsed:.0f}s")

    return db, tasks


def _generate_mimic_tasks(records: dict) -> list:
    """Generate evaluation tasks from MIMIC-III records (one task per patient)."""
    tasks = []
    rec_list = list(records.values())

    task_templates = [
        {
            "category": "chart_review",
            "difficulty": "medium",
            "template": "Review the complete chart for patient {name} (admission {hadm_id}). Summarize the clinical course including: primary diagnosis, key lab findings, active medications, and current clinical trajectory (improving/stable/deteriorating).",
            "expected_tools": ["get_patient_summary", "get_lab_results", "get_medication_orders"],
            "rubric_keys": ["primary diagnosis identified", "lab abnormalities noted", "medications reviewed", "clinical trajectory assessment"],
        },
        {
            "category": "critical_value_identification",
            "difficulty": "hard",
            "template": "Identify all critical or abnormal lab values for the admission {hadm_id} ({name}). Prioritize by clinical urgency, explain the significance of each, and suggest next steps.",
            "expected_tools": ["get_lab_results", "get_lab_trend", "get_patient_summary"],
            "rubric_keys": ["critical values identified", "prioritized by urgency", "clinical significance explained", "appropriate next steps"],
        },
        {
            "category": "medication_reconciliation",
            "difficulty": "medium",
            "template": "Perform a medication reconciliation for patient {name} ({hadm_id}). List all current medications, check for potential interactions, and verify appropriateness given the diagnoses ({dx}).",
            "expected_tools": ["get_medication_orders", "get_patient_summary", "get_lab_results"],
            "rubric_keys": ["all medications listed", "interactions checked", "appropriateness verified", "dose adjustments considered"],
        },
        {
            "category": "lab_trend_analysis",
            "difficulty": "medium",
            "template": "Analyze the lab value trends for admission {hadm_id} ({name}). Which labs show improving trends? Which are worsening? What clinical conditions do these trends suggest?",
            "expected_tools": ["get_lab_results", "get_lab_trend", "get_patient_summary"],
            "rubric_keys": ["trends correctly identified", "improving vs worsening distinguished", "clinical correlation provided"],
        },
        {
            "category": "discharge_readiness",
            "difficulty": "hard",
            "template": "Assess whether patient {name} ({hadm_id}) is ready for discharge. Review labs, vitals, medications, and clinical scores. Provide a discharge checklist and identify any barriers to safe discharge.",
            "expected_tools": ["get_patient_summary", "get_lab_results", "get_vital_signs", "get_medication_orders", "get_quality_indicators"],
            "rubric_keys": ["discharge criteria assessed", "lab stability confirmed", "medication reconciliation complete", "follow-up plan recommended", "barriers identified"],
        },
    ]

    for i, rec in enumerate(rec_list):
        template = task_templates[i % len(task_templates)]
        name = rec["demographics"]["name"]
        hadm_id = rec["admission"]["hadm_id"]
        dx = rec["admission"]["diagnosis_at_admission"]
        patient_id = rec["demographics"]["patient_id"]

        task = {
            "id": f"mimic_ehr_{i+1:05d}",
            "domain": "ehr_management",
            "source": "mimic_iii",
            "category": template["category"],
            "difficulty": template["difficulty"],
            "ticket": template["template"].format(name=name, hadm_id=hadm_id, dx=dx),
            "patient_id": patient_id,
            "hadm_id": hadm_id,
            "expected_actions": [
                {"tool": t, "args": {"hadm_id": hadm_id}}
                for t in template["expected_tools"]
            ],
            "evaluation_criteria": {
                "actions": [
                    {"name": t, "arguments": {"hadm_id": hadm_id}, "compare_args": ["hadm_id"]}
                    for t in template["expected_tools"]
                ],
                "rubric_keys": template["rubric_keys"],
            },
            "split": "test",
        }
        tasks.append(task)

    print(f"  Generated {len(tasks):,} MIMIC-III evaluation tasks")
    return tasks


# ═══════════════════════════════════════════════════════════════════════════
# eICU Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_eicu_benchmark(eicu_dir: Path, n_patients: int = 0,
                         labs_per_patient: int = 100,
                         meds_per_patient: int = 30,
                         vitals_per_patient: int = 60):
    """Build benchmark from eICU Collaborative Research Database.

    Args:
        eicu_dir: Path to eICU extracted directory.
        n_patients: Max patients to include. 0 = all qualifying patients.
        labs_per_patient: Max lab results per patient.
        meds_per_patient: Max medications per patient.
        vitals_per_patient: Max vital sign readings per patient.
    """
    print(f"\n{'='*60}")
    limit_str = f"{n_patients}" if n_patients > 0 else "ALL qualifying"
    print(f"  Building eICU Benchmark ({limit_str} patients)")
    print(f"{'='*60}")

    build_t0 = time.time()

    # ── Step 1: Load patients ─────────────────────────────────────────────
    print("  [1/7] Loading patient table...", flush=True)
    patient_rows = []
    for row in read_gz_csv(eicu_dir / "patient.csv.gz"):
        patient_rows.append(row)
    print(f"    → {len(patient_rows):,} patient unit stays")

    # ── Step 2: Filter patients with useful data ──────────────────────────
    print("  [2/7] Selecting patient cohort...", flush=True)
    candidates = [
        p for p in patient_rows
        if p.get("apacheadmissiondx", "").strip()
        and p.get("age", "").strip()
        and p.get("unittype", "").strip()
    ]
    print(f"    → {len(candidates):,} qualifying candidates (have dx, age, unit type)")

    random.seed(SEED + 1)
    random.shuffle(candidates)
    if n_patients > 0:
        candidates = candidates[:n_patients * 3]

    selected_ids = {p["patientunitstayid"] for p in candidates}

    # ── Step 3: Load diagnoses ────────────────────────────────────────────
    print("  [3/7] Loading diagnoses...", flush=True)
    eicu_diagnoses = defaultdict(list)
    for row in read_gz_csv_progress(eicu_dir / "diagnosis.csv.gz", desc="diagnosis"):
        if row["patientunitstayid"] in selected_ids:
            eicu_diagnoses[row["patientunitstayid"]].append(row)
    print(f"    → Diagnoses for {len(eicu_diagnoses):,} patients")

    # ── Step 4: Load labs ─────────────────────────────────────────────────
    print("  [4/7] Loading lab results (full scan)...", flush=True)
    eicu_labs = defaultdict(list)
    for row in read_gz_csv_progress(eicu_dir / "lab.csv.gz", desc="lab"):
        pid = row["patientunitstayid"]
        if pid in selected_ids and len(eicu_labs[pid]) < labs_per_patient:
            eicu_labs[pid].append(row)

    pids_with_labs = {pid for pid, labs in eicu_labs.items() if len(labs) >= 3}
    print(f"    → Labs for {len(eicu_labs):,} patients ({len(pids_with_labs):,} with ≥3 labs)")

    # ── Step 5: Load medications ──────────────────────────────────────────
    print("  [5/7] Loading medications...", flush=True)
    eicu_meds = defaultdict(list)
    for row in read_gz_csv_progress(eicu_dir / "medication.csv.gz", desc="medication"):
        pid = row["patientunitstayid"]
        if pid in selected_ids and len(eicu_meds[pid]) < meds_per_patient:
            eicu_meds[pid].append(row)
    print(f"    → Medications for {len(eicu_meds):,} patients")

    # ── Step 6: Load vitals ───────────────────────────────────────────────
    print("  [6/7] Loading vital signs (full scan)...", flush=True)
    eicu_vitals = defaultdict(list)
    for row in read_gz_csv_progress(eicu_dir / "vitalPeriodic.csv.gz", desc="vitalPeriodic"):
        pid = row["patientunitstayid"]
        if pid in selected_ids and len(eicu_vitals[pid]) < vitals_per_patient:
            eicu_vitals[pid].append(row)
    print(f"    → Vitals for {len(eicu_vitals):,} patients")

    # ── Step 7: Load APACHE results ───────────────────────────────────────
    print("  [7/7] Loading APACHE scores...", flush=True)
    eicu_apache = {}
    for row in read_gz_csv(eicu_dir / "apachePatientResult.csv.gz"):
        pid = row["patientunitstayid"]
        if pid in selected_ids:
            eicu_apache[pid] = row
    print(f"    → APACHE scores for {len(eicu_apache):,} patients")

    # ── Build EHR records ─────────────────────────────────────────────────
    print("  Building EHR records...", flush=True)
    records = {}
    patient_index = defaultdict(list)
    final_count = 0
    max_count = n_patients if n_patients > 0 else len(candidates)

    for pat in candidates:
        if final_count >= max_count:
            break

        pid = pat["patientunitstayid"]
        if pid not in pids_with_labs:
            continue

        anon_patient_id = anon_id("EPAT", pat.get("uniquepid", pid))
        anon_hadm_id = anon_id("EHADM", pid)

        # Demographics
        age_str = pat.get("age", "70")
        if age_str == "> 89":
            age = 90
        else:
            age = safe_int(age_str, 70)

        gender = pat.get("gender", "")
        sex = "M" if gender == "Male" else ("F" if gender == "Female" else "O")
        ethnicity = pat.get("ethnicity", "")

        # Admission info
        admit_dx = pat.get("apacheadmissiondx", "")
        unit_type = pat.get("unittype", "MICU")
        discharge_status = pat.get("hospitaldischargestatus", "Alive")

        # Unit stay times
        admit_offset = safe_int(pat.get("hospitaladmitoffset", 0), 0)
        discharge_offset = safe_int(pat.get("hospitaldischargeoffset", 0), 0)
        los_hours = (discharge_offset - admit_offset) / 60 if discharge_offset else None
        los_days = round(los_hours / 24, 2) if los_hours else None

        # ICU stay
        unit_discharge_offset = safe_int(pat.get("unitdischargeoffset", 0), 0)
        icu_los_hours = round(unit_discharge_offset / 60, 1) if unit_discharge_offset else None
        anon_icu_id = anon_id("EICU", pid)

        # Diagnoses / ICD codes
        diag_list = eicu_diagnoses.get(pid, [])
        icd_codes = []
        for d in diag_list[:10]:
            code = d.get("icd9code", "").strip()
            if code:
                for c in code.split(","):
                    c = c.strip()
                    if c:
                        icd_codes.append(c)
        icd_codes = icd_codes[:10]

        # Lab events
        lab_records = []
        for lab in eicu_labs.get(pid, []):
            val = safe_float(lab.get("labresult"))
            if val is None:
                continue
            lab_name = lab.get("labname", "Unknown")
            lab_records.append({
                "itemid": lab.get("labid", ""),
                "label": lab_name,
                "value": val,
                "valueuom": lab.get("labmeasurenamesystem", ""),
                "flag": None,
                "ref_range_lower": None,
                "ref_range_upper": None,
                "charttime": f"offset_{lab.get('labresultoffset', '0')}",
            })

        # Vital signs
        vital_records = []
        for v in eicu_vitals.get(pid, []):
            vital_records.append({
                "charttime": f"offset_{v.get('observationoffset', '0')}",
                "heart_rate": safe_int(v.get("heartrate")),
                "sbp": safe_int(v.get("systemicsystolic")),
                "dbp": safe_int(v.get("systemicdiastolic")),
                "mean_bp": safe_int(v.get("systemicmean")),
                "resp_rate": safe_int(v.get("respiration")),
                "temperature": safe_float(v.get("temperature")),
                "spo2": safe_int(v.get("sao2")),
                "fio2": None,
                "gcs_total": None,
            })

        # Medications
        med_records = []
        for i, med in enumerate(eicu_meds.get(pid, [])):
            drug_name = med.get("drugname", "")
            med_records.append({
                "order_id": anon_id("EORD", f"{pid}_{i}"),
                "drug": drug_name,
                "drug_type": "MAIN",
                "dose_val": med.get("dosage", ""),
                "dose_unit": "",
                "route": med.get("routeadmin", ""),
                "frequency": med.get("frequency", ""),
                "start_time": f"offset_{med.get('drugstartoffset', '0')}",
                "end_time": f"offset_{med.get('drugstopoffset', '')}" if med.get("drugstopoffset") else None,
                "status": "completed",
            })

        # APACHE scores
        apache = eicu_apache.get(pid, {})
        clinical_scores = []
        if apache:
            apache_score = safe_float(apache.get("apachescore"))
            aps_score = safe_float(apache.get("acutephysiologyscore"))
            pred_mort = safe_float(apache.get("predictedhospitalmortality"))
            if apache_score is not None:
                clinical_scores.append({
                    "score_name": f"APACHE-{apache.get('apacheversion', 'IV')}",
                    "score_value": apache_score,
                    "interpretation": f"Acute Physiology Score: {aps_score}, Predicted Mortality: {pred_mort}",
                    "components": {
                        "acute_physiology": aps_score or 0,
                        "predicted_icu_mortality": safe_float(apache.get("predictedicumortality"), 0),
                        "predicted_hospital_mortality": pred_mort or 0,
                        "predicted_icu_los": safe_float(apache.get("predictediculos"), 0),
                    },
                    "calculated_at": "admission",
                })

        # Quality indicators
        mortality_risk = safe_float(apache.get("predictedhospitalmortality"), 0.1) if apache else 0.1
        actual_mortality = 1.0 if discharge_status == "Expired" else 0.0

        record = {
            "demographics": {
                "patient_id": anon_patient_id,
                "name": f"eICU Patient {final_count + 1}",
                "age": min(max(age, 18), 99),
                "sex": sex,
                "date_of_birth": "REDACTED",
                "ethnicity": ethnicity,
                "language": "English",
                "insurance": "",
                "marital_status": "",
            },
            "admission": {
                "hadm_id": anon_hadm_id,
                "patient_id": anon_patient_id,
                "admit_time": pat.get("hospitaladmittime24", ""),
                "discharge_time": pat.get("hospitaldischargetime24"),
                "admit_type": "emergency",
                "admit_location": pat.get("unitadmitsource", ""),
                "discharge_location": pat.get("hospitaldischargelocation"),
                "diagnosis_at_admission": admit_dx,
                "icd_codes": icd_codes,
                "drg_code": None,
                "los_days": los_days,
                "icu_stays": [anon_icu_id],
                "is_readmission": safe_int(pat.get("unitvisitnumber", 1), 1) > 1,
            },
            "icu_stays": [{
                "icustay_id": anon_icu_id,
                "hadm_id": anon_hadm_id,
                "patient_id": anon_patient_id,
                "icu_type": unit_type,
                "intime": pat.get("unitadmittime24", ""),
                "outtime": pat.get("unitdischargetime24"),
                "los_icu_hours": icu_los_hours,
            }],
            "lab_events": lab_records,
            "vital_events": vital_records,
            "medication_orders": med_records,
            "procedures": [],
            "discharge_summary": None,
            "clinical_scores": clinical_scores,
            "quality_indicators": {
                "readmission_risk": 0.3 if safe_int(pat.get("unitvisitnumber", 1), 1) > 1 else 0.15,
                "mortality_risk": mortality_risk,
                "expected_los": safe_float(apache.get("predictedhospitallos"), los_days or 5.0) if apache else (los_days or 5.0),
                "sepsis_flag": "sepsis" in admit_dx.lower(),
                "aki_stage": None,
                "notes": f"Actual outcome: {discharge_status}",
            },
            "prior_admissions": [],
        }

        records[anon_hadm_id] = record
        patient_index[anon_patient_id].append(anon_hadm_id)
        final_count += 1

        if final_count % 10000 == 0:
            print(f"    Built {final_count:,} records...", flush=True)

    print(f"  Built {len(records):,} eICU patient records")

    # ── Build lab reference ranges ────────────────────────────────────────
    lab_reference_ranges = {
        "creatinine": {"lower": 0.7, "upper": 1.3, "unit": "mg/dL"},
        "potassium": {"lower": 3.5, "upper": 5.0, "unit": "mEq/L"},
        "sodium": {"lower": 136, "upper": 145, "unit": "mEq/L"},
        "Hgb": {"lower": 12.0, "upper": 17.5, "unit": "g/dL"},
        "WBC x 1000": {"lower": 4.5, "upper": 11.0, "unit": "K/uL"},
        "platelets x 1000": {"lower": 150, "upper": 400, "unit": "K/uL"},
        "glucose": {"lower": 70, "upper": 100, "unit": "mg/dL"},
        "BUN": {"lower": 7, "upper": 20, "unit": "mg/dL"},
        "bicarbonate": {"lower": 22, "upper": 29, "unit": "mEq/L"},
        "chloride": {"lower": 96, "upper": 106, "unit": "mEq/L"},
        "lactate": {"lower": 0.5, "upper": 2.0, "unit": "mmol/L"},
        "troponin - I": {"lower": 0, "upper": 0.04, "unit": "ng/mL"},
        "PT - INR": {"lower": 0.8, "upper": 1.2, "unit": ""},
        "albumin": {"lower": 3.5, "upper": 5.5, "unit": "g/dL"},
        "total bilirubin": {"lower": 0.1, "upper": 1.2, "unit": "mg/dL"},
        "anion gap": {"lower": 8, "upper": 12, "unit": "mEq/L"},
    }

    db = {
        "records": records,
        "patient_index": dict(patient_index),
        "lab_reference_ranges": lab_reference_ranges,
        "icd_descriptions": {},
        "query_log": [],
    }

    tasks = _generate_eicu_tasks(records)

    elapsed = time.time() - build_t0
    print(f"  eICU build complete: {len(records):,} records, "
          f"{len(tasks):,} tasks in {elapsed:.0f}s")

    return db, tasks


def _generate_eicu_tasks(records: dict) -> list:
    """Generate evaluation tasks from eICU records (one task per patient)."""
    tasks = []
    rec_list = list(records.values())

    task_templates = [
        {
            "category": "icu_assessment",
            "difficulty": "hard",
            "template": "Patient {name} ({hadm_id}) is in the {icu_type}. Review their APACHE scores, vital signs, lab results, and medications. Assess the patient's current ICU severity and provide a comprehensive clinical assessment.",
            "expected_tools": ["get_patient_summary", "get_clinical_scores", "get_vital_signs", "get_lab_results"],
            "rubric_keys": ["APACHE score interpreted", "vital sign assessment", "lab abnormalities identified", "severity assessment provided"],
        },
        {
            "category": "vital_monitoring",
            "difficulty": "medium",
            "template": "Review the vital sign trends for ICU patient {name} ({hadm_id}). Identify any concerning patterns (hemodynamic instability, respiratory compromise, etc.) and suggest appropriate interventions.",
            "expected_tools": ["get_vital_signs", "detect_vital_alerts", "get_patient_summary"],
            "rubric_keys": ["vital trends reviewed", "abnormalities detected", "interventions suggested", "clinical context provided"],
        },
        {
            "category": "mortality_prediction",
            "difficulty": "hard",
            "template": "Based on all available clinical data for patient {name} ({hadm_id}), assess the mortality risk. Consider APACHE scores, lab trends, vital signs, and current interventions. How does the predicted mortality compare with the clinical picture?",
            "expected_tools": ["get_clinical_scores", "get_quality_indicators", "get_lab_results", "get_vital_signs"],
            "rubric_keys": ["APACHE mortality prediction discussed", "clinical risk factors identified", "lab risk markers noted", "overall prognosis assessment"],
        },
        {
            "category": "lab_trend_analysis",
            "difficulty": "medium",
            "template": "Analyze the complete lab panel for ICU patient {name} ({hadm_id}). Identify critical values, organ dysfunction markers, and trending abnormalities. Provide a systems-based assessment.",
            "expected_tools": ["get_lab_results", "get_lab_trend", "get_patient_summary"],
            "rubric_keys": ["critical values identified", "organ dysfunction assessed", "trends analyzed", "systems-based approach used"],
        },
        {
            "category": "medication_review",
            "difficulty": "medium",
            "template": "Review the ICU medication regimen for patient {name} ({hadm_id}), admitted for {dx}. Verify dosing appropriateness, check for interactions, and ensure all indicated therapies are ordered.",
            "expected_tools": ["get_medication_orders", "get_patient_summary", "get_lab_results"],
            "rubric_keys": ["medication list reviewed", "dosing verified", "interactions checked", "indicated therapies confirmed"],
        },
    ]

    for i, rec in enumerate(rec_list):
        template = task_templates[i % len(task_templates)]
        name = rec["demographics"]["name"]
        hadm_id = rec["admission"]["hadm_id"]
        dx = rec["admission"]["diagnosis_at_admission"]
        patient_id = rec["demographics"]["patient_id"]
        icu_type = rec["icu_stays"][0]["icu_type"] if rec["icu_stays"] else "ICU"

        task = {
            "id": f"eicu_ehr_{i+1:05d}",
            "domain": "ehr_management",
            "source": "eicu",
            "category": template["category"],
            "difficulty": template["difficulty"],
            "ticket": template["template"].format(
                name=name, hadm_id=hadm_id, dx=dx, icu_type=icu_type
            ),
            "patient_id": patient_id,
            "hadm_id": hadm_id,
            "expected_actions": [
                {"tool": t, "args": {"hadm_id": hadm_id}}
                for t in template["expected_tools"]
            ],
            "evaluation_criteria": {
                "actions": [
                    {"name": t, "arguments": {"hadm_id": hadm_id}, "compare_args": ["hadm_id"]}
                    for t in template["expected_tools"]
                ],
                "rubric_keys": template["rubric_keys"],
            },
            "split": "test",
        }
        tasks.append(task)

    print(f"  Generated {len(tasks):,} eICU evaluation tasks")
    return tasks


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build EHR Benchmark from MIMIC-III & eICU (full dataset by default)"
    )
    parser.add_argument("--mimic-dir", type=str, default=str(DEFAULT_MIMIC_DIR))
    parser.add_argument("--eicu-dir", type=str, default=str(DEFAULT_EICU_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-patients", type=int, default=0,
                        help="Number of patients per dataset (0 = ALL qualifying patients)")
    parser.add_argument("--labs-per-patient", type=int, default=100,
                        help="Max lab events to keep per patient admission")
    parser.add_argument("--meds-per-patient", type=int, default=30,
                        help="Max medications per patient")
    parser.add_argument("--vitals-per-patient", type=int, default=60,
                        help="Max vital readings per patient (eICU only)")
    parser.add_argument("--procs-per-patient", type=int, default=20,
                        help="Max procedures per patient (MIMIC-III only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_t0 = time.time()

    # ── Build MIMIC-III benchmark ─────────────────────────────────────────
    mimic_dir = Path(args.mimic_dir)
    if mimic_dir.exists():
        mimic_db, mimic_tasks = build_mimic_benchmark(
            mimic_dir,
            n_patients=args.n_patients,
            labs_per_patient=args.labs_per_patient,
            meds_per_patient=args.meds_per_patient,
            procs_per_patient=args.procs_per_patient,
        )
        mimic_out = output_dir / "mimic_iii_bench.json.gz"
        print(f"\n  Writing MIMIC-III benchmark...", flush=True)
        write_gzip_json({"db": mimic_db, "tasks": mimic_tasks}, mimic_out)
        print(f"  → {len(mimic_db['records']):,} records, {len(mimic_tasks):,} tasks")
    else:
        print(f"  [SKIP] MIMIC-III dir not found: {mimic_dir}")

    # ── Build eICU benchmark ──────────────────────────────────────────────
    eicu_dir = Path(args.eicu_dir)
    if eicu_dir.exists():
        eicu_db, eicu_tasks = build_eicu_benchmark(
            eicu_dir,
            n_patients=args.n_patients,
            labs_per_patient=args.labs_per_patient,
            meds_per_patient=args.meds_per_patient,
            vitals_per_patient=args.vitals_per_patient,
        )
        eicu_out = output_dir / "eicu_bench.json.gz"
        print(f"\n  Writing eICU benchmark...", flush=True)
        write_gzip_json({"db": eicu_db, "tasks": eicu_tasks}, eicu_out)
        print(f"  → {len(eicu_db['records']):,} records, {len(eicu_tasks):,} tasks")
    else:
        print(f"  [SKIP] eICU dir not found: {eicu_dir}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  EHR Benchmark Build Complete! ({total_elapsed:.0f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
