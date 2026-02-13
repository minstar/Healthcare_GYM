#!/usr/bin/env python3
"""Generate GYM-ready tasks.json + db.json from benchmark data.

Creates:
  data/domains/medical_qa/tasks.json     — 50 curated medical QA tasks
  data/domains/medical_qa/db.json        — knowledge base (articles + evidence + wiki)
  data/domains/medical_qa/split_tasks.json — train/test split

Usage:
    cd BIOAgents
    python scripts/generate_gym_data.py --num_tasks 50
    python scripts/generate_gym_data.py --num_tasks 200 --include_train
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark"
OUTPUT_DIR = PROJECT_ROOT / "data" / "domains" / "medical_qa"

# ─────────────────────────────────────────────
# 1. Raw data loaders
# ─────────────────────────────────────────────


def load_jsonl(path: Path, max_lines: int = None) -> list[dict]:
    """Load a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_lines and len(records) >= max_lines:
                break
    return records


def parse_options(text: str) -> list[dict]:
    """Extract options from question text."""
    matches = re.findall(r"Option\s+([A-E]):\s*(.*?)(?=\nOption\s+[A-E]:|$)", text, re.DOTALL)
    return [{"label": l.strip(), "text": t.strip()} for l, t in matches]


def extract_correct_label(answer_text: str, options: list[dict]) -> str:
    """Match answer text to option label."""
    # Single letter
    if len(answer_text.strip()) == 1 and answer_text.strip().upper() in "ABCDE":
        return answer_text.strip().upper()
    answer_lower = answer_text.strip().lower()
    for opt in options:
        if opt["text"].strip().lower() == answer_lower:
            return opt["label"]
    for opt in options:
        if answer_lower in opt["text"].strip().lower() or opt["text"].strip().lower() in answer_lower:
            return opt["label"]
    return ""


def clean_question(text: str) -> str:
    """Extract just the question (remove options, instruction prefix)."""
    q_match = re.search(r"QUESTION:\s*(.*?)(?=\nOption\s+[A-E]:)", text, re.DOTALL)
    if q_match:
        return q_match.group(1).strip()
    parts = text.split("Option")[0].strip()
    if parts.startswith("QUESTION:"):
        parts = parts[9:].strip()
    return parts


def detect_category(question: str) -> str:
    """Heuristic category detection from question text."""
    q = question.lower()
    category_keywords = {
        "pharmacology": ["drug", "medication", "mechanism", "receptor", "inhibit", "agonist", "antagonist",
                         "side effect", "adverse", "pharmacokinetic", "dosage", "antibiotic", "chemotherapy"],
        "pathology": ["biopsy", "histology", "microscop", "tissue", "patholog", "neoplasm", "carcinoma",
                      "tumor", "lesion", "malignant", "benign", "specimen"],
        "anatomy": ["nerve", "artery", "vein", "muscle", "bone", "ligament", "anatomic",
                     "foramen", "fossa", "innervat"],
        "physiology": ["hormone", "electrolyte", "membrane", "potential", "channel",
                       "secretion", "absorption", "receptor", "signal"],
        "microbiology": ["bacteria", "virus", "fungal", "parasite", "gram-positive", "gram-negative",
                         "culture", "antibiotic resistance", "organism"],
        "biochemistry": ["enzyme", "substrate", "metabol", "glycolysis", "krebs", "amino acid",
                         "protein", "nucleotide", "lipid"],
        "ethics": ["ethics", "autonomy", "beneficence", "disclosure", "consent",
                   "confidential", "competence"],
        "surgery": ["surgical", "incision", "resection", "operative", "postoperative",
                    "laparoscop", "cholecystectomy"],
        "pediatrics": ["newborn", "neonate", "infant", "child", "pediatric",
                       "vaccination", "developmental"],
        "cardiology": ["cardiac", "heart", "echocardiogram", "murmur", "arrhythmia",
                       "coronary", "myocardial", "atrial", "ventricular"],
        "neurology": ["brain", "spinal", "seizure", "stroke", "neuropathy",
                      "consciousness", "reflex", "cranial nerve"],
        "psychiatry": ["depression", "anxiety", "schizophrenia", "psychosis", "bipolar",
                       "antidepressant", "delusion", "hallucination"],
        "obstetrics_gynecology": ["pregnan", "gestational", "fetus", "uterine", "ovarian",
                                  "menstrual", "cervical", "trimester"],
        "nephrology": ["kidney", "renal", "glomerular", "creatinine", "dialysis",
                       "proteinuria", "nephr"],
        "pulmonology": ["lung", "pulmonary", "pneumonia", "bronch", "asthma",
                        "respiratory", "dyspnea", "alveol"],
        "gastroenterology": ["liver", "hepat", "gastric", "intestin", "colon",
                             "pancreat", "esophag", "bowel"],
        "endocrinology": ["thyroid", "diabetes", "insulin", "cortisol", "adrenal",
                          "pituitary", "growth hormone"],
        "dermatology": ["skin", "rash", "dermat", "lesion", "pruritus",
                        "erythema", "psoriasis"],
        "hematology": ["blood", "anemia", "leukemia", "platelet", "coagulation",
                       "lymphoma", "hemoglobin"],
        "immunology": ["immune", "antibod", "antigen", "autoimmun", "allerg",
                       "complement", "lymphocyte"],
        "ophthalmology": ["eye", "vision", "retina", "conjunctiv", "glaucoma",
                          "cataract", "optic"],
        "orthopedics": ["fracture", "joint", "bone", "osteo", "arthritis",
                        "tendon", "ligament"],
    }
    scores = {}
    for cat, keywords in category_keywords.items():
        scores[cat] = sum(1 for kw in keywords if kw in q)
    if not scores or max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def detect_difficulty(question: str, options: list[dict]) -> str:
    """Heuristic difficulty level."""
    q = question.lower()
    # Long clinical vignettes are harder
    word_count = len(question.split())
    has_lab_values = bool(re.search(r'\d+\s*(mg|g|mmol|mEq|mm|dB|%|/mm|/L|U/L)', question))
    num_options = len(options)

    if word_count > 150 and has_lab_values:
        return "hard"
    elif word_count > 80 or has_lab_values:
        return "medium"
    return "easy"


# ─────────────────────────────────────────────
# 2. Build tasks.json
# ─────────────────────────────────────────────


def build_task(record: dict, source: str, task_idx: int) -> dict | None:
    """Convert one raw record into a BIOAgents task dict."""
    # Support multiple data formats:
    # Format 1 (test): {"instances": {"input": "...", "output": "..."}}
    # Format 2 (train_gpt4): {"question": "...", "answer": "...", "instruction": "...", "instances": {...}}
    instances = record.get("instances", {})
    if isinstance(instances, dict):
        question_text = instances.get("input", "")
        answer_text = instances.get("output", "")
    else:
        question_text = ""
        answer_text = ""

    # Fallback: direct question/answer fields (train_gpt4 format)
    if not question_text:
        question_text = record.get("question", "")
        # Reconstruct full question with options if needed
        instruction = record.get("instruction", "")
        if question_text and instruction:
            question_text = f"QUESTION: {question_text}"
    if not answer_text:
        answer_text = record.get("answer", "")

    if not question_text or not answer_text:
        return None

    options = parse_options(question_text)
    if not options:
        return None

    correct_label = extract_correct_label(answer_text, options)
    if not correct_label:
        return None

    question = clean_question(question_text)
    category = detect_category(question)
    difficulty = detect_difficulty(question, options)

    task_id = f"{source.lower()}_{task_idx:05d}"

    # Build ticket
    ticket_parts = [f"QUESTION: {question}", ""]
    for opt in options:
        ticket_parts.append(f"Option {opt['label']}: {opt['text']}")
    ticket = "\n".join(ticket_parts)

    # Determine which search tools are expected
    expected_search = "search_pubmed" if difficulty != "easy" else "retrieve_evidence"

    return {
        "id": task_id,
        "description": {
            "purpose": f"Answer a {source} medical question on {category}",
            "difficulty": difficulty,
            "source": source,
            "category": category,
            "key_challenges": [
                f"Requires {category} knowledge",
                "Evidence-based reasoning needed" if difficulty != "easy" else "Basic recall question",
            ],
        },
        "ticket": ticket,
        "correct_answer": correct_label,
        "options": {opt["label"]: opt["text"] for opt in options},
        "raw_question": question,
        "raw_answer": answer_text,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "search_evidence",
                    "name": expected_search,
                    "arguments": {},
                    "info": f"Search for evidence relevant to {category}",
                },
                {
                    "action_id": "submit",
                    "name": "submit_answer",
                    "arguments": {"answer": correct_label},
                    "compare_args": ["answer"],
                    "info": f"Submit the correct answer: {correct_label}",
                },
            ],
            "nl_assertions": [
                f"The agent selected option {correct_label} as the correct answer",
                "The agent used evidence search tools before answering",
                f"The agent demonstrated {category} reasoning",
            ],
            "reward_basis": ["ACTION", "NL_ASSERTION"],
        },
    }


def select_balanced_tasks(
    all_records: list[tuple[dict, str]],
    num_tasks: int,
    seed: int = 42,
) -> list[dict]:
    """Select a balanced subset of tasks across sources and categories."""
    random.seed(seed)

    # Convert all
    tasks = []
    for idx, (record, source) in enumerate(all_records):
        t = build_task(record, source, idx)
        if t:
            tasks.append(t)

    if len(tasks) <= num_tasks:
        return tasks

    # Group by (source, category)
    groups = defaultdict(list)
    for t in tasks:
        key = (t["description"]["source"], t["description"]["category"])
        groups[key].append(t)

    # Select proportionally
    selected = []
    keys = list(groups.keys())
    random.shuffle(keys)

    per_group = max(1, num_tasks // len(keys))
    remaining = num_tasks

    for key in keys:
        pool = groups[key]
        random.shuffle(pool)
        take = min(per_group, len(pool), remaining)
        selected.extend(pool[:take])
        remaining -= take
        if remaining <= 0:
            break

    # Fill remaining from largest groups
    if remaining > 0:
        all_remaining = []
        for key in keys:
            pool = groups[key]
            used_ids = {t["id"] for t in selected}
            all_remaining.extend(t for t in pool if t["id"] not in used_ids)
        random.shuffle(all_remaining)
        selected.extend(all_remaining[:remaining])

    # Re-number task IDs
    for i, t in enumerate(selected):
        src = t["description"]["source"].lower()
        t["id"] = f"{src}_{i:05d}"

    return selected


# ─────────────────────────────────────────────
# 3. Build db.json (knowledge base)
# ─────────────────────────────────────────────


def build_db_from_evidence(
    evidence_records: list[dict],
    tasks: list[dict],
) -> dict:
    """Build the medical QA knowledge base from evidence data."""
    articles = {}
    evidence_passages = {}
    wiki_entries = {}
    questions = {}

    # Track which evidence is relevant to which task
    task_question_map = {}
    for task in tasks:
        q_hash = hashlib.md5(task["raw_question"][:100].encode()).hexdigest()[:8]
        task_question_map[task["raw_question"][:100].lower()] = task["id"]

    seen_pmids = set()
    passage_counter = 0

    for record in evidence_records:
        instances = record.get("instances", {})
        question_text = instances.get("input", "")
        ctxs = record.get("ctxs", [])

        if not ctxs:
            continue

        # Match to task
        q_clean = clean_question(question_text)[:100].lower()
        matched_task_id = task_question_map.get(q_clean, "")

        relevant_passage_ids = []

        for ctx in ctxs[:5]:  # Top 5 evidence passages per question
            pmid = ctx.get("pmid", "")
            title = ctx.get("title", "Untitled")
            text = ctx.get("text", "")
            journal = ctx.get("journal_title", "")
            year = ctx.get("PubDate_year", 2024)
            score = ctx.get("score", 0.0)

            if not text:
                continue

            # Add as article if has PMID and not seen
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                articles[pmid] = {
                    "pmid": pmid,
                    "title": title,
                    "abstract": text[:1000],
                    "authors": [],
                    "journal": journal,
                    "year": int(year) if year else 2024,
                    "keywords": _extract_keywords(title + " " + text),
                    "doi": "",
                    "sections": {},
                }

            # Always add as evidence passage
            passage_id = f"EP_{passage_counter:05d}"
            passage_counter += 1
            category = detect_category(text)

            evidence_passages[passage_id] = {
                "passage_id": passage_id,
                "source": f"PubMed:{pmid}" if pmid else "MedCPT",
                "title": title,
                "text": text,
                "relevance_score": float(score) if score else 0.0,
                "category": category,
            }
            relevant_passage_ids.append(passage_id)

        # Build question entry
        if matched_task_id:
            task = next((t for t in tasks if t["id"] == matched_task_id), None)
            if task:
                q_id = f"Q_{matched_task_id}"
                questions[q_id] = {
                    "question_id": q_id,
                    "source": task["description"]["source"],
                    "question": task["raw_question"],
                    "options": [
                        {"label": k, "text": v}
                        for k, v in task.get("options", {}).items()
                    ],
                    "correct_answer": task["correct_answer"],
                    "explanation": task.get("raw_answer", ""),
                    "category": task["description"]["category"],
                    "difficulty": task["description"]["difficulty"],
                    "relevant_evidence_ids": relevant_passage_ids,
                }

    # Add some wiki entries from high-scoring passages
    wiki_counter = 0
    for pid, passage in sorted(
        evidence_passages.items(), key=lambda x: x[1]["relevance_score"], reverse=True
    )[:50]:
        if len(passage["text"]) > 200:
            entry_id = f"WIKI_{wiki_counter:04d}"
            wiki_entries[entry_id] = {
                "entry_id": entry_id,
                "title": passage["title"],
                "url": "",
                "summary": passage["text"][:300],
                "full_text": passage["text"],
                "categories": [passage["category"]],
                "related_entries": [],
            }
            wiki_counter += 1

    return {
        "articles": articles,
        "evidence_passages": evidence_passages,
        "wiki_entries": wiki_entries,
        "questions": questions,
        "search_log": [],
    }


def _extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Extract medical keywords from text."""
    # Simple keyword extraction
    stop_words = {
        "the", "a", "an", "is", "was", "were", "are", "of", "in", "to",
        "for", "with", "and", "or", "on", "at", "by", "from", "that",
        "this", "it", "as", "be", "has", "had", "not", "but", "which",
        "can", "may", "will", "also", "been", "more", "than", "other",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq = Counter(w for w in words if w not in stop_words)
    return [w for w, _ in freq.most_common(max_keywords)]


# ─────────────────────────────────────────────
# 4. Main pipeline
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate GYM data from benchmarks")
    parser.add_argument("--num_tasks", type=int, default=50, help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_train", action="store_true", help="Also generate train split tasks")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="Ratio of test tasks")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== BIOAgents GYM Data Generator ===")
    print(f"Output: {output_dir}")
    print(f"Target tasks: {args.num_tasks}")
    print()

    # ---- Load raw data ----
    # CRITICAL: Use TRAIN files only to prevent data contamination!
    # Test benchmark files (med_qa_test.jsonl, mmlu_test.jsonl, etc.) must NEVER
    # be used for GYM task generation or SFT training data, as they are used
    # for external benchmark evaluation (MedQA, MMLU, MedMCQA scores in paper).
    print("Loading benchmark TRAIN data (contamination-safe)...")
    all_records = []

    # ─── CONTAMINATION GUARD (see BUGLOG.md BUG-001) ───
    # Runtime check: block any accidental loading from test files
    _forbidden_test_files = list(BENCHMARK_DIR.glob("*_test*"))
    def _assert_not_test(path: Path):
        assert path not in _forbidden_test_files, (
            f"CONTAMINATION BLOCKED: Attempted to load test file '{path.name}' for training data! "
            f"See BUGLOG.md BUG-001. Use *_train* files only."
        )
    # ────────────────────────────────────────────────────

    # MedQA — TRAIN only
    for medqa_name in ["med_qa_train_gpt4.jsonl", "med_qa_5options_train.json"]:
        medqa_path = BENCHMARK_DIR / medqa_name
        if medqa_path.exists():
            _assert_not_test(medqa_path)
            records = load_jsonl(medqa_path)
            all_records.extend((r, "MedQA") for r in records)
            print(f"  MedQA train ({medqa_name}): {len(records)} records")
            break
    else:
        print("  WARNING: No MedQA train file found!")

    # MedMCQA — TRAIN only
    for medmcqa_name in ["medmc_qa_train_gpt4.jsonl", "medmc_qa_train.json"]:
        medmcqa_path = BENCHMARK_DIR / medmcqa_name
        if medmcqa_path.exists():
            _assert_not_test(medmcqa_path)
            raw_records = load_jsonl(medmcqa_path)
            # Handle nested list format (some files store [record, record, ...] per line)
            records = []
            for r in raw_records:
                if isinstance(r, list):
                    records.extend(r)
                else:
                    records.append(r)
            all_records.extend((r, "MedMCQA") for r in records)
            print(f"  MedMCQA train ({medmcqa_name}): {len(records)} records")
            break
    else:
        print("  WARNING: No MedMCQA train file found!")

    # MMLU — NO train file available; skip for GYM tasks (eval-only benchmark)
    # MMLU is evaluated separately via run_full_baseline_eval.py
    print("  MMLU: Skipped (evaluation-only benchmark, no train split available)")

    print(f"  Total raw records: {len(all_records)}")
    if not all_records:
        print("  ERROR: No train data loaded! Check benchmark directory.")
        sys.exit(1)
    print()

    # ---- Select balanced tasks ----
    print(f"Selecting {args.num_tasks} balanced tasks...")
    tasks = select_balanced_tasks(all_records, args.num_tasks, seed=args.seed)
    print(f"  Selected: {len(tasks)} tasks")

    # Print stats
    source_counts = Counter(t["description"]["source"] for t in tasks)
    category_counts = Counter(t["description"]["category"] for t in tasks)
    difficulty_counts = Counter(t["description"]["difficulty"] for t in tasks)
    answer_counts = Counter(t["correct_answer"] for t in tasks)

    print(f"\n  By source: {dict(source_counts)}")
    print(f"  By category (top 10): {dict(category_counts.most_common(10))}")
    print(f"  By difficulty: {dict(difficulty_counts)}")
    print(f"  Answer distribution: {dict(answer_counts)}")

    # ---- Build knowledge base (db.json) ----
    print("\nBuilding knowledge base from evidence files...")
    evidence_records = []
    for evidence_file in [
        "med_qa_evidence.json",
        "medmc_qa_evidence.json",
        "mmlu_evidence.json",
    ]:
        epath = BENCHMARK_DIR / evidence_file
        if epath.exists():
            recs = load_jsonl(epath, max_lines=2000)
            evidence_records.extend(recs)
            print(f"  {evidence_file}: {len(recs)} records")

    db = build_db_from_evidence(evidence_records, tasks)
    print(f"\n  Knowledge Base Stats:")
    print(f"    Articles: {len(db['articles'])}")
    print(f"    Evidence passages: {len(db['evidence_passages'])}")
    print(f"    Wiki entries: {len(db['wiki_entries'])}")
    print(f"    Questions: {len(db['questions'])}")

    # ---- Create train/test split ----
    random.seed(args.seed)
    task_ids = [t["id"] for t in tasks]
    random.shuffle(task_ids)
    split_idx = int(len(task_ids) * (1 - args.test_ratio))

    split_tasks = {
        "train": task_ids[:split_idx],
        "test": task_ids[split_idx:],
        "base": task_ids[:10],  # Quick evaluation set
    }
    print(f"\n  Splits: train={len(split_tasks['train'])}, test={len(split_tasks['test'])}, base={len(split_tasks['base'])}")

    # ---- Save outputs ----
    print("\nSaving outputs...")

    # tasks.json
    tasks_path = output_dir / "tasks.json"
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"  tasks.json: {len(tasks)} tasks → {tasks_path}")

    # db.json
    db_path = output_dir / "db.json"
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"  db.json: {size_mb:.1f} MB → {db_path}")

    # split_tasks.json
    split_path = output_dir / "split_tasks.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_tasks, f, indent=2, ensure_ascii=False)
    print(f"  split_tasks.json → {split_path}")

    print(f"\n✅ Done! Generated {len(tasks)} tasks with {len(db['evidence_passages'])} evidence passages.")
    print(f"   Run evaluation: python scripts/run_medqa_experiment.py --num_tasks 10")


if __name__ == "__main__":
    main()
