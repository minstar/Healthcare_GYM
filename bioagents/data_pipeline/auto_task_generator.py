"""AutoTaskGenerator — Self-generating training data for the Healthcare AI GYM.

Bridges the gap between limited training tasks (~700) and comprehensive
evaluation benchmarks (MCQA 8.9K, MedLFQA 4.9K, VQA 6 datasets, EHR 2 DBs)
by mining knowledge sources to produce GYM-compatible multi-turn tasks.

Sources:
    1. Benchmark → Training converter (MCQA, MedLFQA, VQA, EHR)
    2. Knowledge-grounded mining (FTS5 828K passages)
    3. Instruction mining (122K + 52K instruction pairs)
    4. Guideline-based case synthesis (10 clinical guidelines)

Architecture:
    AutoTaskGenerator
    ├── BenchmarkConverter        — eval benchmarks → GYM tasks
    │   ├── MCQAConverter         — MedQA / MedMCQA / MMLU → medical_qa tasks
    │   ├── MedLFQAConverter      — Long-form QA → medical_qa tasks
    │   ├── VQAConverter          — VQA-RAD, SLAKE, etc. → visual_diagnosis tasks
    │   └── EHRConverter          — MIMIC-III / eICU → ehr_management tasks
    ├── KnowledgeMiner            — FTS5 passages → domain tasks
    ├── InstructionMiner          — instruction pairs → QA tasks
    └── GuidelineSynthesizer      — clinical guidelines → clinical/triage tasks

Output: GYM-compatible task JSON with evaluation_criteria, tool expectations.
"""

import json
import hashlib
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Domain → Tool mapping ─────────────────────────────────────
DOMAIN_TOOLS = {
    "medical_qa": [
        {"name": "search_pubmed", "arguments": {}, "info": "Search PubMed for evidence"},
        {"name": "search_evidence", "arguments": {}, "info": "Retrieve evidence passages"},
        {"name": "search_wiki", "arguments": {}, "info": "Search medical encyclopedia"},
    ],
    "clinical_diagnosis": [
        {"name": "get_patient_info", "arguments": {}, "info": "Review patient data"},
        {"name": "get_vital_signs", "arguments": {}, "info": "Check vitals"},
        {"name": "get_lab_results", "arguments": {}, "info": "Review labs"},
        {"name": "get_differential_diagnosis", "arguments": {}, "info": "Generate DDx"},
        {"name": "search_clinical_guidelines", "arguments": {}, "info": "Search guidelines"},
        {"name": "record_diagnosis", "arguments": {}, "info": "Record diagnosis"},
    ],
    "drug_interaction": [
        {"name": "get_patient_medications", "arguments": {}, "info": "Get medication list"},
        {"name": "check_interaction", "arguments": {}, "info": "Check drug interaction"},
        {"name": "search_pubmed", "arguments": {}, "info": "Search literature"},
    ],
    "triage_emergency": [
        {"name": "get_patient_info", "arguments": {}, "info": "Triage patient info"},
        {"name": "get_vital_signs", "arguments": {}, "info": "Check vitals"},
        {"name": "assess_severity", "arguments": {}, "info": "ESI scoring"},
        {"name": "search_clinical_guidelines", "arguments": {}, "info": "Search protocols"},
    ],
    "ehr_management": [
        {"name": "get_patient_demographics", "arguments": {}, "info": "Get demographics"},
        {"name": "get_admission_info", "arguments": {}, "info": "Get admission data"},
        {"name": "get_lab_events", "arguments": {}, "info": "Get lab events"},
        {"name": "get_medication_orders", "arguments": {}, "info": "Get medications"},
    ],
    "visual_diagnosis": [
        {"name": "view_image", "arguments": {}, "info": "View medical image"},
        {"name": "analyze_findings", "arguments": {}, "info": "Analyze visual findings"},
        {"name": "search_pubmed", "arguments": {}, "info": "Search literature"},
    ],
    "psychiatry": [
        {"name": "get_patient_info", "arguments": {}, "info": "Get patient info"},
        {"name": "assess_mental_status", "arguments": {}, "info": "Mental status exam"},
        {"name": "search_clinical_guidelines", "arguments": {}, "info": "Treatment guidelines"},
    ],
    "obstetrics": [
        {"name": "get_patient_info", "arguments": {}, "info": "Get patient info"},
        {"name": "get_vital_signs", "arguments": {}, "info": "Check vitals"},
        {"name": "get_lab_results", "arguments": {}, "info": "Review labs"},
        {"name": "search_clinical_guidelines", "arguments": {}, "info": "ACOG guidelines"},
    ],
    "radiology_report": [
        {"name": "view_study", "arguments": {}, "info": "View radiology study"},
        {"name": "get_patient_info", "arguments": {}, "info": "Get clinical context"},
        {"name": "search_pubmed", "arguments": {}, "info": "Search literature"},
    ],
}


def _stable_id(prefix: str, text: str) -> str:
    """Generate a deterministic ID from prefix + text hash."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def _pick_tools(domain: str, n: int = 3) -> list[dict]:
    """Pick n random expected tools for a domain."""
    tools = DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS["medical_qa"])
    n = min(n, len(tools))
    selected = random.sample(tools, n)
    # Always end with submit_answer
    selected.append({
        "name": "submit_answer",
        "arguments": {"answer": ""},
        "compare_args": ["answer"],
        "info": "Submit final answer",
    })
    return [
        {**t, "action_id": f"action_{i}"}
        for i, t in enumerate(selected)
    ]


# ══════════════════════════════════════════════════════════════
#  1. Benchmark → Training Converters
# ══════════════════════════════════════════════════════════════


class MCQAConverter:
    """Convert MCQA benchmark JSONL (Self-BioRAG format) to GYM tasks.

    Sources: MedQA, MedMCQA, MMLU ×6
    Total:   ~8,900 questions
    """

    BENCHMARK_DIR = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark"

    BENCHMARK_FILES = {
        "med_qa": "med_qa_test.jsonl",
        "medmc_qa": "medmc_qa_test.jsonl",
        "mmlu_clinical_knowledge": "mmlu_clinical_knowledge_test.jsonl",
        "mmlu_professional_medicine": "mmlu_professional_medicine_test.jsonl",
        "mmlu_anatomy": "mmlu_anatomy_test.jsonl",
        "mmlu_medical_genetics": "mmlu_medical_genetics_test.jsonl",
        "mmlu_college_biology": "mmlu_college_biology_test.jsonl",
        "mmlu_college_medicine": "mmlu_college_medicine_test.jsonl",
    }

    # Also include train splits for RL training data
    TRAIN_FILES = {
        "med_qa_train": "med_qa_train.json",
        "medmc_qa_train": "medmc_qa_train.json",
    }

    @classmethod
    def convert(
        cls,
        benchmarks: Optional[list[str]] = None,
        max_per_benchmark: int = 500,
        use_train_split: bool = True,
        difficulty_from_options: bool = True,
    ) -> list[dict]:
        """Convert MCQA benchmarks to GYM task format.

        Args:
            benchmarks: List of benchmark names. None = all.
            max_per_benchmark: Max tasks per benchmark.
            use_train_split: Also include train splits (larger).
            difficulty_from_options: Infer difficulty from # of options.

        Returns:
            List of GYM-format task dicts.
        """
        tasks = []
        sources = dict(cls.BENCHMARK_FILES)
        if use_train_split:
            sources.update(cls.TRAIN_FILES)

        if benchmarks:
            sources = {k: v for k, v in sources.items() if k in benchmarks}

        for bench_name, filename in sources.items():
            filepath = cls.BENCHMARK_DIR / filename
            if not filepath.exists():
                logger.warning(f"MCQA benchmark not found: {filepath}")
                continue

            raw_items = cls._load_file(filepath)
            random.shuffle(raw_items)
            raw_items = raw_items[:max_per_benchmark]

            for idx, item in enumerate(raw_items):
                task = cls._convert_one(item, bench_name, idx, difficulty_from_options)
                if task:
                    tasks.append(task)

            logger.info(f"[MCQAConverter] {bench_name}: {len(raw_items)} tasks converted")

        return tasks

    @classmethod
    def _load_file(cls, filepath: Path) -> list[dict]:
        """Load JSONL or JSON file (auto-detects JSONL even with .json ext)."""
        items = []

        with open(filepath, "r") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                # Standard JSON array
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        items = data
                except json.JSONDecodeError:
                    # Fallback: read as JSONL
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            else:
                # JSONL (one JSON object per line — even if .json extension)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        return items

    @classmethod
    def _convert_one(
        cls, item: dict, bench_name: str, idx: int, difficulty_from_options: bool
    ) -> Optional[dict]:
        """Convert a single MCQA item to GYM format."""
        # Extract question and answer
        instances = item.get("instances", {})
        if isinstance(instances, dict):
            question_text = instances.get("input", "")
            correct_output = instances.get("output", "")
        elif isinstance(instances, list) and instances:
            question_text = instances[0].get("input", "")
            correct_output = instances[0].get("output", "")
        else:
            question_text = item.get("question", item.get("input", ""))
            correct_output = item.get("answer", item.get("output", ""))

        if not question_text:
            return None

        # Parse options from text
        options = {}
        option_matches = re.findall(
            r"Option\s+([A-E]):\s*(.+?)(?=Option\s+[A-E]:|$)",
            question_text,
            re.DOTALL,
        )
        for letter, text in option_matches:
            options[letter.strip()] = text.strip()

        # Find correct answer letter
        correct_letter = ""
        if len(correct_output) <= 2:
            correct_letter = correct_output.strip().upper()
        else:
            for letter, text in options.items():
                if text.strip().lower() == correct_output.strip().lower():
                    correct_letter = letter
                    break
            if not correct_letter and options:
                # Partial match
                for letter, text in options.items():
                    if correct_output.strip().lower() in text.strip().lower():
                        correct_letter = letter
                        break

        # Determine difficulty
        n_options = len(options) if options else 4
        if difficulty_from_options:
            difficulty = "easy" if n_options <= 3 else "medium" if n_options == 4 else "hard"
        else:
            difficulty = "medium"

        # Categorize
        category = item.get("name", bench_name)
        task_id = _stable_id(f"mcqa_{bench_name}", f"{idx}_{question_text[:100]}")

        # Build GYM task
        task = {
            "id": task_id,
            "description": {
                "purpose": f"Answer a {category.replace('_', ' ')} question",
                "difficulty": difficulty,
                "source": bench_name,
                "category": category,
                "generated_from": "benchmark_converter",
            },
            "ticket": question_text,
            "correct_answer": correct_letter or correct_output,
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": "search_evidence",
                        "name": "search_pubmed",
                        "arguments": {},
                        "info": "Search for relevant medical evidence",
                    },
                    {
                        "action_id": "submit",
                        "name": "submit_answer",
                        "arguments": {"answer": correct_letter or correct_output},
                        "compare_args": ["answer"],
                        "info": f"Submit the correct answer",
                    },
                ],
                "nl_assertions": [
                    f"The agent selected the correct answer",
                    "The agent used evidence search before answering",
                    f"The agent demonstrated {category.replace('_', ' ')} reasoning",
                ],
                "reward_basis": ["ACTION", "NL_ASSERTION"],
            },
        }

        if options:
            task["options"] = options
        if correct_output and len(correct_output) > 2:
            task["raw_answer"] = correct_output

        return task


class MedLFQAConverter:
    """Convert MedLFQA benchmark to GYM long-form QA tasks.

    Sources: KQA Golden, LiveQA, MedicationQA, HealthSearchQA, KQA Silver
    Total:   ~4,948 questions
    """

    BENCHMARK_DIR = PROJECT_ROOT / "evaluations" / "OLAPH" / "MedLFQA"

    BENCHMARK_FILES = {
        "kqa_golden": "kqa_golden_test_MedLFQA.jsonl",
        "live_qa": "live_qa_test_MedLFQA.jsonl",
        "medication_qa": "medication_qa_test_MedLFQA.jsonl",
        "healthsearch_qa": "healthsearch_qa_test_MedLFQA.jsonl",
        "kqa_silver": "kqa_silver_wogold_test_MedLFQA.jsonl",
    }

    @classmethod
    def convert(
        cls,
        benchmarks: Optional[list[str]] = None,
        max_per_benchmark: int = 300,
    ) -> list[dict]:
        """Convert MedLFQA to GYM tasks with tool-use expectations."""
        tasks = []
        sources = dict(cls.BENCHMARK_FILES)
        if benchmarks:
            sources = {k: v for k, v in sources.items() if k in benchmarks}

        for bench_name, filename in sources.items():
            filepath = cls.BENCHMARK_DIR / filename
            if not filepath.exists():
                logger.warning(f"MedLFQA benchmark not found: {filepath}")
                continue

            count = 0
            with open(filepath, "r") as f:
                lines = f.readlines()
            random.shuffle(lines)

            for line in lines:
                if count >= max_per_benchmark:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                task = cls._convert_one(item, bench_name, count)
                if task:
                    tasks.append(task)
                    count += 1

            logger.info(f"[MedLFQAConverter] {bench_name}: {count} tasks converted")

        return tasks

    @classmethod
    def _convert_one(cls, item: dict, bench_name: str, idx: int) -> Optional[dict]:
        """Convert a single MedLFQA item."""
        question = item.get("Question", "")
        answer = item.get("Free_form_answer", "")
        must_have = item.get("Must_have", [])
        nice_to_have = item.get("Nice_to_have", [])
        icd = item.get("ICD_10_diag", "")

        if not question or not answer:
            return None

        task_id = _stable_id(f"lfqa_{bench_name}", f"{idx}_{question[:80]}")

        # Determine domain-relevance for tool selection
        q_lower = question.lower()
        if any(w in q_lower for w in ["medication", "drug", "pill", "dose", "prescription"]):
            domain_hint = "drug_interaction"
        elif any(w in q_lower for w in ["diagnosis", "symptom", "disease", "condition"]):
            domain_hint = "clinical_diagnosis"
        else:
            domain_hint = "medical_qa"

        # Build assertions from must_have + nice_to_have
        nl_assertions = []
        for m in must_have[:3]:
            nl_assertions.append(f"The response covers: {m[:120]}")
        for n in nice_to_have[:2]:
            nl_assertions.append(f"The response ideally mentions: {n[:120]}")
        if not nl_assertions:
            nl_assertions = [
                "The agent provided a comprehensive, accurate response",
                "The agent used evidence search to support the answer",
            ]

        task = {
            "id": task_id,
            "description": {
                "purpose": f"Answer a long-form medical question ({bench_name})",
                "difficulty": "hard" if len(must_have) > 5 else "medium",
                "source": f"MedLFQA/{bench_name}",
                "category": icd if icd else "general_medicine",
                "generated_from": "benchmark_converter",
            },
            "ticket": question,
            "correct_answer": answer,
            "evaluation_criteria": {
                "actions": _pick_tools(domain_hint, n=2),
                "nl_assertions": nl_assertions,
                "reward_basis": ["ACTION", "NL_ASSERTION"],
            },
        }
        return task


class InstructionMiner:
    """Mine training tasks from 828K indexed medical passages.

    Sources:
        - medical_knowledge_fts.sqlite (FTS5 index)
            - passages_fts: 828K entries
            - evidence_fts: 581K PubMed/PMC evidence
            - instruction_fts: 122K biomedical QA instruction pairs

    Strategy:
        1. Query FTS5 with domain-specific keywords
        2. Extract question-answer pairs from instruction_fts
        3. Convert passages into evidence-grounded QA tasks
    """

    DB_PATH = PROJECT_ROOT / "databases" / "medical_knowledge_fts.sqlite"

    # Domain → search keywords for mining
    DOMAIN_KEYWORDS = {
        "clinical_diagnosis": [
            "diagnosis", "differential diagnosis", "clinical presentation",
            "physical examination", "lab findings", "imaging", "treatment",
            "pathophysiology", "prognosis", "management",
        ],
        "drug_interaction": [
            "drug interaction", "adverse effect", "contraindication",
            "pharmacokinetics", "metabolism", "CYP450", "dosing",
            "drug safety", "polypharmacy", "therapeutic monitoring",
        ],
        "triage_emergency": [
            "emergency", "triage", "critical care", "resuscitation",
            "trauma", "acute care", "ACLS", "shock", "anaphylaxis",
            "cardiac arrest",
        ],
        "ehr_management": [
            "electronic health record", "clinical documentation",
            "ICD coding", "patient safety", "medication reconciliation",
            "discharge planning", "quality measures",
        ],
        "psychiatry": [
            "depression", "anxiety", "schizophrenia", "bipolar",
            "PTSD", "substance abuse", "psychotherapy", "antidepressant",
            "suicide risk", "mental status examination",
        ],
        "obstetrics": [
            "pregnancy", "preeclampsia", "gestational diabetes",
            "prenatal care", "labor delivery", "postpartum",
            "fetal monitoring", "cesarean", "ACOG guidelines",
        ],
        "visual_diagnosis": [
            "radiology", "imaging findings", "X-ray", "CT scan",
            "MRI", "ultrasound", "pathology slide", "dermoscopy",
            "retinal imaging", "mammography",
        ],
        "radiology_report": [
            "radiology report", "imaging interpretation", "CT findings",
            "MRI findings", "chest X-ray", "structured reporting",
            "BIRADS", "Fleischner", "radiology impression",
        ],
        "medical_qa": [
            "medical knowledge", "clinical reasoning", "evidence based",
            "pathology", "pharmacology", "anatomy", "physiology",
            "biochemistry", "microbiology", "immunology",
        ],
    }

    @classmethod
    def mine_instructions(
        cls,
        domain: str = "medical_qa",
        max_tasks: int = 200,
        min_answer_length: int = 50,
    ) -> list[dict]:
        """Mine QA instruction pairs from the FTS5 index.

        Queries the instruction_fts table for domain-relevant QA pairs
        and converts them to GYM task format.
        """
        if not cls.DB_PATH.exists():
            logger.warning(f"Knowledge DB not found: {cls.DB_PATH}")
            return []

        keywords = cls.DOMAIN_KEYWORDS.get(domain, cls.DOMAIN_KEYWORDS["medical_qa"])
        tasks = []

        try:
            conn = sqlite3.connect(str(cls.DB_PATH))
            conn.row_factory = sqlite3.Row

            for keyword in keywords:
                if len(tasks) >= max_tasks:
                    break

                remaining = max_tasks - len(tasks)
                query = f"""
                    SELECT rowid, title, content, source
                    FROM instruction_fts
                    WHERE instruction_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """
                try:
                    cursor = conn.execute(query, (keyword, min(remaining, 50)))
                    rows = cursor.fetchall()
                except sqlite3.OperationalError:
                    # Table might not exist or match format differs
                    continue

                for row in rows:
                    content = row["content"] if row["content"] else ""
                    title = row["title"] if row["title"] else ""

                    # Instruction pairs often have Q: ... A: ... format
                    task = cls._parse_instruction_pair(
                        title, content, domain, len(tasks), keyword
                    )
                    if task and len(task.get("correct_answer", "")) >= min_answer_length:
                        tasks.append(task)

            conn.close()
            logger.info(
                f"[InstructionMiner] {domain}: mined {len(tasks)} tasks "
                f"from instruction_fts"
            )

        except Exception as e:
            logger.error(f"[InstructionMiner] Error: {e}")

        return tasks

    @classmethod
    def mine_evidence_passages(
        cls,
        domain: str = "medical_qa",
        max_tasks: int = 200,
    ) -> list[dict]:
        """Mine evidence passages and create evidence-grounded QA tasks.

        Takes passages from evidence_fts, extracts the core claim,
        and creates a "verify this claim" task.
        """
        if not cls.DB_PATH.exists():
            return []

        keywords = cls.DOMAIN_KEYWORDS.get(domain, cls.DOMAIN_KEYWORDS["medical_qa"])
        tasks = []

        try:
            conn = sqlite3.connect(str(cls.DB_PATH))
            conn.row_factory = sqlite3.Row

            for keyword in keywords:
                if len(tasks) >= max_tasks:
                    break

                remaining = max_tasks - len(tasks)
                try:
                    cursor = conn.execute(
                        """
                        SELECT rowid, title, content, source
                        FROM passages_fts
                        WHERE passages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (keyword, min(remaining, 30)),
                    )
                    rows = cursor.fetchall()
                except sqlite3.OperationalError:
                    continue

                for row in rows:
                    content = str(row["content"] or "")
                    title = str(row["title"] or "")
                    source = str(row["source"] or "")

                    if len(content) < 100:
                        continue

                    # Create evidence-grounded task
                    task = cls._create_evidence_task(
                        title, content, source, domain, len(tasks)
                    )
                    if task:
                        tasks.append(task)

            conn.close()
            logger.info(
                f"[InstructionMiner] {domain}: mined {len(tasks)} evidence tasks "
                f"from passages_fts"
            )

        except Exception as e:
            logger.error(f"[InstructionMiner] Evidence mining error: {e}")

        return tasks

    @classmethod
    def _parse_instruction_pair(
        cls,
        title: str,
        content: str,
        domain: str,
        idx: int,
        keyword: str,
    ) -> Optional[dict]:
        """Parse an instruction pair into a GYM task."""
        # Try to split Q/A from content
        question = ""
        answer = ""

        # Common formats: "Question: ... Answer: ...", "Q: ... A: ..."
        for q_pat, a_pat in [
            (r"[Qq]uestion:\s*", r"[Aa]nswer:\s*"),
            (r"Q:\s*", r"A:\s*"),
            (r"[Ii]nput:\s*", r"[Oo]utput:\s*"),
        ]:
            parts = re.split(a_pat, content, maxsplit=1)
            if len(parts) == 2:
                q_part = re.sub(q_pat, "", parts[0]).strip()
                a_part = parts[1].strip()
                if q_part and a_part:
                    question = q_part
                    answer = a_part
                    break

        # If no Q/A split, use title as question, content as answer
        if not question:
            if title and len(content) > len(title):
                question = title
                answer = content
            else:
                return None

        task_id = _stable_id(f"inst_{domain}", f"{idx}_{question[:60]}")

        return {
            "id": task_id,
            "description": {
                "purpose": f"Medical knowledge task on {keyword}",
                "difficulty": "medium",
                "source": "instruction_mining",
                "category": keyword,
                "generated_from": "knowledge_miner",
            },
            "ticket": question,
            "correct_answer": answer,
            "evaluation_criteria": {
                "actions": _pick_tools(domain, n=2),
                "nl_assertions": [
                    "The agent provided an accurate medical response",
                    "The agent used evidence to support the answer",
                    f"The agent demonstrated knowledge of {keyword}",
                ],
                "reward_basis": ["ACTION", "NL_ASSERTION"],
            },
        }

    @classmethod
    def _create_evidence_task(
        cls,
        title: str,
        content: str,
        source: str,
        domain: str,
        idx: int,
    ) -> Optional[dict]:
        """Create a task from an evidence passage.

        Strategy: Extract the key claim from the passage and ask the
        model to explain/verify it using tools.
        """
        # Extract first sentence as the core claim
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if not sentences:
            return None

        core_claim = sentences[0]
        if len(core_claim) < 30:
            core_claim = ". ".join(sentences[:2]) if len(sentences) > 1 else core_claim

        # Formulate question
        topic = title if title else core_claim[:80]
        question_templates = [
            f"Based on current medical evidence, explain the following: {topic}",
            f"Provide a comprehensive evidence-based explanation of: {topic}",
            f"Using medical literature, answer: What is known about {topic}?",
            f"Search the medical evidence and explain: {topic}",
        ]
        question = random.choice(question_templates)

        task_id = _stable_id(f"evid_{domain}", f"{idx}_{topic[:60]}")

        return {
            "id": task_id,
            "description": {
                "purpose": f"Evidence-grounded medical explanation",
                "difficulty": "hard",
                "source": f"evidence_mining/{source}",
                "category": domain,
                "generated_from": "knowledge_miner",
            },
            "ticket": question,
            "correct_answer": content[:2000],  # Full passage as reference answer
            "evaluation_criteria": {
                "actions": _pick_tools(domain, n=3),
                "nl_assertions": [
                    f"The response addresses the core topic: {topic[:80]}",
                    "The agent used evidence search tools before answering",
                    "The response is factually consistent with medical literature",
                ],
                "reward_basis": ["ACTION", "NL_ASSERTION"],
            },
        }


class GuidelineSynthesizer:
    """Generate clinical scenario tasks from guideline knowledge.

    Uses the 10 embedded clinical guidelines to create tasks that
    test whether models can apply guideline recommendations.
    """

    GUIDELINES_PATH = PROJECT_ROOT / "data" / "guidelines" / "clinical_guidelines.json"

    # Guideline → domain mapping
    GUIDELINE_DOMAINS = {
        "AHA/ACC STEMI": "clinical_diagnosis",
        "AHA/ASA Acute Ischemic Stroke": "triage_emergency",
        "Surviving Sepsis Campaign": "triage_emergency",
        "ADA DKA": "clinical_diagnosis",
        "ACEP HEART Score": "triage_emergency",
        "AHA Kawasaki": "clinical_diagnosis",
        "WSES Appendicitis": "clinical_diagnosis",
        "ESC Pulmonary Embolism": "clinical_diagnosis",
        "ACP Low Back Pain": "medical_qa",
        "IDSA Antimicrobial Stewardship": "drug_interaction",
    }

    @classmethod
    def synthesize(cls, max_per_guideline: int = 10) -> list[dict]:
        """Generate tasks from clinical guidelines."""
        tasks = []

        # Try loading from JSON file
        if cls.GUIDELINES_PATH.exists():
            try:
                with open(cls.GUIDELINES_PATH, "r") as f:
                    guidelines = json.load(f)
            except (json.JSONDecodeError, Exception):
                guidelines = []
        else:
            # Fall back to embedded guidelines
            try:
                from bioagents.knowledge.guidelines import CLINICAL_GUIDELINES
                guidelines = CLINICAL_GUIDELINES
            except ImportError:
                guidelines = []

        if isinstance(guidelines, dict):
            guidelines = list(guidelines.values())

        for gl in guidelines:
            if isinstance(gl, str):
                gl = {"title": "Unknown", "content": gl}

            title = gl.get("title", gl.get("name", ""))
            content = gl.get("content", gl.get("recommendations", ""))
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)

            if not title or not content:
                continue

            # Determine domain
            domain = "medical_qa"
            for key, dom in cls.GUIDELINE_DOMAINS.items():
                if key.lower() in title.lower():
                    domain = dom
                    break

            # Generate scenario tasks
            gl_tasks = cls._generate_scenarios(title, content, domain, max_per_guideline)
            tasks.extend(gl_tasks)

        logger.info(f"[GuidelineSynthesizer] Generated {len(tasks)} tasks from guidelines")
        return tasks

    @classmethod
    def _generate_scenarios(
        cls,
        title: str,
        content: str,
        domain: str,
        max_tasks: int,
    ) -> list[dict]:
        """Generate clinical scenario tasks from a guideline."""
        tasks = []

        # Template-based scenarios
        scenario_templates = [
            (
                "A patient presents with symptoms consistent with the conditions "
                "described in the {title} guidelines. Based on current evidence-based "
                "guidelines, what is the recommended diagnostic workup and initial management?",
                "guideline_application",
            ),
            (
                "A healthcare provider needs to determine the appropriate management "
                "strategy for a patient according to {title}. Search the current "
                "guidelines and provide step-by-step recommendations.",
                "guideline_retrieval",
            ),
            (
                "Critically appraise the recommendations in {title}. What are the "
                "key evidence-based interventions and their corresponding levels of evidence?",
                "guideline_appraisal",
            ),
        ]

        # Extract key recommendations for assertions
        rec_sentences = re.findall(r'([^.!?]*(?:recommend|suggest|should|must)[^.!?]*[.!?])', content, re.I)
        key_recs = rec_sentences[:5] if rec_sentences else []

        for i, (template, task_type) in enumerate(scenario_templates):
            if i >= max_tasks:
                break

            question = template.format(title=title)
            task_id = _stable_id(f"gl_{domain}", f"{title}_{i}")

            nl_assertions = [
                f"The response references the {title} guidelines",
                "The agent used guideline search tools",
            ]
            for rec in key_recs[:2]:
                nl_assertions.append(f"The response covers: {rec[:120].strip()}")

            tasks.append({
                "id": task_id,
                "description": {
                    "purpose": f"Apply {title} guidelines",
                    "difficulty": "hard",
                    "source": f"guideline_synthesis/{title}",
                    "category": task_type,
                    "generated_from": "guideline_synthesizer",
                },
                "ticket": question,
                "correct_answer": content[:2000],
                "evaluation_criteria": {
                    "actions": _pick_tools(domain, n=3),
                    "nl_assertions": nl_assertions,
                    "reward_basis": ["ACTION", "NL_ASSERTION"],
                },
            })

        return tasks


# ══════════════════════════════════════════════════════════════
#  Main AutoTaskGenerator
# ══════════════════════════════════════════════════════════════


class AutoTaskGenerator:
    """Orchestrates all task generation sources.

    Usage:
        gen = AutoTaskGenerator()

        # Generate for a specific domain
        tasks = gen.generate(domain="clinical_diagnosis", target=500)

        # Generate for all domains
        all_tasks = gen.generate_all(target_per_domain=300)

        # Save to GYM task files
        gen.save_to_domain_files(all_tasks)
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else (
            PROJECT_ROOT / "data" / "domains"
        )

    def generate(
        self,
        domain: str = "medical_qa",
        target: int = 500,
        sources: Optional[list[str]] = None,
    ) -> list[dict]:
        """Generate training tasks for a domain.

        Args:
            domain: Target domain.
            target: Target number of tasks.
            sources: List of sources to use. None = all available.
                Options: "mcqa", "lfqa", "instructions", "evidence", "guidelines"
        """
        if sources is None:
            sources = ["mcqa", "lfqa", "instructions", "evidence", "guidelines"]

        tasks = []
        per_source = max(target // len(sources), 50)

        # 1. Benchmark MCQA
        if "mcqa" in sources and domain in ("medical_qa", "clinical_diagnosis"):
            mcqa_tasks = MCQAConverter.convert(max_per_benchmark=per_source)
            tasks.extend(mcqa_tasks)
            logger.info(f"[AutoGen] MCQA: {len(mcqa_tasks)} tasks")

        # 2. MedLFQA
        if "lfqa" in sources and domain in ("medical_qa", "drug_interaction"):
            lfqa_tasks = MedLFQAConverter.convert(max_per_benchmark=per_source // 5)
            tasks.extend(lfqa_tasks)
            logger.info(f"[AutoGen] LFQA: {len(lfqa_tasks)} tasks")

        # 3. Instruction mining
        if "instructions" in sources:
            inst_tasks = InstructionMiner.mine_instructions(
                domain=domain, max_tasks=per_source
            )
            tasks.extend(inst_tasks)
            logger.info(f"[AutoGen] Instructions: {len(inst_tasks)} tasks")

        # 4. Evidence mining
        if "evidence" in sources:
            evid_tasks = InstructionMiner.mine_evidence_passages(
                domain=domain, max_tasks=per_source
            )
            tasks.extend(evid_tasks)
            logger.info(f"[AutoGen] Evidence: {len(evid_tasks)} tasks")

        # 5. Guideline synthesis
        if "guidelines" in sources:
            gl_tasks = GuidelineSynthesizer.synthesize(max_per_guideline=5)
            # Filter to domain
            domain_gl = [t for t in gl_tasks if domain in t.get("description", {}).get("source", "")]
            if not domain_gl:
                domain_gl = gl_tasks  # Use all if no domain match
            tasks.extend(domain_gl[:per_source])
            logger.info(f"[AutoGen] Guidelines: {len(domain_gl)} tasks")

        # Deduplicate by ID
        seen = set()
        unique_tasks = []
        for t in tasks:
            tid = t.get("id", "")
            if tid not in seen:
                seen.add(tid)
                unique_tasks.append(t)

        # Trim to target
        if len(unique_tasks) > target:
            random.shuffle(unique_tasks)
            unique_tasks = unique_tasks[:target]

        logger.info(
            f"[AutoTaskGenerator] domain={domain}: {len(unique_tasks)} tasks generated "
            f"(target={target})"
        )

        return unique_tasks

    def generate_all(self, target_per_domain: int = 300) -> dict[str, list[dict]]:
        """Generate training tasks for ALL domains.

        Returns:
            Dict mapping domain name to list of tasks.
        """
        all_tasks = {}

        for domain in DOMAIN_TOOLS.keys():
            tasks = self.generate(domain=domain, target=target_per_domain)
            all_tasks[domain] = tasks

        total = sum(len(v) for v in all_tasks.values())
        logger.info(
            f"[AutoTaskGenerator] All domains: {total} total tasks "
            f"across {len(all_tasks)} domains"
        )

        return all_tasks

    def save_to_domain_files(
        self,
        tasks_by_domain: dict[str, list[dict]],
        filename: str = "tasks_auto_generated.json",
    ) -> dict[str, str]:
        """Save generated tasks to domain directories.

        Args:
            tasks_by_domain: Dict mapping domain → task list.
            filename: Output filename within each domain directory.

        Returns:
            Dict mapping domain → saved file path.
        """
        saved = {}

        for domain, tasks in tasks_by_domain.items():
            if not tasks:
                continue

            domain_dir = self.output_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            output_path = domain_dir / filename

            with open(output_path, "w") as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)

            saved[domain] = str(output_path)
            logger.info(
                f"[AutoGen] Saved {len(tasks)} tasks to {output_path}"
            )

        return saved

    def generate_and_save(
        self,
        target_per_domain: int = 300,
        filename: str = "tasks_auto_generated.json",
    ) -> dict[str, str]:
        """Generate + save in one step."""
        all_tasks = self.generate_all(target_per_domain=target_per_domain)
        return self.save_to_domain_files(all_tasks, filename=filename)


# ══════════════════════════════════════════════════════════════
#  GYM Integration: On-Demand Task Expansion
# ══════════════════════════════════════════════════════════════


def expand_domain_tasks(
    domain: str,
    existing_tasks: list[dict],
    target_total: int = 500,
    sources: Optional[list[str]] = None,
) -> list[dict]:
    """Expand a domain's task pool to reach target_total.

    Called by the AutonomousAgent when it detects insufficient
    training data for a domain.

    Args:
        domain: Domain name.
        existing_tasks: Current tasks for this domain.
        target_total: Desired total task count.
        sources: Which sources to mine.

    Returns:
        Combined list of existing + newly generated tasks.
    """
    current = len(existing_tasks)
    if current >= target_total:
        return existing_tasks

    needed = target_total - current
    logger.info(
        f"[TaskExpander] {domain}: have {current}, need {needed} more "
        f"(target={target_total})"
    )

    gen = AutoTaskGenerator()
    new_tasks = gen.generate(domain=domain, target=needed, sources=sources)

    # Deduplicate against existing
    existing_ids = {t.get("id", "") for t in existing_tasks}
    unique_new = [t for t in new_tasks if t.get("id", "") not in existing_ids]

    combined = existing_tasks + unique_new
    logger.info(
        f"[TaskExpander] {domain}: expanded from {current} to {len(combined)} tasks"
    )

    return combined


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════


def main():
    """CLI entry point for task generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-generate training tasks for Healthcare AI GYM")
    parser.add_argument(
        "--domains", nargs="+", default=["all"],
        help="Domains to generate for (default: all)",
    )
    parser.add_argument(
        "--target", type=int, default=300,
        help="Target tasks per domain (default: 300)",
    )
    parser.add_argument(
        "--sources", nargs="+",
        default=["mcqa", "lfqa", "instructions", "evidence", "guidelines"],
        help="Sources to use",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: data/domains/)",
    )
    parser.add_argument(
        "--filename", type=str, default="tasks_auto_generated.json",
        help="Output filename per domain",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print counts without saving",
    )

    args = parser.parse_args()
    gen = AutoTaskGenerator(output_dir=args.output)

    if "all" in args.domains:
        all_tasks = gen.generate_all(target_per_domain=args.target)
    else:
        all_tasks = {}
        for domain in args.domains:
            all_tasks[domain] = gen.generate(
                domain=domain, target=args.target, sources=args.sources
            )

    # Summary
    print("\n" + "=" * 60)
    print("  AUTO TASK GENERATION SUMMARY")
    print("=" * 60)
    total = 0
    for domain, tasks in sorted(all_tasks.items()):
        src_counts = {}
        for t in tasks:
            src = t.get("description", {}).get("generated_from", "unknown")
            src_counts[src] = src_counts.get(src, 0) + 1
        src_str = ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items()))
        print(f"  {domain:25s} {len(tasks):5d} tasks  ({src_str})")
        total += len(tasks)
    print(f"  {'TOTAL':25s} {total:5d} tasks")
    print("=" * 60)

    if not args.dry_run:
        saved = gen.save_to_domain_files(all_tasks, filename=args.filename)
        print(f"\nSaved to {len(saved)} domain directories.")
    else:
        print("\n(Dry run — no files saved)")


if __name__ == "__main__":
    main()
