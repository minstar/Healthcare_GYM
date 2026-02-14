#!/usr/bin/env python3
"""Build BM25 FTS5 indexes for medical knowledge sources.

Creates SQLite FTS5 full-text search indexes for:
1. MedCPT Evidence Passages (581K entries from PubMed/PMC)
2. Biomedical Instructions (122K entries from Self-BioRAG)
3. Biomedical Generator passages (84K entries with retrieval tokens)
4. Critic evaluation data (16K entries)

Output:
    databases/medical_knowledge_fts.sqlite
        - Table: evidence_fts  (MedCPT evidence passages)
        - Table: instruction_fts (Biomedical QA instruction/answer pairs)
        - Table: passages_fts (Combined searchable passages)

Usage:
    python scripts/build_medical_fts_index.py
    python scripts/build_medical_fts_index.py --output databases/medical_knowledge_fts.sqlite
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_fts_database(db_path: str) -> sqlite3.Connection:
    """Create SQLite database with FTS5 tables."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-256000;")  # 256MB cache

    # Main searchable passages table (unified across all sources)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
            doc_id,
            source,
            title,
            content,
            category,
            dataset_name,
            tokenize='porter unicode61'
        )
    """)

    # Evidence-specific table (MedCPT top-10 evidence)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
            doc_id,
            question,
            evidence,
            dataset_name,
            tokenize='porter unicode61'
        )
    """)

    # Instruction/QA table
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS instruction_fts USING fts5(
            doc_id,
            instruction,
            answer,
            topic,
            dataset_name,
            tokenize='porter unicode61'
        )
    """)

    return conn


def index_medcpt_evidence(conn: sqlite3.Connection, base_dir: Path):
    """Index MedCPT top-10 evidence passages (JSONL format, 581K entries)."""
    filepath = base_dir / "databases" / "retriever" / "medcpt_top10_evidence_createret.json"
    if not filepath.exists():
        print(f"  SKIP: {filepath} not found")
        return 0

    print(f"  Indexing MedCPT evidence: {filepath}")
    count = 0
    batch = []
    batch_size = 5000

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = item.get("q_id", f"medcpt_{line_num}")
            question = item.get("question", item.get("instruction", ""))
            evidence = item.get("evidence", "")
            dataset_name = item.get("dataset_name", "medcpt")

            if not evidence:
                continue

            # Insert into evidence_fts
            batch.append((doc_id, question, evidence, dataset_name))

            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO evidence_fts(doc_id, question, evidence, dataset_name) VALUES (?,?,?,?)",
                    batch,
                )
                count += len(batch)
                batch = []
                if count % 50000 == 0:
                    print(f"    ... {count:,} evidence passages indexed")

    if batch:
        conn.executemany(
            "INSERT INTO evidence_fts(doc_id, question, evidence, dataset_name) VALUES (?,?,?,?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    print(f"    Total MedCPT evidence: {count:,}")
    return count


def index_medcpt_passages_to_unified(conn: sqlite3.Connection, base_dir: Path):
    """Index MedCPT evidence as unified passages for cross-source search."""
    filepath = base_dir / "databases" / "retriever" / "medcpt_top10_evidence_createret.json"
    if not filepath.exists():
        return 0

    print(f"  Indexing MedCPT → unified passages")
    count = 0
    batch = []
    batch_size = 5000

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            evidence = item.get("evidence", "")
            if not evidence or len(evidence) < 20:
                continue

            doc_id = f"medcpt_{item.get('q_id', line_num)}"
            # Extract a title from the first sentence or question context
            question = item.get("question", "")
            title = question[:120] if question else evidence[:80]
            dataset_name = item.get("dataset_name", "medcpt")

            batch.append((
                doc_id, "medcpt_evidence", title, evidence,
                "medical_evidence", dataset_name,
            ))

            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
                    "VALUES (?,?,?,?,?,?)",
                    batch,
                )
                count += len(batch)
                batch = []

    if batch:
        conn.executemany(
            "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
            "VALUES (?,?,?,?,?,?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    print(f"    Unified passages from MedCPT: {count:,}")
    return count


def index_biomedical_instructions(conn: sqlite3.Connection, base_dir: Path):
    """Index biomedical instruction data (122K entries)."""
    filepath = base_dir / "databases" / "instruction" / "all_biomedical_instruction.json"
    if not filepath.exists():
        print(f"  SKIP: {filepath} not found")
        return 0

    print(f"  Indexing biomedical instructions: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    batch = []
    batch_size = 5000

    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        topic = item.get("topic", "")
        dataset_name = item.get("dataset_name", "biomedical")
        doc_id = item.get("id", f"instr_{count}")

        # Combine instruction + input for the question
        full_instruction = f"{instruction} {input_text}".strip()
        if not full_instruction:
            continue

        # instruction_fts
        batch.append((doc_id, full_instruction, output_text, topic, dataset_name))

        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT INTO instruction_fts(doc_id, instruction, answer, topic, dataset_name) "
                "VALUES (?,?,?,?,?)",
                batch,
            )
            count += len(batch)
            batch = []

    if batch:
        conn.executemany(
            "INSERT INTO instruction_fts(doc_id, instruction, answer, topic, dataset_name) "
            "VALUES (?,?,?,?,?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    print(f"    Total instructions: {count:,}")
    return count


def index_instructions_to_unified(conn: sqlite3.Connection, base_dir: Path):
    """Index instruction QA pairs as unified passages."""
    filepath = base_dir / "databases" / "instruction" / "all_biomedical_instruction.json"
    if not filepath.exists():
        return 0

    print(f"  Indexing instructions → unified passages")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    batch = []
    batch_size = 5000

    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        doc_id = item.get("id", f"instr_{count}")
        dataset_name = item.get("dataset_name", "biomedical")
        topic = item.get("topic", "")

        # Use answer as content (this is the knowledge)
        content = output_text.strip()
        if not content or len(content) < 20:
            continue

        title = instruction[:120]
        category = topic or "biomedical_qa"

        batch.append((
            doc_id, "biomedical_instruction", title, content,
            category, dataset_name,
        ))

        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
                "VALUES (?,?,?,?,?,?)",
                batch,
            )
            count += len(batch)
            batch = []

    if batch:
        conn.executemany(
            "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
            "VALUES (?,?,?,?,?,?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    print(f"    Unified passages from instructions: {count:,}")
    return count


def index_generator_passages(conn: sqlite3.Connection, base_dir: Path):
    """Index generator retrieval token passages (84K entries)."""
    filepath = base_dir / "databases" / "generator" / "bio_generator_train.json"
    if not filepath.exists():
        print(f"  SKIP: {filepath} not found")
        return 0

    print(f"  Indexing generator passages: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    batch = []
    batch_size = 5000

    for item in data:
        output = item.get("output", "")
        doc_id = item.get("id", f"gen_{count}")
        dataset_name = item.get("dataset_name", "generator")
        instruction = item.get("instruction", "")

        # Extract paragraph content from [Retrieval]<paragraph>...</paragraph> tags
        paragraphs = re.findall(r"<paragraph>(.*?)</paragraph>", output, re.DOTALL)
        if not paragraphs:
            # Use raw output as content
            content = output.strip()
            if len(content) < 30:
                continue
            paragraphs = [content]

        for pidx, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 20:
                continue

            p_id = f"{doc_id}_p{pidx}"
            title = instruction[:120] if instruction else para[:80]

            batch.append((
                p_id, "generator_retrieval", title, para,
                "biomedical_knowledge", dataset_name,
            ))

            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
                    "VALUES (?,?,?,?,?,?)",
                    batch,
                )
                count += len(batch)
                batch = []

    if batch:
        conn.executemany(
            "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
            "VALUES (?,?,?,?,?,?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    print(f"    Unified passages from generator: {count:,}")
    return count


def index_medinstruct(conn: sqlite3.Connection, base_dir: Path):
    """Index MedInstruct-52k as unified passages."""
    filepath = base_dir / "databases" / "instruction" / "MedInstruct-52k.json"
    if not filepath.exists():
        print(f"  SKIP: {filepath} not found")
        return 0

    print(f"  Indexing MedInstruct-52k: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    batch = []
    batch_size = 5000

    for idx, item in enumerate(data):
        instruction = item.get("instruction", "")
        output_text = item.get("output", "")

        content = output_text.strip()
        if not content or len(content) < 30:
            continue

        doc_id = f"medinstr_{idx}"
        title = instruction[:120] if instruction else content[:80]

        batch.append((
            doc_id, "medinstruct", title, content,
            "medical_instruction", "MedInstruct-52k",
        ))

        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
                "VALUES (?,?,?,?,?,?)",
                batch,
            )
            count += len(batch)
            batch = []

    if batch:
        conn.executemany(
            "INSERT INTO passages_fts(doc_id, source, title, content, category, dataset_name) "
            "VALUES (?,?,?,?,?,?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    print(f"    Unified passages from MedInstruct-52k: {count:,}")
    return count


def optimize_database(conn: sqlite3.Connection):
    """Optimize FTS5 indexes for faster queries."""
    print("  Optimizing FTS5 indexes...")
    conn.execute("INSERT INTO passages_fts(passages_fts) VALUES('optimize')")
    conn.execute("INSERT INTO evidence_fts(evidence_fts) VALUES('optimize')")
    conn.execute("INSERT INTO instruction_fts(instruction_fts) VALUES('optimize')")
    conn.commit()
    print("  Optimization complete")


def print_stats(conn: sqlite3.Connection):
    """Print index statistics."""
    print("\n=== FTS5 Index Statistics ===")
    for table in ["passages_fts", "evidence_fts", "instruction_fts"]:
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table}: {count:,} entries")

    # Source breakdown for passages_fts
    print("\n  passages_fts by source:")
    cur = conn.execute(
        "SELECT source, COUNT(*) as cnt FROM passages_fts GROUP BY source ORDER BY cnt DESC"
    )
    for source, cnt in cur.fetchall():
        print(f"    {source}: {cnt:,}")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 FTS5 indexes for medical knowledge")
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "databases" / "medical_knowledge_fts.sqlite"),
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--base-dir", type=str,
        default=str(PROJECT_ROOT),
        help="Project base directory",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_path = args.output

    print("=" * 60)
    print("Building Medical Knowledge BM25 FTS5 Index")
    print("=" * 60)
    print(f"Base dir: {base_dir}")
    print(f"Output: {output_path}")
    print()

    # Remove existing database to rebuild
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing database: {output_path}")

    start = time.time()
    conn = create_fts_database(output_path)

    total = 0

    # 1. MedCPT Evidence (evidence_fts + passages_fts)
    print("\n[1/5] MedCPT Evidence Passages")
    total += index_medcpt_evidence(conn, base_dir)
    total += index_medcpt_passages_to_unified(conn, base_dir)

    # 2. Biomedical Instructions (instruction_fts + passages_fts)
    print("\n[2/5] Biomedical Instructions")
    total += index_biomedical_instructions(conn, base_dir)
    total += index_instructions_to_unified(conn, base_dir)

    # 3. Generator retrieval passages (passages_fts only)
    print("\n[3/5] Generator Retrieval Passages")
    total += index_generator_passages(conn, base_dir)

    # 4. MedInstruct-52k (passages_fts only)
    print("\n[4/5] MedInstruct-52k")
    total += index_medinstruct(conn, base_dir)

    # 5. Optimize
    print("\n[5/5] Optimizing...")
    optimize_database(conn)

    # Stats
    print_stats(conn)

    elapsed = time.time() - start
    db_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nTotal entries indexed: {total:,}")
    print(f"Database size: {db_size:.1f} MB")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\nDone! Index saved to: {output_path}")

    conn.close()


if __name__ == "__main__":
    main()
