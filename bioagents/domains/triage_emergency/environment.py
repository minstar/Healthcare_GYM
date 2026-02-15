"""Environment setup for the Triage & Emergency domain."""

import json
from pathlib import Path
from typing import Optional

from bioagents.domains.triage_emergency.data_model import (
    TriageEmergencyDB,
    DB_PATH,
    POLICY_PATH,
    TASKS_PATH,
)
from bioagents.domains.triage_emergency.tools import TriageEmergencyTools
from bioagents.environment.environment import Environment
from bioagents.environment.toolkit import CompositeToolKit
from bioagents.tools.knowledge_tools import KnowledgeTools


def get_environment(
    db: Optional[TriageEmergencyDB] = None,
    max_turns: int = 15,
) -> Environment:
    """Create a Triage & Emergency environment."""
    if db is None:
        db = TriageEmergencyDB.load(DB_PATH)

    domain_tools = TriageEmergencyTools(db)
    knowledge_tools = KnowledgeTools(db=db)
    tools = CompositeToolKit(domain_tools, knowledge_tools)

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = f.read()

    env = Environment(
        domain_name="triage_emergency",
        policy=policy,
        tools=tools,
        max_turns=max_turns,
    )

    return env


def get_tasks(task_split: Optional[str] = None) -> list[dict]:
    """Load tasks for the Triage & Emergency domain."""
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if task_split is None:
        return tasks

    # Check for split file
    split_file = Path(TASKS_PATH).parent / "split_tasks.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if task_split in splits:
            valid_ids = set(splits[task_split])
            return [t for t in tasks if t["id"] in valid_ids]

    return tasks
