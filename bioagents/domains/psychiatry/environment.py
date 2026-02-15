"""Environment setup for the Psychiatry / Mental Health domain."""

import json
from pathlib import Path
from typing import Optional

from bioagents.domains.psychiatry.data_model import PsychiatryDB, DB_PATH, POLICY_PATH, TASKS_PATH
from bioagents.domains.psychiatry.tools import PsychiatryTools
from bioagents.environment.environment import Environment
from bioagents.environment.toolkit import CompositeToolKit
from bioagents.tools.knowledge_tools import KnowledgeTools


def get_environment(db: Optional[PsychiatryDB] = None, max_turns: int = 20) -> Environment:
    if db is None:
        db = PsychiatryDB.load(DB_PATH)
    domain_tools = PsychiatryTools(db)
    knowledge_tools = KnowledgeTools(db=db)
    tools = CompositeToolKit(domain_tools, knowledge_tools)
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = f.read()
    return Environment(domain_name="psychiatry", policy=policy, tools=tools, max_turns=max_turns)


def get_tasks(task_split: Optional[str] = None) -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if task_split is None:
        return tasks
    split_file = Path(TASKS_PATH).parent / "split_tasks.json"
    if split_file.exists():
        with open(split_file, "r") as f:
            splits = json.load(f)
        if task_split in splits:
            valid_ids = set(splits[task_split])
            return [t for t in tasks if t["id"] in valid_ids]
    return tasks
