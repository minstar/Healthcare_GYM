"""Environment setup for the EHR Management domain."""

import json
from pathlib import Path
from typing import Optional

from bioagents.domains.ehr_management.data_model import (
    EHRDB,
    DB_PATH,
    POLICY_PATH,
    TASKS_PATH,
)
from bioagents.domains.ehr_management.tools import EHRTools
from bioagents.environment.environment import Environment
from bioagents.environment.toolkit import CompositeToolKit
from bioagents.tools.knowledge_tools import KnowledgeTools


def get_environment(
    db: Optional[EHRDB] = None,
    max_turns: int = 15,
) -> Environment:
    """Create an EHR Management environment.

    Args:
        db: Optional pre-loaded database. If None, loads from default path.
        max_turns: Maximum number of interaction turns.

    Returns:
        Configured Environment instance.
    """
    if db is None:
        db = EHRDB.load(DB_PATH)

    domain_tools = EHRTools(db)
    knowledge_tools = KnowledgeTools(db=db)
    tools = CompositeToolKit(domain_tools, knowledge_tools)

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = f.read()

    env = Environment(
        domain_name="ehr_management",
        policy=policy,
        tools=tools,
        max_turns=max_turns,
    )

    return env


def get_tasks(task_split: Optional[str] = None) -> list[dict]:
    """Load tasks for the EHR Management domain.

    Args:
        task_split: Optional split name ('train', 'test', 'base').
                    None returns all tasks.

    Returns:
        List of task dictionaries.
    """
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if task_split is None:
        return tasks

    # First: check if tasks have inline 'split' field
    has_inline_split = any("split" in t for t in tasks)
    if has_inline_split:
        filtered = [t for t in tasks if t.get("split") == task_split]
        if filtered:
            return filtered

    # Fallback: check for split file
    split_file = Path(TASKS_PATH).parent / "split_tasks.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if task_split not in splits:
            raise ValueError(
                f"Invalid split '{task_split}'. Available: {list(splits.keys())}"
            )
        valid_ids = set(splits[task_split])
        return [t for t in tasks if t["id"] in valid_ids]

    return tasks
