"""Gymnasium-compatible environment for BIOAgents Healthcare AI GYM.

Provides a standard Gymnasium interface for training RL agents
in medical/biomedical tool-use tasks.

Supports:
- 7 medical domains (clinical_diagnosis, medical_qa, visual_diagnosis,
  drug_interaction, ehr_management, triage_emergency, radiology_report)
- Scaled tasks (tasks_scaled.json) alongside original tasks
- Unified evaluation criteria schema across all domains
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
from loguru import logger

BIOAGENT_ENV_ID = "BioAgent-v0"

# Domain registry
_DOMAIN_REGISTRY = {}
_DOMAINS_LOADED = False


def _register_domain(domain_name: str, get_environment_fn, get_tasks_fn):
    """Register a domain for use in the gym."""
    _DOMAIN_REGISTRY[domain_name] = {
        "get_environment": get_environment_fn,
        "get_tasks": get_tasks_fn,
    }


def get_registered_domains() -> list[str]:
    """Return list of registered domain names."""
    _load_default_domains()
    return list(_DOMAIN_REGISTRY.keys())


def _load_default_domains():
    """Load all default domains (lazy, once)."""
    global _DOMAINS_LOADED
    if _DOMAINS_LOADED:
        return
    _DOMAINS_LOADED = True

    _domain_imports = [
        ("clinical_diagnosis", "bioagents.domains.clinical_diagnosis.environment"),
        ("medical_qa", "bioagents.domains.medical_qa.environment"),
        ("visual_diagnosis", "bioagents.domains.visual_diagnosis.environment"),
        ("drug_interaction", "bioagents.domains.drug_interaction.environment"),
        ("ehr_management", "bioagents.domains.ehr_management.environment"),
        ("triage_emergency", "bioagents.domains.triage_emergency.environment"),
        ("radiology_report", "bioagents.domains.radiology_report.environment"),
        ("cross_domain", "bioagents.domains.cross_domain.environment"),
        ("psychiatry", "bioagents.domains.psychiatry.environment"),
        ("obstetrics", "bioagents.domains.obstetrics.environment"),
    ]

    for domain_name, module_path in _domain_imports:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            _register_domain(
                domain_name,
                mod.get_environment,
                mod.get_tasks,
            )
        except Exception:
            pass


class BioAgentGymEnv(gym.Env):
    """Gymnasium-compatible environment for biomedical agent training.
    
    Observation space: Text (conversation history + tool results)
    Action space: Text (agent messages or tool calls in JSON)
    
    Usage:
        register_bioagent_gym()
        env = gym.make("BioAgent-v0", domain="clinical_diagnosis", task_id="dx_pneumonia_001")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        domain: str = "clinical_diagnosis",
        task_id: Optional[str] = None,
        task_split: Optional[str] = None,
        max_turns: int = 20,
        render_mode: Optional[str] = None,
        use_scaled_tasks: bool = False,
        include_original: bool = True,
        **kwargs,
    ):
        """Initialize the GYM environment.

        Args:
            domain: Medical domain name.
            task_id: Specific task ID to run. None = allow any.
            task_split: 'train', 'test', or None (all).
            max_turns: Maximum interaction turns.
            render_mode: 'human' or 'ansi' or None.
            use_scaled_tasks: If True, also load tasks_scaled.json.
            include_original: If True (default), include original tasks
                              alongside scaled tasks.
        """
        super().__init__()
        
        self.domain_name = domain
        self.task_id = task_id
        self.task_split = task_split
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.use_scaled_tasks = use_scaled_tasks
        
        # Load domain
        _load_default_domains()
        if domain not in _DOMAIN_REGISTRY:
            raise ValueError(
                f"Domain '{domain}' not registered. "
                f"Available: {list(_DOMAIN_REGISTRY.keys())}"
            )
        
        domain_info = _DOMAIN_REGISTRY[domain]
        self._get_environment_fn = domain_info["get_environment"]
        self._get_tasks_fn = domain_info["get_tasks"]
        
        # Load tasks (original + optional scaled)
        self._tasks = self._load_tasks(task_split, use_scaled_tasks, include_original)
        
        # Normalize evaluation schema for all tasks
        self._tasks = [_normalize_task_schema(t) for t in self._tasks]
        self._task_map = {t["id"]: t for t in self._tasks}
        
        # Gymnasium spaces (text-based)
        # Use default alphanumeric + common punctuation charset
        _charset = "".join(chr(i) for i in range(32, 127))  # printable ASCII
        self.observation_space = spaces.Text(
            min_length=0, max_length=100000, charset=_charset
        )
        self.action_space = spaces.Text(
            min_length=1, max_length=10000, charset=_charset
        )
        
        # State
        self._env = None
        self._current_task = None
        self._turn_count = 0
        self._conversation_history = []
        self._tool_call_log = []
    
    def _load_tasks(
        self,
        task_split: Optional[str],
        use_scaled: bool,
        include_original: bool,
    ) -> list[dict]:
        """Load tasks with optional scaled tasks support.

        Args:
            task_split: Split name ('train', 'test', None).
            use_scaled: Whether to include scaled tasks.
            include_original: Whether to include original tasks.

        Returns:
            Combined list of task dicts.
        """
        all_tasks = []
        seen_ids = set()

        # Load original tasks
        if include_original:
            original = self._get_tasks_fn(task_split)
            for t in original:
                if t["id"] not in seen_ids:
                    all_tasks.append(t)
                    seen_ids.add(t["id"])

        # Load scaled tasks
        if use_scaled:
            scaled = _load_scaled_tasks(self.domain_name, task_split)
            for t in scaled:
                if t["id"] not in seen_ids:
                    all_tasks.append(t)
                    seen_ids.add(t["id"])

        # Load auto-generated tasks (from AutoTaskGenerator / knowledge mining)
        auto_tasks = _load_auto_generated_tasks(self.domain_name, task_split)
        for t in auto_tasks:
            if t["id"] not in seen_ids:
                all_tasks.append(t)
                seen_ids.add(t["id"])

        if not all_tasks:
            logger.warning(
                f"No tasks found for domain={self.domain_name}, "
                f"split={task_split}, scaled={use_scaled}"
            )

        return all_tasks

    @property
    def num_tasks(self) -> int:
        """Number of available tasks."""
        return len(self._tasks)

    @property
    def task_ids(self) -> list[str]:
        """List of available task IDs."""
        return list(self._task_map.keys())

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[str, dict]:
        """Reset the environment with a task.
        
        Args:
            seed: Random seed
            options: Optional dict with 'task_id' to select a specific task
            
        Returns:
            observation: Initial observation (system prompt + patient ticket)
            info: Dict with task info, tools, policy
        """
        super().reset(seed=seed)
        
        # Select task
        task_id = self.task_id
        if options and "task_id" in options:
            task_id = options["task_id"]
        
        if task_id:
            if task_id not in self._task_map:
                raise ValueError(f"Task '{task_id}' not found. Available: {list(self._task_map.keys())}")
            self._current_task = self._task_map[task_id]
        else:
            # Random task selection
            idx = self.np_random.integers(0, len(self._tasks))
            self._current_task = self._tasks[idx]
        
        # Create fresh environment
        self._env = self._get_environment_fn(max_turns=self.max_turns)
        self._turn_count = 0
        self._conversation_history = []
        self._tool_call_log = []
        
        # Build initial observation
        initial_obs = self._build_initial_observation()
        
        info = {
            "task_id": self._current_task["id"],
            "task_description": self._current_task.get("description", {}),
            "domain": self.domain_name,
            "policy": self._env.policy,
            "tools": self._env.get_tool_definitions(),
            "max_turns": self.max_turns,
        }
        
        return initial_obs, info
    
    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Execute an agent action.
        
        The action can be:
        1. A JSON tool call: {"name": "tool_name", "arguments": {...}}
        2. A text message to the user
        
        Returns:
            observation: Result of the action
            reward: Computed reward (0.0 during interaction, final at end)
            terminated: Whether the episode ended
            truncated: Whether max turns exceeded
            info: Additional information
        """
        self._turn_count += 1
        
        observation, reward, terminated, truncated, info = self._env.step(action)
        
        # Track tool calls
        if info.get("tool_response"):
            self._tool_call_log.append({
                "turn": self._turn_count,
                "tool_name": self._env._last_tool_name,
                "arguments": self._env._last_tool_args,
                "response": info["tool_response"],
            })
        
        # Record in conversation history
        self._conversation_history.append({
            "turn": self._turn_count,
            "agent_action": action,
            "observation": observation,
        })
        
        # Check truncation
        if self._turn_count >= self.max_turns:
            truncated = True
        
        # Compute reward at episode end
        if terminated or truncated:
            reward = self._compute_reward()
        
        info.update({
            "turn_count": self._turn_count,
            "tool_calls": self._tool_call_log,
            "task_id": self._current_task["id"] if self._current_task else None,
        })
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _build_initial_observation(self) -> str:
        """Build the initial observation from the task."""
        task = self._current_task
        
        # Domain-specific header and instructions
        if self.domain_name == "medical_qa":
            header = f"=== BIOAgents Medical QA Task: {task['id']} ==="
            ticket_label = "--- Question ---"
            instructions = [
                "--- Instructions ---",
                "Use the available tools to search for evidence and reason through the question.",
                "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "When you are ready, use the submit_answer tool to submit your final answer.",
            ]
        elif self.domain_name == "visual_diagnosis":
            header = f"=== BIOAgents Visual Diagnosis Task: {task['id']} ==="
            ticket_label = "--- Visual Diagnosis Task ---"
            instructions = [
                "--- Instructions ---",
                "Use the available tools to analyze the medical image and answer the visual question.",
                "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "When you are ready, use answer_visual_question to submit your answer.",
            ]
        elif self.domain_name == "drug_interaction":
            header = f"=== BIOAgents Drug Interaction Task: {task['id']} ==="
            ticket_label = "--- Drug Interaction Review ---"
            instructions = [
                "--- Instructions ---",
                "Review the patient's medication profile and check for drug-drug interactions.",
                "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "When done, use submit_answer to provide your recommendation.",
            ]
        elif self.domain_name == "ehr_management":
            header = f"=== BIOAgents EHR Management Task: {task['id']} ==="
            ticket_label = "--- EHR Clinical Task ---"
            instructions = [
                "--- Instructions ---",
                "Review the patient's Electronic Health Records using the available tools.",
                "Analyze labs, vitals, medications, procedures, and clinical scores as needed.",
                "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "When done, use submit_answer to provide your clinical assessment and recommendation.",
            ]
        elif self.domain_name == "triage_emergency":
            header = f"=== BIOAgents Triage & Emergency Task: {task['id']} ==="
            ticket_label = "--- Emergency Triage Case ---"
            instructions = [
                "--- Instructions ---",
                "You are an emergency medicine physician performing triage assessment.",
                "Evaluate the patient's condition, determine ESI level, and provide initial management.",
                "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "When done, use submit_answer to provide your triage decision and initial orders.",
            ]
        elif self.domain_name == "radiology_report":
            header = f"=== BIOAgents Radiology Report Task: {task['id']} ==="
            ticket_label = "--- Radiology Reporting Task ---"
            instructions = [
                "--- Instructions ---",
                "You are a radiologist generating a structured report for a medical image.",
                "Analyze the image findings, provide clinical correlation, and write a complete report.",
                "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "When done, use submit_report to submit your structured radiology report.",
            ]
        else:
            header = f"=== BIOAgents Clinical Task: {task['id']} ==="
            ticket_label = "--- Patient Ticket ---"
            instructions = [
                "--- Instructions ---",
                "Use the available tools to assess the patient and provide your clinical recommendation.",
                "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "When done, provide your final assessment as a text message.",
            ]
        
        parts = [
            header,
            "",
            f"Domain: {self.domain_name}",
            "",
            ticket_label,
            task.get("ticket", "No ticket provided."),
            "",
            "--- Available Tools ---",
        ]
        
        tool_defs = self._env.get_tool_definitions()
        for i, td in enumerate(tool_defs, 1):
            func = td.get("function", {})
            parts.append(f"{i}. {func.get('name', '?')}: {func.get('description', '')[:100]}")
        
        parts.extend([""] + instructions)
        
        return "\n".join(parts)
    
    def _compute_reward(self) -> float:
        """Compute the reward for the completed episode.
        
        Evaluates:
        1. ACTION score: Did the agent call the expected tools?
        2. NL_ASSERTION score: Did the agent's reasoning meet clinical criteria?
        """
        if self._current_task is None:
            return 0.0
        
        eval_criteria = self._current_task.get("evaluation_criteria", {})
        expected_actions = eval_criteria.get("actions", [])
        reward_basis = eval_criteria.get("reward_basis", ["ACTION"])
        
        total_score = 0.0
        num_components = 0
        
        # --- ACTION score ---
        if "ACTION" in reward_basis and expected_actions:
            action_score = self._score_actions(expected_actions)
            total_score += action_score
            num_components += 1
        
        # --- NL_ASSERTION score ---
        if "NL_ASSERTION" in reward_basis:
            assertions = eval_criteria.get("assertions", [])
            if assertions:
                assertion_score = self._score_nl_assertions(assertions)
            else:
                # Fallback: mirror action score when no assertions defined
                assertion_score = total_score / max(num_components, 1)
            total_score += assertion_score
            num_components += 1
        
        return total_score / max(num_components, 1)
    
    def _score_actions(self, expected_actions: list[dict]) -> float:
        """Score the agent's tool usage against expected actions."""
        if not expected_actions:
            return 1.0
        
        actual_tool_names = [tc["tool_name"] for tc in self._tool_call_log]
        
        matched = 0
        for exp in expected_actions:
            exp_name = exp.get("name", "")
            compare_args = exp.get("compare_args", [])
            exp_args = exp.get("arguments", {})
            
            for tc in self._tool_call_log:
                if tc["tool_name"] == exp_name:
                    # Check argument matching if specified
                    if compare_args:
                        all_match = all(
                            str(tc["arguments"].get(k, "")).lower() == str(exp_args.get(k, "")).lower()
                            for k in compare_args
                            if k in exp_args
                        )
                        if all_match:
                            matched += 1
                            break
                    else:
                        matched += 1
                        break
        
        return matched / len(expected_actions)
    
    def _score_nl_assertions(self, assertions: list[str]) -> float:
        """Score agent's reasoning against natural language assertions.
        
        Uses heuristic keyword matching as a fast proxy.
        For LLM-as-Judge evaluation, use bioagents.gym.self_play.TrajectoryJudge.
        """
        if not assertions:
            return 1.0
        
        # Collect all agent text
        agent_text = " ".join(
            entry["agent_action"] + " " + entry.get("observation", "")
            for entry in self._conversation_history
        ).lower()
        
        # Also check tool names used
        tool_names_used = {tc["tool_name"].lower() for tc in self._tool_call_log}
        
        matched = 0
        for assertion in assertions:
            assertion_lower = assertion.lower()
            
            # Extract key concepts from assertion
            # "Agent should consider pneumonia in differential" → check for "pneumonia", "differential"
            keywords = []
            for word in assertion_lower.split():
                if len(word) > 4 and word not in (
                    "agent", "should", "consider", "provide", "identify",
                    "assess", "review", "check", "order", "appropriate",
                    "recommend", "ensure", "verify", "document",
                ):
                    keywords.append(word.strip(",.;:"))
            
            # Check if keywords appear in agent's text or tool calls
            if keywords:
                keyword_hits = sum(1 for kw in keywords if kw in agent_text)
                keyword_ratio = keyword_hits / len(keywords)
                
                # Also check tool_names for tool-related assertions
                tool_hits = sum(1 for kw in keywords if any(kw in tn for tn in tool_names_used))
                
                score = min(1.0, keyword_ratio + tool_hits * 0.2)
                if score >= 0.4:
                    matched += 1
            else:
                # No extractable keywords, give partial credit
                matched += 0.5
        
        return matched / len(assertions)

    def render(self):
        """Render the environment state."""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Domain: {self.domain_name} | Turn: {self._turn_count}/{self.max_turns}")
        output.append(f"Task: {self._current_task['id'] if self._current_task else 'None'}")
        output.append(f"Tool calls made: {len(self._tool_call_log)}")
        output.append(f"{'='*60}")
        
        for entry in self._conversation_history[-3:]:  # Show last 3 turns
            output.append(f"\n[Turn {entry['turn']}]")
            action_preview = entry['agent_action'][:200]
            output.append(f"  Action: {action_preview}")
            obs_preview = entry['observation'][:200]
            output.append(f"  Result: {obs_preview}")
        
        text = "\n".join(output)
        if self.render_mode == "human":
            print(text)
        return text
    
    def get_trajectory(self) -> dict:
        """Get the complete interaction trajectory for logging."""
        return {
            "domain": self.domain_name,
            "task_id": self._current_task["id"] if self._current_task else None,
            "total_turns": self._turn_count,
            "tool_call_log": self._tool_call_log,
            "conversation_history": self._conversation_history,
            "final_reward": self._compute_reward() if self._current_task else 0.0,
        }


def register_bioagent_gym():
    """Register the BIOAgents environment with Gymnasium."""
    try:
        gym.register(
            id=BIOAGENT_ENV_ID,
            entry_point="bioagents.gym.agent_env:BioAgentGymEnv",
        )
    except gym.error.Error:
        # Already registered
        pass


# ══════════════════════════════════════════════════════════════
#  Scaled Tasks Loader
# ══════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_scaled_tasks(
    domain: str,
    task_split: Optional[str] = None,
) -> list[dict]:
    """Load scaled tasks from tasks_scaled.json.

    Args:
        domain: Domain name.
        task_split: Optional split ('train', 'test').

    Returns:
        List of scaled task dicts.
    """
    scaled_path = _PROJECT_ROOT / "data" / "domains" / domain / "tasks_scaled.json"
    if not scaled_path.exists():
        return []

    with open(scaled_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if task_split is None:
        return tasks

    # Check split file
    split_path = scaled_path.parent / "split_tasks_scaled.json"
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if task_split in splits:
            valid_ids = set(splits[task_split])
            return [t for t in tasks if t["id"] in valid_ids]

    return tasks


# ══════════════════════════════════════════════════════════════
#  Auto-Generated Tasks Loader
# ══════════════════════════════════════════════════════════════


def _load_auto_generated_tasks(
    domain: str,
    task_split: Optional[str] = None,
) -> list[dict]:
    """Load auto-generated tasks from tasks_auto_generated.json.

    These are produced by ``bioagents.data_pipeline.auto_task_generator``
    from knowledge sources (FTS5 passages, benchmark conversions,
    instruction mining, guideline synthesis).

    Args:
        domain: Domain name.
        task_split: Optional split ('train', 'test').
            For auto-generated tasks, all are treated as 'train'
            unless a split file exists.

    Returns:
        List of auto-generated task dicts.
    """
    auto_path = _PROJECT_ROOT / "data" / "domains" / domain / "tasks_auto_generated.json"
    if not auto_path.exists():
        return []

    try:
        with open(auto_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except (json.JSONDecodeError, Exception):
        return []

    if not isinstance(tasks, list):
        return []

    # Auto-generated tasks are always available for training
    # but excluded from 'test' split to avoid data leakage
    if task_split == "test":
        return []

    return tasks


# ══════════════════════════════════════════════════════════════
#  Schema Normalization
# ══════════════════════════════════════════════════════════════

def _normalize_task_schema(task: dict) -> dict:
    """Normalize task schema to unified evaluation_criteria format.

    Converts EHR-style tasks (expected_actions/rubric) to the standard
    evaluation_criteria format used by other domains.

    Standard format:
    {
        "evaluation_criteria": {
            "actions": [{"name": str, "arguments": dict, "compare_args": list}],
            "nl_assertions": [str],
            "reward_basis": ["ACTION", "NL_ASSERTION"]
        }
    }
    """
    # Already has evaluation_criteria — validate and return
    if "evaluation_criteria" in task:
        ec = task["evaluation_criteria"]
        # Ensure required keys exist
        if "actions" not in ec:
            ec["actions"] = []
        if "reward_basis" not in ec:
            ec["reward_basis"] = ["ACTION"]
        # Normalize assertions key (some use 'assertions', some 'nl_assertions')
        if "assertions" in ec and "nl_assertions" not in ec:
            ec["nl_assertions"] = ec.pop("assertions")
        return task

    # EHR-style: convert expected_actions + rubric → evaluation_criteria
    normalized = {**task}
    actions = []
    nl_assertions = []

    if "expected_actions" in task:
        for ea in task["expected_actions"]:
            actions.append({
                "name": ea.get("tool", ""),
                "arguments": ea.get("args", {}),
                "compare_args": list(ea.get("args", {}).keys()),
                "info": f"Expected tool call: {ea.get('tool', '')}",
            })

    if "rubric" in task:
        rubric = task["rubric"]
        must_mention = rubric.get("must_mention", [])
        for item in must_mention:
            nl_assertions.append(f"Agent should identify: {item}")

    if "expected_answer" in task:
        # Extract key phrases from expected answer as assertions
        answer = task["expected_answer"]
        if len(answer) > 50:
            # Long answer: extract first sentence as assertion
            first_sent = answer.split(".")[0].strip()
            nl_assertions.append(f"Agent's assessment should include: {first_sent}")

    normalized["evaluation_criteria"] = {
        "actions": actions,
        "nl_assertions": nl_assertions,
        "reward_basis": ["ACTION", "NL_ASSERTION"] if nl_assertions else ["ACTION"],
    }

    return normalized


def get_gym_stats() -> dict:
    """Get comprehensive statistics about the GYM environment.

    Returns:
        Dict with domain counts, task counts, tool counts, etc.
    """
    _load_default_domains()

    stats = {
        "total_domains": len(_DOMAIN_REGISTRY),
        "domains": {},
    }

    total_tasks = 0
    total_tools = 0

    for domain_name, info in _DOMAIN_REGISTRY.items():
        domain_stats = {"name": domain_name}

        # Count tasks
        try:
            tasks = info["get_tasks"](None)
            domain_stats["original_tasks"] = len(tasks)
        except Exception:
            domain_stats["original_tasks"] = 0

        # Count scaled tasks
        scaled = _load_scaled_tasks(domain_name)
        domain_stats["scaled_tasks"] = len(scaled)
        domain_stats["total_tasks"] = domain_stats["original_tasks"] + len(scaled)

        # Count tools
        try:
            env = info["get_environment"](max_turns=1)
            tools = env.get_tool_definitions()
            domain_stats["tools"] = len(tools)
            domain_stats["tool_names"] = [
                t.get("function", {}).get("name", "?") for t in tools
            ]
        except Exception:
            domain_stats["tools"] = 0
            domain_stats["tool_names"] = []

        total_tasks += domain_stats["total_tasks"]
        total_tools += domain_stats["tools"]
        stats["domains"][domain_name] = domain_stats

    stats["total_tasks"] = total_tasks
    stats["total_tools"] = total_tools

    return stats
