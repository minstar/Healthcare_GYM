"""SharedLogbook -- Cross-Agent Record Sharing System for Autonomous GYM.

All agents write to and read from this shared logbook. It enables:
1. Cross-agent learning: Agent A can see Agent B's strengths/weaknesses
2. Collective intelligence: Aggregated patterns reveal what ALL agents struggle with
3. Competitive motivation: Leaderboards inspire agents to self-improve
4. Anti-herding: Diversity tracking prevents all agents from chasing the same domain

Usage:
    logbook = SharedLogbook("logs/shared_logbook")

    # Agent records a workout
    logbook.record_workout(WorkoutEntry(
        agent_id="qwen3_8b_v1",
        domain="clinical_diagnosis",
        action_score=0.72,
        errors=["safety_violation", "premature_stop"],
    ))

    # Another agent reads the logbook to decide what to do
    leaderboard = logbook.get_leaderboard("clinical_diagnosis")
    suggestions = logbook.get_improvement_suggestions("my_agent_id")
"""

import json
import math
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


# ============================================================
# 1. Data Structures
# ============================================================

@dataclass
class WorkoutEntry:
    """A single gym workout record for an agent."""
    agent_id: str
    domain: str
    task_id: str = ""
    timestamp: str = ""

    # Performance
    action_score: float = 0.0
    reward_score: float = 0.0
    composite_score: float = 0.0

    # What went wrong
    errors: list = field(default_factory=list)
    error_details: list = field(default_factory=list)

    # What went right
    tools_used: list = field(default_factory=list)
    successful_patterns: list = field(default_factory=list)

    # Agent's self-reflection (if available)
    self_reflection: str = ""
    confidence: float = 0.5

    # Training context
    iteration: int = 0
    model_path: str = ""
    training_method: str = ""
    gpu_id: int = -1
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentProfile:
    """Aggregated profile of an agent's capabilities."""
    agent_id: str
    model_path: str = ""
    created_at: str = ""
    last_active: str = ""

    # Per-domain performance
    domain_scores: dict = field(default_factory=dict)
    domain_mastery: dict = field(default_factory=dict)

    # Aggregate stats
    total_workouts: int = 0
    total_domains_visited: int = 0
    avg_score: float = 0.0

    # Strengths and weaknesses
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)

    # Error pattern summary
    recurring_errors: dict = field(default_factory=dict)

    # Learning trajectory
    score_history: list = field(default_factory=list)


@dataclass
class LeaderboardEntry:
    """A single entry in the domain leaderboard."""
    agent_id: str
    domain: str
    score: float
    mastery: str
    total_attempts: int
    last_updated: str
    trend: str = "stable"


# ============================================================
# 2. Workout Logger
# ============================================================

class WorkoutLogger:
    """Thread-safe workout logging with file persistence."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "workouts"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, entry: WorkoutEntry):
        """Log a workout entry (thread-safe)."""
        if not entry.timestamp:
            entry.timestamp = datetime.now().isoformat()

        with self._lock:
            agent_file = self.log_dir / f"{entry.agent_id}.jsonl"
            with open(agent_file, "a") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

            global_file = self.log_dir / "all_workouts.jsonl"
            with open(global_file, "a") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def get_agent_workouts(
        self, agent_id: str, domain: str = "", last_n: int = 0
    ) -> list:
        """Get workout history for a specific agent."""
        agent_file = self.log_dir / f"{agent_id}.jsonl"
        if not agent_file.exists():
            return []

        entries = []
        with open(agent_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if domain and entry.get("domain") != domain:
                        continue
                    entries.append(entry)

        if last_n > 0:
            entries = entries[-last_n:]
        return entries

    def get_all_workouts(self, domain: str = "", since: str = "") -> list:
        """Get all workouts across all agents."""
        global_file = self.log_dir / "all_workouts.jsonl"
        if not global_file.exists():
            return []

        entries = []
        with open(global_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if domain and entry.get("domain") != domain:
                        continue
                    if since and entry.get("timestamp", "") < since:
                        continue
                    entries.append(entry)
        return entries

    def get_unique_agents(self) -> list:
        """Get list of all agent IDs that have logged workouts."""
        agents = []
        for f in self.log_dir.glob("*.jsonl"):
            if f.name != "all_workouts.jsonl":
                agents.append(f.stem)
        return sorted(agents)


# ============================================================
# 3. Profile Manager
# ============================================================

class ProfileManager:
    """Manages agent profiles -- aggregated from workout history."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "profiles"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict = {}
        self._lock = threading.Lock()
        self._load_profiles()

    def _load_profiles(self):
        """Load all profiles from disk."""
        for f in self.log_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                profile = AgentProfile(**data)
                self._profiles[profile.agent_id] = profile
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load profile {f}: {e}")

    def update_profile(self, agent_id: str, workout: WorkoutEntry):
        """Update an agent's profile based on a new workout."""
        with self._lock:
            if agent_id not in self._profiles:
                self._profiles[agent_id] = AgentProfile(
                    agent_id=agent_id,
                    model_path=workout.model_path,
                    created_at=datetime.now().isoformat(),
                )

            profile = self._profiles[agent_id]
            profile.last_active = datetime.now().isoformat()
            profile.total_workouts += 1

            # Update domain scores
            profile.domain_scores[workout.domain] = workout.action_score
            profile.total_domains_visited = len(profile.domain_scores)

            # Update mastery
            mastery = self._score_to_mastery(workout.action_score)
            profile.domain_mastery[workout.domain] = mastery

            # Update avg score
            if profile.domain_scores:
                profile.avg_score = (
                    sum(profile.domain_scores.values()) / len(profile.domain_scores)
                )

            # Track errors
            for err in workout.errors:
                profile.recurring_errors[err] = (
                    profile.recurring_errors.get(err, 0) + 1
                )

            # Update strengths/weaknesses
            self._compute_strengths_weaknesses(profile)

            # Score history (keep last 100)
            profile.score_history.append({
                "timestamp": workout.timestamp,
                "domain": workout.domain,
                "score": workout.action_score,
            })
            if len(profile.score_history) > 100:
                profile.score_history = profile.score_history[-100:]

            self._save_profile(profile)

    def get_profile(self, agent_id: str) -> Optional[AgentProfile]:
        return self._profiles.get(agent_id)

    def get_all_profiles(self) -> dict:
        return dict(self._profiles)

    def _compute_strengths_weaknesses(self, profile: AgentProfile):
        if not profile.domain_scores:
            return
        avg = profile.avg_score
        threshold = 0.1
        profile.strengths = [
            d for d, s in sorted(profile.domain_scores.items(), key=lambda x: -x[1])
            if s > avg + threshold
        ][:5]
        profile.weaknesses = [
            d for d, s in sorted(profile.domain_scores.items(), key=lambda x: x[1])
            if s < avg - threshold
        ][:5]

    def _score_to_mastery(self, score: float) -> str:
        if score < 0.30:
            return "novice"
        elif score < 0.50:
            return "beginner"
        elif score < 0.70:
            return "intermediate"
        elif score < 0.85:
            return "advanced"
        elif score < 0.95:
            return "expert"
        else:
            return "master"

    def _save_profile(self, profile: AgentProfile):
        path = self.log_dir / f"{profile.agent_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(profile), f, indent=2, ensure_ascii=False)


# ============================================================
# 4. Leaderboard
# ============================================================

class Leaderboard:
    """Per-domain leaderboards tracking agent rankings."""

    def __init__(self, profiles: ProfileManager):
        self.profiles = profiles

    def get_domain_leaderboard(self, domain: str) -> list:
        """Get sorted leaderboard for a specific domain."""
        entries = []
        for agent_id, profile in self.profiles.get_all_profiles().items():
            if domain in profile.domain_scores:
                recent_scores = [
                    h["score"] for h in profile.score_history[-10:]
                    if h.get("domain") == domain
                ]
                trend = self._compute_trend(recent_scores)
                entries.append(LeaderboardEntry(
                    agent_id=agent_id,
                    domain=domain,
                    score=profile.domain_scores[domain],
                    mastery=profile.domain_mastery.get(domain, "unknown"),
                    total_attempts=sum(
                        1 for h in profile.score_history
                        if h.get("domain") == domain
                    ),
                    last_updated=profile.last_active,
                    trend=trend,
                ))
        entries.sort(key=lambda e: e.score, reverse=True)
        return entries

    def get_global_leaderboard(self) -> list:
        """Get global leaderboard across all domains."""
        entries = []
        for agent_id, profile in self.profiles.get_all_profiles().items():
            entries.append({
                "agent_id": agent_id,
                "avg_score": profile.avg_score,
                "domains_visited": profile.total_domains_visited,
                "total_workouts": profile.total_workouts,
                "strengths": profile.strengths[:3],
                "weaknesses": profile.weaknesses[:3],
                "last_active": profile.last_active,
            })
        entries.sort(key=lambda e: e["avg_score"], reverse=True)
        return entries

    def _compute_trend(self, scores: list, window: int = 3) -> str:
        if len(scores) < 2:
            return "stable"
        recent = scores[-window:] if len(scores) >= window else scores
        earlier = scores[:-window] if len(scores) > window else scores[:1]
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        if recent_avg > earlier_avg + 0.05:
            return "improving"
        elif recent_avg < earlier_avg - 0.05:
            return "declining"
        return "stable"


# ============================================================
# 5. Insight Engine
# ============================================================

class InsightEngine:
    """Generates cross-agent insights and recommendations."""

    def __init__(self, profiles: ProfileManager, workout_logger: WorkoutLogger):
        self.profiles = profiles
        self.workouts = workout_logger

    def get_collective_weaknesses(self) -> list:
        """Find domains/skills that ALL agents struggle with."""
        all_profiles = self.profiles.get_all_profiles()
        if not all_profiles:
            return []

        domain_scores = defaultdict(list)
        for profile in all_profiles.values():
            for domain, score in profile.domain_scores.items():
                domain_scores[domain].append(score)

        weaknesses = []
        for domain, scores in domain_scores.items():
            avg = sum(scores) / len(scores)
            if avg < 0.6:
                weaknesses.append({
                    "domain": domain,
                    "avg_score": avg,
                    "num_agents_attempted": len(scores),
                    "best_score": max(scores),
                    "worst_score": min(scores),
                    "recommendation": (
                        f"ALL agents struggle with {domain} (avg {avg:.1%}). "
                        f"This domain may need better training data or tools."
                    ),
                })
        weaknesses.sort(key=lambda w: w["avg_score"])
        return weaknesses

    def get_complementary_strengths(self) -> dict:
        """Find which agents are best at which domains."""
        all_profiles = self.profiles.get_all_profiles()
        domain_best = defaultdict(list)
        for agent_id, profile in all_profiles.items():
            for domain, score in profile.domain_scores.items():
                domain_best[domain].append((agent_id, score))

        result = {}
        for domain, agents in domain_best.items():
            agents.sort(key=lambda x: -x[1])
            result[domain] = [a[0] for a in agents]
        return result

    def detect_herding(self, recent_window: int = 20) -> dict:
        """Detect if agents are all training on the same domain (herding)."""
        recent_workouts = self.workouts.get_all_workouts()[-recent_window:]
        if not recent_workouts:
            return {"herding_detected": False, "diversity_score": 1.0}

        domain_counts = Counter(w.get("domain", "") for w in recent_workouts)
        total = sum(domain_counts.values())

        entropy = 0.0
        for count in domain_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(max(len(domain_counts), 1)) if domain_counts else 1.0
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        dominant = domain_counts.most_common(1)[0] if domain_counts else ("none", 0)
        dominant_ratio = dominant[1] / total if total > 0 else 0

        herding = diversity < 0.5 and dominant_ratio > 0.5

        return {
            "herding_detected": herding,
            "dominant_domain": dominant[0],
            "dominant_ratio": dominant_ratio,
            "domain_distribution": dict(domain_counts),
            "diversity_score": round(diversity, 3),
            "recommendation": (
                f"HERDING WARNING: {dominant_ratio:.0%} of recent workouts are in "
                f"{dominant[0]}. Encourage agents to explore under-served domains."
            ) if herding else "Diversity is healthy.",
        }

    def get_improvement_suggestions(self, agent_id: str) -> list:
        """Get improvement suggestions for a specific agent based on others' records."""
        my_profile = self.profiles.get_profile(agent_id)
        if not my_profile:
            return [{"suggestion": "No profile found. Start working out!"}]

        all_profiles = self.profiles.get_all_profiles()
        suggestions = []

        # 1. Domains where I'm weak but others are strong
        for domain, my_score in my_profile.domain_scores.items():
            others_scores = [
                p.domain_scores.get(domain, 0)
                for aid, p in all_profiles.items()
                if aid != agent_id and domain in p.domain_scores
            ]
            if others_scores:
                best_other = max(others_scores)
                if best_other - my_score > 0.15:
                    best_agent = [
                        aid for aid, p in all_profiles.items()
                        if p.domain_scores.get(domain, 0) == best_other
                    ]
                    suggestions.append({
                        "type": "learn_from_peer",
                        "domain": domain,
                        "my_score": my_score,
                        "best_peer_score": best_other,
                        "best_peer": best_agent[0] if best_agent else "unknown",
                        "gap": best_other - my_score,
                        "suggestion": (
                            f"In {domain}, you score {my_score:.1%} but "
                            f"{best_agent[0] if best_agent else 'another agent'} "
                            f"scores {best_other:.1%}. Study their approach."
                        ),
                    })

        # 2. Domains I haven't tried but others have
        all_domains = set()
        for p in all_profiles.values():
            all_domains.update(p.domain_scores.keys())

        untried = all_domains - set(my_profile.domain_scores.keys())
        for domain in untried:
            others_scores = [
                p.domain_scores[domain]
                for p in all_profiles.values()
                if domain in p.domain_scores
            ]
            if others_scores:
                avg_other = sum(others_scores) / len(others_scores)
                suggestions.append({
                    "type": "explore_new_domain",
                    "domain": domain,
                    "others_avg": avg_other,
                    "gap": 0.5,
                    "suggestion": (
                        f"You haven't tried {domain} yet. "
                        f"Other agents average {avg_other:.1%} there."
                    ),
                })

        # 3. Error patterns unique to this agent
        my_errors = set(my_profile.recurring_errors.keys())
        others_common_errors = Counter()
        for aid, p in all_profiles.items():
            if aid != agent_id:
                others_common_errors.update(p.recurring_errors.keys())

        unique_errors = my_errors - set(
            err for err, c in others_common_errors.items() if c >= 2
        )
        for err in unique_errors:
            suggestions.append({
                "type": "unique_weakness",
                "error_type": err,
                "my_count": my_profile.recurring_errors[err],
                "gap": 0.3,
                "suggestion": (
                    f"You have a unique weakness: '{err}' "
                    f"({my_profile.recurring_errors[err]}x). "
                    f"Other agents don't struggle with this as much."
                ),
            })

        suggestions.sort(key=lambda s: s.get("gap", 0), reverse=True)
        return suggestions

    def get_diversity_recommendation(self, agent_id: str) -> dict:
        """Recommend which domain an agent should visit next."""
        herding = self.detect_herding()
        my_profile = self.profiles.get_profile(agent_id)

        all_profiles = self.profiles.get_all_profiles()
        domain_visit_counts = Counter()
        for p in all_profiles.values():
            for domain in p.domain_scores:
                domain_visit_counts[domain] += 1

        all_domains = set()
        for p in all_profiles.values():
            all_domains.update(p.domain_scores.keys())

        if my_profile:
            candidates = []
            for domain in all_domains:
                my_score = my_profile.domain_scores.get(domain, 0.0)
                visit_count = domain_visit_counts.get(domain, 0)
                priority = (
                    (1.0 - my_score) * 0.6
                    + (1.0 / max(visit_count, 1)) * 0.4
                )
                candidates.append({
                    "domain": domain,
                    "my_score": my_score,
                    "visit_count": visit_count,
                    "priority": priority,
                })
            candidates.sort(key=lambda c: c["priority"], reverse=True)
        else:
            candidates = [
                {
                    "domain": d,
                    "priority": 1.0 / max(domain_visit_counts.get(d, 1), 1),
                }
                for d in all_domains
            ]

        return {
            "recommended_domain": candidates[0]["domain"] if candidates else None,
            "candidates": candidates[:5],
            "herding_status": herding,
        }


# ============================================================
# 6. SharedLogbook (Main Interface)
# ============================================================

class SharedLogbook:
    """Unified interface for the cross-agent shared logbook.

    Thread-safe. Multiple agents can read/write concurrently.
    """

    def __init__(self, log_dir: str = "logs/shared_logbook"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.workouts = WorkoutLogger(self.log_dir)
        self.profiles = ProfileManager(self.log_dir)
        self.leaderboard = Leaderboard(self.profiles)
        self.insights = InsightEngine(self.profiles, self.workouts)

        logger.info(f"[SharedLogbook] Initialized at {self.log_dir}")
        agent_count = len(self.profiles.get_all_profiles())
        if agent_count:
            logger.info(f"[SharedLogbook] Loaded {agent_count} agent profiles")

    # -- Write Methods --

    def record_workout(self, entry: WorkoutEntry):
        """Record a workout entry."""
        self.workouts.log(entry)
        self.profiles.update_profile(entry.agent_id, entry)
        logger.debug(
            f"[SharedLogbook] {entry.agent_id} recorded: "
            f"{entry.domain} score={entry.action_score:.2f}"
        )

    def record_workouts_batch(self, entries: list):
        """Record multiple workout entries at once."""
        for entry in entries:
            self.record_workout(entry)

    # -- Read Methods --

    def get_leaderboard(self, domain: str) -> list:
        return self.leaderboard.get_domain_leaderboard(domain)

    def get_global_leaderboard(self) -> list:
        return self.leaderboard.get_global_leaderboard()

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        return self.profiles.get_profile(agent_id)

    def get_agent_strengths(self, exclude: str = "") -> dict:
        """Get all agents' strengths, optionally excluding one agent."""
        result = {}
        for agent_id, profile in self.profiles.get_all_profiles().items():
            if agent_id != exclude:
                result[agent_id] = profile.strengths
        return result

    def get_improvement_suggestions(self, agent_id: str) -> list:
        return self.insights.get_improvement_suggestions(agent_id)

    def detect_herding(self) -> dict:
        return self.insights.detect_herding()

    def get_collective_weaknesses(self) -> list:
        return self.insights.get_collective_weaknesses()

    def get_diversity_recommendation(self, agent_id: str) -> dict:
        return self.insights.get_diversity_recommendation(agent_id)

    def get_agent_workout_history(
        self, agent_id: str, domain: str = "", last_n: int = 50
    ) -> list:
        return self.workouts.get_agent_workouts(agent_id, domain=domain, last_n=last_n)

    def get_registered_agents(self) -> list:
        return self.workouts.get_unique_agents()

    # -- Summary Methods --

    def get_gym_summary(self) -> dict:
        """Get comprehensive summary of the gym's state."""
        profiles = self.profiles.get_all_profiles()
        herding = self.insights.detect_herding()
        collective_weak = self.insights.get_collective_weaknesses()

        return {
            "total_agents": len(profiles),
            "total_workouts": sum(p.total_workouts for p in profiles.values()),
            "agents": {
                aid: {
                    "avg_score": p.avg_score,
                    "domains": p.total_domains_visited,
                    "workouts": p.total_workouts,
                    "strengths": p.strengths[:3],
                    "weaknesses": p.weaknesses[:3],
                }
                for aid, p in profiles.items()
            },
            "herding": herding,
            "collective_weaknesses": [w["domain"] for w in collective_weak],
            "diversity_score": herding.get("diversity_score", 0),
        }

    def print_gym_dashboard(self):
        """Print a visual dashboard of the gym state."""
        summary = self.get_gym_summary()

        print()
        print("=" * 70)
        print("  Healthcare AI GYM -- Shared Logbook Dashboard")
        print(
            f"  Agents: {summary['total_agents']}  |  "
            f"Total Workouts: {summary['total_workouts']}  |  "
            f"Diversity: {summary['diversity_score']:.2f}"
        )
        print("=" * 70)

        if summary["agents"]:
            print()
            print(
                f"  {'Agent':<20} {'Avg':>6} {'Domains':>8} {'Workouts':>9} "
                f"{'Strengths':<20} {'Weaknesses'}"
            )
            print("  " + "-" * 68)

            for aid, info in sorted(
                summary["agents"].items(),
                key=lambda x: x[1]["avg_score"],
                reverse=True,
            ):
                strengths = ", ".join(info["strengths"][:2]) or "-"
                weaknesses = ", ".join(info["weaknesses"][:2]) or "-"
                print(
                    f"  {aid:<20} {info['avg_score']:>5.1%} "
                    f"{info['domains']:>8} {info['workouts']:>9} "
                    f"{strengths:<20} {weaknesses}"
                )

        if summary["collective_weaknesses"]:
            print()
            print(
                f"  Collective weaknesses: "
                f"{', '.join(summary['collective_weaknesses'])}"
            )

        herding = summary["herding"]
        if herding.get("herding_detected"):
            print(
                f"  WARNING: Herding detected! "
                f"{herding.get('recommendation', '')}"
            )

        print("=" * 70)
