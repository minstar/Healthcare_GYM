"""Guideline Auto-Updater — Keeps AGENT_GUIDELINE.md alive.

After each GYM cycle, the updater analyses collective agent experience
and appends/updates the guideline with:
  1. Discovered best practices (which tools work best per domain)
  2. Common failure patterns and how to avoid them
  3. Updated score baselines per model
  4. Peer learning insights (who excels where)
  5. Domain-specific tips from top performers

This ensures any new model entering the GYM has the latest intelligence.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUIDELINE_PATH = PROJECT_ROOT / "AGENT_GUIDELINE.md"
INSIGHTS_PATH = PROJECT_ROOT / "logs" / "guideline_insights.jsonl"


def _read_guideline() -> str:
    """Read the current guideline."""
    if GUIDELINE_PATH.exists():
        return GUIDELINE_PATH.read_text(encoding="utf-8")
    return ""


def _write_guideline(content: str):
    """Write the updated guideline."""
    GUIDELINE_PATH.write_text(content, encoding="utf-8")


def _append_insight(insight: dict):
    """Append an insight to the JSONL log."""
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INSIGHTS_PATH, "a") as f:
        f.write(json.dumps(insight, ensure_ascii=False, default=str) + "\n")


def update_from_cycle_results(
    agent_id: str,
    domain: str,
    cycle_result: dict,
    logbook=None,
):
    """Update the guideline after a GYM cycle completes.

    Called by AutonomousAgent after run_one_cycle().

    Args:
        agent_id: Agent that completed the cycle.
        domain: Domain trained on.
        cycle_result: Workout results dict.
        logbook: SharedLogbook instance (for peer insights).
    """
    try:
        workout = cycle_result.get("workout", cycle_result)
        pre_score = workout.get("pre_score", 0.0)
        post_score = workout.get("post_score", 0.0)
        improvement = workout.get("improvement", 0.0)
        errors = workout.get("errors", [])
        tasks_completed = workout.get("tasks_completed", 0)
        success = workout.get("success", False)

        # ── Record insight ──
        insight = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "domain": domain,
            "pre_score": pre_score,
            "post_score": post_score,
            "improvement": improvement,
            "errors": errors,
            "tasks_completed": tasks_completed,
            "success": success,
        }
        _append_insight(insight)

        # ── Decide if guideline update is needed ──
        # Update conditions:
        # 1. Significant improvement (> 10%)
        # 2. New error pattern not yet documented
        # 3. Every 10th cycle (periodic refresh)

        insights = _load_recent_insights(limit=50)
        cycle_count = len(insights)

        should_update = False
        update_reasons = []

        if improvement > 0.10:
            should_update = True
            update_reasons.append(
                f"{agent_id} improved {improvement:.0%} on {domain}"
            )

        if cycle_count > 0 and cycle_count % 10 == 0:
            should_update = True
            update_reasons.append(f"Periodic refresh (cycle {cycle_count})")

        # Check for new error patterns
        known_errors = {
            "premature_stop", "reasoning_error", "tool_use_failure",
            "over_investigation", "eval_crash",
        }
        novel_errors = [e for e in errors if e not in known_errors]
        if novel_errors:
            should_update = True
            update_reasons.append(f"New error patterns: {novel_errors}")

        if not should_update:
            return

        logger.info(
            f"[GuidelineUpdater] Updating guideline: {', '.join(update_reasons)}"
        )

        # ── Build dynamic section ──
        dynamic_section = _build_dynamic_section(insights, logbook)
        _inject_dynamic_section(dynamic_section)

    except Exception as e:
        logger.debug(f"[GuidelineUpdater] Update failed: {e}")


def refresh_guideline(logbook=None):
    """Full guideline refresh from all accumulated insights.

    Can be called manually or at GYM shutdown.
    """
    insights = _load_recent_insights(limit=200)
    if not insights:
        logger.info("[GuidelineUpdater] No insights to refresh from")
        return

    dynamic_section = _build_dynamic_section(insights, logbook)
    _inject_dynamic_section(dynamic_section)
    logger.info(
        f"[GuidelineUpdater] Full refresh from {len(insights)} insights"
    )


def _load_recent_insights(limit: int = 100) -> list[dict]:
    """Load recent insights from the JSONL log."""
    if not INSIGHTS_PATH.exists():
        return []

    insights = []
    with open(INSIGHTS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    insights.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return insights[-limit:]


def _build_dynamic_section(
    insights: list[dict],
    logbook=None,
) -> str:
    """Build the dynamic section of the guideline from insights."""
    lines = []
    lines.append("")
    lines.append("## 13. Live Intelligence (Auto-Updated)")
    lines.append("")
    lines.append(
        f"> Last auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"| Based on {len(insights)} recent cycles"
    )
    lines.append("")

    # ── 13.1 Domain Performance Baselines ──
    lines.append("### 13.1 Current Score Baselines")
    lines.append("")
    lines.append("| Domain | Avg Score | Best Agent | Common Errors |")
    lines.append("|--------|-----------|------------|---------------|")

    from collections import defaultdict, Counter
    domain_data = defaultdict(lambda: {"scores": [], "errors": [], "agents": {}})

    for ins in insights:
        d = ins.get("domain", "")
        a = ins.get("agent_id", "")
        score = ins.get("post_score", ins.get("pre_score", 0.0))
        errs = ins.get("errors", [])

        if d:
            domain_data[d]["scores"].append(score)
            domain_data[d]["errors"].extend(errs)
            if a:
                if a not in domain_data[d]["agents"]:
                    domain_data[d]["agents"][a] = []
                domain_data[d]["agents"][a].append(score)

    for domain in sorted(domain_data.keys()):
        data = domain_data[domain]
        avg_score = (
            sum(data["scores"]) / len(data["scores"])
            if data["scores"] else 0.0
        )
        error_counts = Counter(data["errors"])
        top_errors = ", ".join(
            f"{e}({c})" for e, c in error_counts.most_common(3)
        ) or "none"

        best_agent = ""
        best_score = 0.0
        for agent, scores in data["agents"].items():
            agent_avg = sum(scores) / len(scores) if scores else 0.0
            if agent_avg > best_score:
                best_score = agent_avg
                best_agent = agent

        lines.append(
            f"| {domain} | {avg_score:.1%} | {best_agent} ({best_score:.1%}) | {top_errors} |"
        )
    lines.append("")

    # ── 13.2 Discovered Best Practices ──
    lines.append("### 13.2 Discovered Best Practices")
    lines.append("")

    # Find agents with significant improvement
    improvements = []
    for ins in insights:
        imp = ins.get("improvement", 0.0)
        if imp > 0.05:
            improvements.append(ins)

    if improvements:
        lines.append("**What works (from successful training cycles):**")
        lines.append("")
        for ins in improvements[-5:]:
            lines.append(
                f"- **{ins['agent_id']}** improved **{ins['improvement']:.1%}** "
                f"on `{ins['domain']}` "
                f"(tasks: {ins.get('tasks_completed', '?')})"
            )
        lines.append("")
    else:
        lines.append("- No significant improvements recorded yet.")
        lines.append("- Agents are still in baseline evaluation phase.")
        lines.append("")

    # ── 13.3 Known Pitfalls ──
    lines.append("### 13.3 Known Pitfalls (from real agent experience)")
    lines.append("")

    all_errors = []
    for ins in insights:
        for e in ins.get("errors", []):
            all_errors.append((e, ins.get("domain", ""), ins.get("agent_id", "")))

    error_by_type = defaultdict(list)
    for err, domain, agent in all_errors:
        error_by_type[err].append(domain)

    if error_by_type:
        for err_type, domains in sorted(
            error_by_type.items(), key=lambda x: -len(x[1])
        ):
            domain_counts = Counter(domains)
            top_domain = domain_counts.most_common(1)[0] if domain_counts else ("?", 0)
            lines.append(
                f"- **`{err_type}`** ({len(domains)} occurrences): "
                f"Most common in `{top_domain[0]}` ({top_domain[1]}x)"
            )
    else:
        lines.append("- No error patterns recorded yet.")
    lines.append("")

    # ── 13.4 Peer Leaderboard ──
    if logbook:
        lines.append("### 13.4 Peer Leaderboard")
        lines.append("")
        try:
            lb = logbook.get_global_leaderboard()
            if lb:
                lines.append("| Rank | Agent | Avg Score | Domains | Trend |")
                lines.append("|------|-------|-----------|---------|-------|")
                for i, entry in enumerate(lb[:10], 1):
                    lines.append(
                        f"| {i} | {entry.agent_id} | "
                        f"{entry.avg_score:.1%} | "
                        f"{entry.domains_visited} | "
                        f"{entry.trend} |"
                    )
            lines.append("")
        except Exception:
            pass

    # ── 13.5 Recommended Next Actions ──
    lines.append("### 13.5 Recommended Focus Areas")
    lines.append("")

    # Find worst-performing domains
    worst_domains = sorted(
        domain_data.items(),
        key=lambda x: (
            sum(x[1]["scores"]) / len(x[1]["scores"])
            if x[1]["scores"] else 0.0
        ),
    )[:3]

    for domain, data in worst_domains:
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
        lines.append(
            f"- `{domain}` (avg: {avg:.1%}) — needs more training"
        )
    lines.append("")

    return "\n".join(lines)


def _inject_dynamic_section(dynamic_section: str):
    """Inject or replace the dynamic section in the guideline."""
    content = _read_guideline()

    # Update the timestamp
    content = re.sub(
        r"(> \*\*Last Updated\*\*:) .*",
        f"\\1 {datetime.now().strftime('%Y-%m-%d %H:%M')} (auto-updated by GYM system)",
        content,
    )

    # Find and replace existing dynamic section
    marker_start = "## 13. Live Intelligence (Auto-Updated)"
    marker_end = "---\n\n*This guideline is maintained"

    if marker_start in content:
        # Replace existing section
        start_idx = content.index(marker_start)
        # Find the end of the dynamic section (before the footer)
        if marker_end in content[start_idx:]:
            end_idx = content.index(marker_end, start_idx)
            content = content[:start_idx] + dynamic_section + "\n\n" + content[end_idx:]
        else:
            # Append before end
            content = content[:start_idx] + dynamic_section + "\n\n" + content[start_idx:]
    else:
        # Insert before the footer
        if marker_end in content:
            end_idx = content.index(marker_end)
            content = content[:end_idx] + dynamic_section + "\n\n" + content[end_idx:]
        else:
            content += "\n" + dynamic_section

    _write_guideline(content)
