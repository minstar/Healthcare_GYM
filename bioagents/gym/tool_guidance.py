"""Adaptive Tool Usage Guidance for Healthcare AI GYM.

Generates dynamic, task-specific prompts that guide the agent on
*which tools to use* and *when to use them* based on:
- Current task domain and difficulty
- Agent's known weaknesses (from logbook)
- Available tools in the environment
- Task characteristics (MC, open-ended, multi-step)

This module is injected into the system prompt at runtime, making
the agent aware of optimal tool-use strategies without hard-coding
fixed instructions.

Architecture:
    TaskAnalyzer  -> Analyzes task to determine characteristics
    ToolGuidance  -> Generates adaptive guidance text
    GuidanceInjector -> Injects guidance into system prompts

Usage:
    guidance = ToolGuidance.generate(
        domain="medical_qa",
        task=task_dict,
        tools=tool_definitions,
        agent_profile=reflection_dict,
    )
    # Inject into system prompt
    system_prompt = build_system_prompt(policy, tools, domain) + guidance
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# 1. Task Analyzer
# ============================================================

@dataclass
class TaskCharacteristics:
    """Analyzed characteristics of a task."""
    domain: str = ""
    is_multiple_choice: bool = False
    is_open_ended: bool = False
    is_multi_step: bool = False
    requires_evidence_search: bool = False
    requires_tool_chain: bool = False
    requires_knowledge_retrieval: bool = False
    requires_calculation: bool = False
    requires_image_analysis: bool = False
    difficulty: str = "moderate"  # easy, moderate, hard, expert
    topic_keywords: list = field(default_factory=list)
    expected_tool_sequence: list = field(default_factory=list)


class TaskAnalyzer:
    """Analyzes tasks to determine characteristics for guidance."""

    # Keywords that suggest evidence search is needed
    _EVIDENCE_KEYWORDS = {
        "mechanism", "pathophysiology", "etiology", "according to",
        "evidence", "guideline", "recommended", "first-line",
        "contraindicated", "interaction", "side effect", "adverse",
        "differential", "diagnosis", "prognosis", "treatment",
        "drug", "medication", "dosage", "pharmacology",
    }

    # Keywords that suggest calculation is needed
    _CALC_KEYWORDS = {
        "calculate", "score", "index", "ratio", "rate",
        "bmi", "gfr", "creatinine clearance", "anion gap",
        "corrected", "adjusted", "wells", "chads", "apache",
        "sofa", "meld", "child-pugh", "bishop",
    }

    # Multi-step indicators
    _MULTISTEP_KEYWORDS = {
        "then", "after that", "next step", "follow-up",
        "workup", "evaluate", "assess and", "order",
        "review", "analyze", "interpret", "compare",
    }

    @classmethod
    def analyze(cls, task: dict, domain: str = "") -> TaskCharacteristics:
        """Analyze a task to determine its characteristics."""
        chars = TaskCharacteristics(domain=domain)

        # Extract text fields
        ticket = task.get("ticket", "")
        description = task.get("description", "")
        if isinstance(description, dict):
            description = json.dumps(description)
        question = task.get("question", "")
        text = f"{ticket} {description} {question}".lower()

        # Multiple choice detection
        if any(p in text for p in ["(a)", "(b)", "(c)", "option a", "option b"]):
            chars.is_multiple_choice = True
        if task.get("options") or task.get("choices"):
            chars.is_multiple_choice = True

        # Open-ended detection
        if not chars.is_multiple_choice:
            chars.is_open_ended = True

        # Evidence search detection
        evidence_hits = sum(1 for kw in cls._EVIDENCE_KEYWORDS if kw in text)
        if evidence_hits >= 2:
            chars.requires_evidence_search = True
            chars.requires_knowledge_retrieval = True

        # Calculation detection
        calc_hits = sum(1 for kw in cls._CALC_KEYWORDS if kw in text)
        if calc_hits >= 1:
            chars.requires_calculation = True

        # Multi-step detection
        multistep_hits = sum(1 for kw in cls._MULTISTEP_KEYWORDS if kw in text)
        if multistep_hits >= 2:
            chars.is_multi_step = True

        # Domain-specific analysis
        chars = cls._domain_specific_analysis(chars, task, domain, text)

        # Extract topic keywords (for search guidance)
        chars.topic_keywords = cls._extract_topic_keywords(text)

        # Estimate difficulty
        chars.difficulty = cls._estimate_difficulty(chars, task)

        # Determine expected tool sequence
        chars.expected_tool_sequence = cls._suggest_tool_sequence(chars, domain)

        return chars

    @classmethod
    def _domain_specific_analysis(
        cls, chars: TaskCharacteristics, task: dict, domain: str, text: str
    ) -> TaskCharacteristics:
        """Apply domain-specific analysis rules."""

        if domain == "medical_qa":
            # Medical QA usually requires evidence search
            chars.requires_evidence_search = True
            chars.requires_knowledge_retrieval = True
            if chars.is_multiple_choice:
                chars.requires_tool_chain = False
            else:
                chars.is_multi_step = True

        elif domain == "clinical_diagnosis":
            chars.is_multi_step = True
            chars.requires_tool_chain = True
            chars.requires_evidence_search = True

        elif domain == "drug_interaction":
            chars.requires_evidence_search = True
            chars.requires_knowledge_retrieval = True
            chars.requires_tool_chain = True

        elif domain == "ehr_management":
            chars.is_multi_step = True
            chars.requires_tool_chain = True
            chars.requires_calculation = True

        elif domain == "triage_emergency":
            chars.is_multi_step = True
            chars.requires_tool_chain = True

        elif domain == "visual_diagnosis":
            chars.requires_image_analysis = True
            chars.requires_evidence_search = True

        elif domain == "radiology_report":
            chars.requires_image_analysis = True
            chars.is_multi_step = True
            chars.requires_tool_chain = True

        elif domain == "psychiatry":
            chars.is_multi_step = True
            chars.requires_tool_chain = True
            chars.requires_evidence_search = True

        elif domain == "obstetrics":
            chars.is_multi_step = True
            chars.requires_tool_chain = True
            chars.requires_evidence_search = True

        return chars

    @classmethod
    def _extract_topic_keywords(cls, text: str) -> list[str]:
        """Extract medical topic keywords for search guidance."""
        # Simple keyword extraction: find medical terms
        medical_terms = set()
        # Remove common stop words and find 2+ char words
        words = re.findall(r"[a-z]{3,}", text)
        stop = {
            "the", "and", "for", "with", "from", "that", "this",
            "which", "what", "how", "are", "was", "were", "has",
            "have", "been", "can", "will", "should", "would",
            "patient", "following", "most", "likely", "correct",
            "answer", "question", "option", "choice", "except",
            "all", "none", "best", "common", "include", "cause",
        }
        for w in words:
            if w not in stop and len(w) > 3:
                medical_terms.add(w)

        # Return top keywords by frequency
        from collections import Counter
        word_freq = Counter(words)
        ranked = [w for w, _ in word_freq.most_common(20) if w in medical_terms]
        return ranked[:8]

    @classmethod
    def _estimate_difficulty(cls, chars: TaskCharacteristics, task: dict) -> str:
        """Estimate task difficulty based on characteristics."""
        score = 0
        if chars.is_multi_step:
            score += 2
        if chars.requires_tool_chain:
            score += 1
        if chars.requires_calculation:
            score += 1
        if chars.requires_image_analysis:
            score += 1
        if chars.is_open_ended:
            score += 1

        # Check task metadata
        meta = task.get("metadata", {})
        if isinstance(meta, dict):
            if meta.get("difficulty") == "hard":
                score += 2
            elif meta.get("difficulty") == "expert":
                score += 3

        if score <= 1:
            return "easy"
        elif score <= 3:
            return "moderate"
        elif score <= 5:
            return "hard"
        else:
            return "expert"

    @classmethod
    def _suggest_tool_sequence(
        cls, chars: TaskCharacteristics, domain: str
    ) -> list[str]:
        """Suggest an optimal tool usage sequence."""
        sequence = []

        if domain == "medical_qa":
            if chars.requires_evidence_search:
                sequence.append("search")
            if chars.requires_knowledge_retrieval:
                sequence.append("search_evidence")
            sequence.append("submit_answer")

        elif domain == "clinical_diagnosis":
            sequence.extend([
                "get_patient_history",
                "get_vital_signs",
                "order_lab_test",
                "search",  # search for differential
            ])
            if chars.requires_evidence_search:
                sequence.append("search_evidence")
            sequence.append("submit_answer")

        elif domain == "drug_interaction":
            sequence.extend([
                "get_medication_list",
                "check_interaction",
                "search",  # drug interaction evidence
            ])
            sequence.append("submit_answer")

        elif domain == "ehr_management":
            sequence.extend([
                "query_lab_results",
                "query_vitals",
                "get_medication_list",
            ])
            if chars.requires_calculation:
                sequence.append("calculate_clinical_score")
            sequence.append("submit_answer")

        elif domain == "triage_emergency":
            sequence.extend([
                "assess_vitals",
                "get_chief_complaint",
                "perform_primary_survey",
            ])
            sequence.append("submit_answer")

        else:
            if chars.requires_evidence_search:
                sequence.append("search")
            sequence.append("submit_answer")

        return sequence


# ============================================================
# 2. Tool Guidance Generator
# ============================================================

class ToolGuidance:
    """Generates adaptive tool usage guidance text for system prompts."""

    @classmethod
    def generate(
        cls,
        domain: str,
        task: dict,
        tools: list[dict],
        agent_profile: Optional[dict] = None,
        reward_strategy: str = "grpo",
    ) -> str:
        """Generate adaptive tool usage guidance.

        Args:
            domain: Task domain name
            task: Task dictionary
            tools: Available tool definitions (OpenAI format)
            agent_profile: Agent's reflection/profile (from logbook)
            reward_strategy: Current reward strategy being used

        Returns:
            Guidance text to inject into system prompt
        """
        chars = TaskAnalyzer.analyze(task, domain)
        tool_names = cls._extract_tool_names(tools)

        sections = []

        # 1. Task-Specific Strategy
        sections.append(cls._task_strategy_section(chars, domain))

        # 2. Tool Usage Priority
        sections.append(cls._tool_priority_section(chars, tool_names, domain))

        # 3. Knowledge Search Guidance
        if chars.requires_evidence_search or chars.requires_knowledge_retrieval:
            sections.append(cls._knowledge_search_section(chars, tool_names))

        # 4. Weakness-Aware Guidance (if profile available)
        if agent_profile:
            weakness_guidance = cls._weakness_guidance_section(
                agent_profile, domain, chars
            )
            if weakness_guidance:
                sections.append(weakness_guidance)

        # 5. Reward Strategy Hints
        sections.append(cls._reward_strategy_hints(reward_strategy, chars))

        # 6. Anti-Pattern Warnings
        sections.append(cls._anti_pattern_section(chars, domain))

        guidance = "\n\n".join(s for s in sections if s)
        return f"\n\n## Adaptive Tool Usage Guide\n{guidance}"

    @classmethod
    def _extract_tool_names(cls, tools: list[dict]) -> list[str]:
        """Extract tool names from OpenAI-format tool definitions."""
        names = []
        for t in tools:
            func = t.get("function", {})
            name = func.get("name", "")
            if name:
                names.append(name)
        return names

    @classmethod
    def _task_strategy_section(cls, chars: TaskCharacteristics, domain: str) -> str:
        """Generate task-specific strategy advice."""
        lines = ["### Strategy"]

        if chars.is_multiple_choice:
            lines.append(
                "This is a multiple-choice question. "
                "Search for evidence to eliminate wrong options before answering. "
                "Do NOT guess — use tools to verify your reasoning."
            )
        elif chars.is_open_ended:
            lines.append(
                "This is an open-ended task requiring comprehensive analysis. "
                "Gather sufficient evidence from multiple sources before synthesizing your answer. "
                "Structure your response with clear clinical reasoning."
            )

        if chars.is_multi_step:
            lines.append(
                "This task requires multiple steps. Follow a systematic approach: "
                "gather information → analyze findings → search for supporting evidence → "
                "synthesize and conclude."
            )

        if chars.difficulty in ("hard", "expert"):
            lines.append(
                f"Difficulty: **{chars.difficulty}**. Take extra care with this task. "
                "Consider edge cases, verify your assumptions with evidence, "
                "and cross-reference multiple sources."
            )

        return "\n".join(lines)

    @classmethod
    def _tool_priority_section(
        cls, chars: TaskCharacteristics, tool_names: list[str], domain: str
    ) -> str:
        """Generate tool usage priority guidance."""
        lines = ["### Tool Priority"]

        # Categorize tools
        search_tools = [t for t in tool_names if "search" in t.lower()]
        browse_tools = [t for t in tool_names if "browse" in t.lower()]
        evidence_tools = [t for t in tool_names if "evidence" in t.lower()]
        clinical_tools = [
            t for t in tool_names
            if t not in search_tools + browse_tools + evidence_tools
            and t != "submit_answer"
        ]
        submit_tools = [t for t in tool_names if "submit" in t.lower()]

        # Suggest optimal sequence
        if chars.expected_tool_sequence:
            available_sequence = [
                t for t in chars.expected_tool_sequence if t in tool_names
            ]
            if available_sequence:
                seq_str = " → ".join(f"`{t}`" for t in available_sequence)
                lines.append(f"Recommended sequence: {seq_str}")

        # Domain-specific tool priorities
        if chars.requires_evidence_search:
            if search_tools:
                lines.append(
                    f"**High Priority**: Use {', '.join(f'`{t}`' for t in search_tools)} "
                    "to find relevant medical evidence BEFORE answering."
                )
            if evidence_tools:
                lines.append(
                    f"**Evidence Retrieval**: Use {', '.join(f'`{t}`' for t in evidence_tools)} "
                    "for PubMed/PMC literature and medical textbook passages."
                )

        if clinical_tools:
            lines.append(
                f"**Clinical Tools**: {', '.join(f'`{t}`' for t in clinical_tools[:5])}"
            )

        if submit_tools:
            lines.append(
                f"**Submit**: Use `{submit_tools[0]}` when you have gathered "
                "sufficient evidence and formed your answer."
            )

        return "\n".join(lines)

    @classmethod
    def _knowledge_search_section(
        cls, chars: TaskCharacteristics, tool_names: list[str]
    ) -> str:
        """Generate knowledge search guidance."""
        lines = ["### Knowledge Search Tips"]

        # Topic-specific search advice
        if chars.topic_keywords:
            keywords = ", ".join(f'"{kw}"' for kw in chars.topic_keywords[:5])
            lines.append(
                f"Key topics for this task: {keywords}. "
                "Use these as search queries."
            )

        # Search strategy
        lines.append(
            "**Search Strategy**:\n"
            "1. Start with a broad search using the main medical concept\n"
            "2. Refine with specific terms if initial results are too general\n"
            "3. Search for contradicting evidence to verify your hypothesis\n"
            "4. If `search_evidence` is available, use it for PubMed/PMC literature"
        )

        # Specific search tools
        if "search" in tool_names:
            lines.append(
                "The `search` tool queries across Wikipedia, PubMed evidence, "
                "medical textbooks, and biomedical instruction databases simultaneously."
            )
        if "search_evidence" in tool_names:
            lines.append(
                "The `search_evidence` tool searches 581K+ MedCPT evidence passages "
                "from PubMed/PMC. Use this for clinical evidence and study findings."
            )

        return "\n".join(lines)

    @classmethod
    def _weakness_guidance_section(
        cls, profile: dict, domain: str, chars: TaskCharacteristics
    ) -> str:
        """Generate guidance based on agent's known weaknesses."""
        weaknesses = profile.get("weaknesses", [])
        error_patterns = profile.get("error_patterns", {})
        confidence_map = profile.get("confidence_map", {})

        if not weaknesses and not error_patterns:
            return ""

        lines = ["### Performance-Aware Guidance"]

        # Domain-specific weakness
        if domain in weaknesses:
            confidence = confidence_map.get(domain, 0.0)
            lines.append(
                f"⚠ You have historically struggled with `{domain}` tasks "
                f"(confidence: {confidence:.0%}). "
                "Pay extra attention to the following:"
            )

        # Error pattern-specific guidance
        if error_patterns.get("premature_stop", 0) > 2:
            lines.append(
                "- **Avoid premature stopping**: You tend to submit answers too early. "
                "Make sure to use at least 2-3 tools before concluding."
            )
        if error_patterns.get("over_investigation", 0) > 2:
            lines.append(
                "- **Be efficient**: You sometimes use too many tools. "
                "Focus on the most relevant tools and submit when confident."
            )
        if error_patterns.get("reasoning_error", 0) > 2:
            lines.append(
                "- **Verify reasoning**: Cross-check your conclusions with "
                "evidence from search tools. Don't rely on assumptions alone."
            )

        return "\n".join(lines) if len(lines) > 1 else ""

    @classmethod
    def _reward_strategy_hints(
        cls, reward_strategy: str, chars: TaskCharacteristics
    ) -> str:
        """Generate hints aligned with the current reward strategy."""
        lines = ["### Optimization Hints"]

        if reward_strategy == "sarl":
            lines.append(
                "Your performance is evaluated on both the final answer AND "
                "how effectively you use tools. Each tool call should have "
                "a clear purpose. Self-assess your confidence before submitting."
            )
        elif reward_strategy == "mrpo":
            lines.append(
                "Quality of reasoning matters at every step. "
                "Produce well-structured intermediate thoughts. "
                "Each response should show clear clinical reasoning."
            )
        elif reward_strategy == "adaptive":
            lines.append(
                "This task uses adaptive evaluation. Focus on: "
                "(1) accuracy of your final answer, "
                "(2) quality of your reasoning process, "
                "(3) appropriate and efficient tool usage."
            )
        else:
            # GRPO default
            lines.append(
                "Focus on: accurate answers, correct tool usage, "
                "clear reasoning format, and safe clinical recommendations."
            )

        return "\n".join(lines)

    @classmethod
    def _anti_pattern_section(cls, chars: TaskCharacteristics, domain: str) -> str:
        """Generate anti-pattern warnings."""
        lines = ["### Common Pitfalls to Avoid"]

        lines.append(
            "- Do NOT call the same tool with identical arguments more than once\n"
            "- Do NOT submit an answer without using any tools first\n"
            "- Do NOT ignore tool results — incorporate findings into your reasoning\n"
            "- Do NOT repeat the question back — focus on answering it"
        )

        if domain == "medical_qa":
            lines.append(
                "- Do NOT guess when you can search for evidence\n"
                "- Always verify your chosen answer against retrieved evidence"
            )
        elif domain in ("clinical_diagnosis", "triage_emergency"):
            lines.append(
                "- Do NOT skip vital signs or history review\n"
                "- Always consider life-threatening conditions first"
            )
        elif domain == "drug_interaction":
            lines.append(
                "- Do NOT assess interactions without checking ALL medications\n"
                "- Consider severity levels (minor, moderate, major, contraindicated)"
            )

        return "\n".join(lines)


# ============================================================
# 3. Guidance Injector
# ============================================================

class GuidanceInjector:
    """Injects adaptive guidance into system prompts.

    Can be used as a hook in AgentRunner.build_system_prompt() or
    the GYM environment's reset observation.
    """

    def __init__(
        self,
        agent_profile: Optional[dict] = None,
        reward_strategy: str = "grpo",
    ):
        self.agent_profile = agent_profile
        self.reward_strategy = reward_strategy

    def inject(
        self,
        system_prompt: str,
        domain: str,
        task: dict,
        tools: list[dict],
    ) -> str:
        """Inject adaptive guidance into an existing system prompt.

        Args:
            system_prompt: Original system prompt
            domain: Task domain
            task: Task dictionary
            tools: Tool definitions

        Returns:
            Enhanced system prompt with adaptive guidance
        """
        guidance = ToolGuidance.generate(
            domain=domain,
            task=task,
            tools=tools,
            agent_profile=self.agent_profile,
            reward_strategy=self.reward_strategy,
        )
        return system_prompt + guidance

    def generate_standalone(
        self, domain: str, task: dict, tools: list[dict]
    ) -> str:
        """Generate standalone guidance text (without base prompt)."""
        return ToolGuidance.generate(
            domain=domain,
            task=task,
            tools=tools,
            agent_profile=self.agent_profile,
            reward_strategy=self.reward_strategy,
        )
