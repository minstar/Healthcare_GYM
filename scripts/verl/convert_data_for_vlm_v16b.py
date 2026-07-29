"""Convert BIOAgents parquet data for veRL VLM training (v16b).

v16b improvements over v16:
    - Think-first instruction: encourages reasoning before tool use
    - submit_answer enforcement: always end with submit_answer call
    - Minimize tool errors: plan tool calls carefully
"""

import json
import re
import os
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path



# Domain-specific system prompts for thinking + tool use mode
DOMAIN_SYSTEM_PROMPTS = {
    "text_qa": (
        "You are a medical AI assistant with access to medical knowledge tools. "
        "Think carefully through the clinical reasoning before answering. "
        "Use available tools to search for evidence when needed. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, provide a concise evidence-based answer."
    ),
    "multimodal_vqa": (
        "You are a medical AI assistant capable of analyzing medical images. "
        "Think carefully about visual findings and their clinical significance. "
        "Use tools to search for similar cases or reference information when needed. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, describe findings and provide your assessment."
    ),
    "clinical_diagnosis": (
        "You are a clinical AI assistant. Think through the differential diagnosis systematically. "
        "Use tools to retrieve patient information, lab results, vital signs, and clinical guidelines. "
        "Consider all available data before making diagnostic decisions. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, provide your clinical assessment with reasoning."
    ),
    "drug_interaction": (
        "You are a pharmacology AI assistant. Think carefully about drug mechanisms and interactions. "
        "Use tools to check drug information, interactions, CYP450 metabolism, and dosage adjustments. "
        "Consider patient-specific factors when making recommendations. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, provide your recommendation with clinical reasoning."
    ),
    "ehr_management": (
        "You are a clinical informatics AI assistant. Analyze electronic health records systematically. "
        "Use tools to review patient summaries, lab trends, vital signs, clinical scores, and quality indicators. "
        "Synthesize information across multiple data sources for comprehensive assessment. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, provide your clinical analysis with supporting data."
    ),
    "triage_emergency": (
        "You are an emergency medicine AI assistant. Think quickly but thoroughly about patient acuity. "
        "Use tools to assess presentations, calculate severity scores (GCS, ESI, qSOFA), and check protocols. "
        "Prioritize life-threatening conditions and time-critical interventions. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, provide your triage decision with clinical reasoning."
    ),
    "psychiatry": (
        "You are a psychiatric AI assistant. Think carefully about the patient's mental health presentation. "
        "Use tools for mental status exams, screening instruments (PHQ-9, GAD-7), suicide risk assessment, and treatment guidelines. "
        "Consider biopsychosocial factors in your assessment. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, provide your psychiatric assessment and management plan."
    ),
    "obstetrics": (
        "You are an obstetrics/gynecology AI assistant. Think carefully about maternal and fetal well-being. "
        "Use tools to assess fetal status, labor progress, prenatal labs, and medication safety in pregnancy. "
        "Follow evidence-based obstetric guidelines and protocols. "
        "For multiple choice questions, end with 'Answer: X' where X is the letter. "
        "For open-ended questions, provide your obstetric assessment and management plan."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a medical AI assistant with access to medical knowledge tools. "
    "Think carefully through the clinical reasoning before answering. "
    "Use available tools to search for evidence when needed. "
    "For multiple choice questions, end with 'Answer: X' where X is the letter. "
    "For open-ended questions, provide a concise evidence-based answer."
)

# v16b: Think-first + submit_answer enforcement instructions
_V16B_SUFFIX = (
    "\n\nIMPORTANT WORKFLOW GUIDELINES:\n"
    "1. THINK FIRST: Before using any search or retrieval tools, reason through what you "
    "already know about the question. Formulate your initial hypothesis, then use tools to "
    "verify or refine it.\n"
    "2. ALWAYS SUBMIT AN ANSWER: You must always call submit_answer as your final action. "
    "Never end your response without submitting an answer, even if you are uncertain. "
    "Provide your best assessment based on available evidence.\n"
    "3. MINIMIZE TOOL ERRORS: Plan your tool calls carefully. Check required parameters "
    "before calling. If a tool returns an error, do not retry the same call — adapt your "
    "approach or reason from what you already know."
)


def get_system_prompt(domain: str) -> str:
    """Get domain-specific system prompt with v16b improvements.

    Tool schemas are injected at runtime by veRL's agent loop via tool_config_path.
    System prompts only contain domain-specific instructions.
    """
    base = DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)
    return base + _V16B_SUFFIX


def convert_row(row):
    """Convert a single row's prompt, extract images, and update system prompt."""
    prompt = row["prompt"]
    extra_info = row.get("extra_info", {})
    domain = extra_info.get("domain", "text_qa") if isinstance(extra_info, dict) else "text_qa"
    images = []
    new_prompt = []

    for msg in prompt:
        content = msg.get("content", "")
        role = msg.get("role", "user")

        # Update system prompt for thinking + tool use mode
        if role == "system":
            content = get_system_prompt(domain)
            new_prompt.append({"role": role, "content": content})
            continue

        # Check for IMAGE: path pattern
        match = re.search(r"IMAGE:\s*(\S+)", content)
        if match:
            image_path = match.group(1)
            new_content = re.sub(r"IMAGE:\s*\S+\s*\n?", "<image>\n", content)

            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                images.append({"bytes": img_bytes})
            else:
                print(f"WARNING: Image not found: {image_path}")
                img = Image.new("RGB", (28, 28), color=(128, 128, 128))
                buf = BytesIO()
                img.save(buf, format="PNG")
                images.append({"bytes": buf.getvalue()})

            new_prompt.append({"role": role, "content": new_content.strip()})
        else:
            new_prompt.append({"role": role, "content": content})

    return new_prompt, images if images else None


def convert_parquet(input_path, output_path):
    """Convert an entire parquet file."""
    df = pd.read_parquet(input_path)
    print(f"Input: {len(df)} rows from {input_path}")

    new_prompts = []
    all_images = []
    domain_counts = {}

    for i in range(len(df)):
        row = df.iloc[i]
        extra_info = row.get("extra_info", {})
        domain = extra_info.get("domain", "text_qa") if isinstance(extra_info, dict) else "text_qa"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        new_prompt, images = convert_row(row)
        new_prompts.append(new_prompt)
        all_images.append(images)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(df)} rows")

    df_out = df.copy()
    # Ensure prompts are Python lists, not numpy arrays (Jinja template compat)
    df_out["prompt"] = [list(p) for p in new_prompts]
    df_out["images"] = all_images

    with_images = sum(1 for img in all_images if img is not None)
    print(f"Rows with images: {with_images}, text-only: {len(df) - with_images}")
    print(f"Domain distribution: {domain_counts}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"Output: {output_path}")
    return df_out


if __name__ == "__main__":
    base = os.environ.get("HCGYM_RUN_ROOT", str(Path(__file__).resolve().parents[2])) + "/data/verl_parquet"
    input_dir = f"{base}/full_4modality"
    output_dir = f"{base}/full_4modality_vlm_v16b"

    for split in ["train", "test"]:
        input_path = f"{input_dir}/{split}.parquet"
        output_path = f"{output_dir}/{split}.parquet"
        if os.path.exists(input_path):
            convert_parquet(input_path, output_path)
            print()
