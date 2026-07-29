"""Generate comprehensive tool_config.yaml from BIOAgents domain toolkits.

Auto-generates veRL-compatible tool schemas from all BIOAgents @is_tool methods.
Supports two modes:
1. Full: All 179 tools in one config (for tool_config.yaml)
2. Per-domain: Separate tool lists per domain (for system message injection)
"""

import json
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, os.environ.get("HCGYM_ROOT", str(Path(__file__).resolve().parents[2])))

# Domain name mapping: data domain -> BIOAgents module
DOMAIN_MAP = {
    "text_qa": "medical_qa",
    "clinical_diagnosis": "clinical_diagnosis",
    "multimodal_vqa": "visual_diagnosis",
    "drug_interaction": "drug_interaction",
    "ehr_management": "ehr_management",
    "triage_emergency": "triage_emergency",
    "psychiatry": "psychiatry",
    "obstetrics": "obstetrics",
    "radiology_report": "radiology_report",
}


def load_domain_tools(domain_name: str) -> list[dict]:
    """Load tool definitions from a BIOAgents domain toolkit."""
    try:
        if domain_name == "medical_qa":
            from bioagents.domains.medical_qa.tools import MedicalQATools
            from bioagents.domains.medical_qa.db import MedicalQADB
            toolkit = MedicalQATools(MedicalQADB())
        elif domain_name == "clinical_diagnosis":
            from bioagents.domains.clinical_diagnosis.tools import ClinicalTools
            from bioagents.domains.clinical_diagnosis.db import ClinicalDB
            toolkit = ClinicalTools(ClinicalDB())
        elif domain_name == "visual_diagnosis":
            from bioagents.domains.visual_diagnosis.tools import VisualDiagnosisTools
            from bioagents.domains.visual_diagnosis.db import VisualDiagnosisDB
            toolkit = VisualDiagnosisTools(VisualDiagnosisDB())
        elif domain_name == "drug_interaction":
            from bioagents.domains.drug_interaction.tools import DrugInteractionTools
            from bioagents.domains.drug_interaction.db import DrugInteractionDB
            toolkit = DrugInteractionTools(DrugInteractionDB())
        elif domain_name == "ehr_management":
            from bioagents.domains.ehr_management.tools import EHRTools
            from bioagents.domains.ehr_management.db import EHRDB
            toolkit = EHRTools(EHRDB())
        elif domain_name == "triage_emergency":
            from bioagents.domains.triage_emergency.tools import TriageEmergencyTools
            from bioagents.domains.triage_emergency.db import TriageEmergencyDB
            toolkit = TriageEmergencyTools(TriageEmergencyDB())
        elif domain_name == "psychiatry":
            from bioagents.domains.psychiatry.tools import PsychiatryTools
            from bioagents.domains.psychiatry.db import PsychiatryDB
            toolkit = PsychiatryTools(PsychiatryDB())
        elif domain_name == "obstetrics":
            from bioagents.domains.obstetrics.tools import ObstetricsTools
            from bioagents.domains.obstetrics.db import ObstetricsDB
            toolkit = ObstetricsTools(ObstetricsDB())
        elif domain_name == "radiology_report":
            from bioagents.domains.radiology_report.tools import RadiologyReportTools
            from bioagents.domains.radiology_report.db import RadiologyReportDB
            toolkit = RadiologyReportTools(RadiologyReportDB())
        else:
            print(f"Unknown domain: {domain_name}")
            return []

        return toolkit.get_tool_definitions_dict()
    except Exception as e:
        print(f"Error loading {domain_name}: {e}")
        return []


def tool_def_to_verl_entry(tool_def: dict, domain: str) -> dict:
    """Convert BIOAgents ToolDefinition dict to veRL tool_config.yaml entry."""
    return {
        "class_name": "bioagents_tool.BIOAgentsMedicalTool",
        "config": {
            "type": "native",
            "domain": domain,
        },
        "tool_schema": tool_def,
    }


def generate_full_config(output_path: str):
    """Generate tool_config.yaml with all tools from all domains."""
    all_tools = []
    domain_counts = {}

    for data_domain, bio_domain in DOMAIN_MAP.items():
        tools = load_domain_tools(bio_domain)
        domain_counts[bio_domain] = len(tools)
        for t in tools:
            all_tools.append(tool_def_to_verl_entry(t, bio_domain))

    config = {"tools": all_tools}

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nGenerated {output_path} with {len(all_tools)} tools:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")


def generate_domain_tool_descriptions(output_path: str):
    """Generate JSON file mapping domain -> list of tool descriptions for system messages."""
    domain_tools = {}

    for data_domain, bio_domain in DOMAIN_MAP.items():
        tools = load_domain_tools(bio_domain)
        # For system message: just name + description (compact)
        tool_list = []
        for t in tools:
            func = t.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            param_names = list(params.keys())
            tool_list.append({
                "name": name,
                "description": desc,
                "parameters": param_names,
            })
        domain_tools[data_domain] = tool_list

    with open(output_path, "w") as f:
        json.dump(domain_tools, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {output_path}")
    for domain, tools in domain_tools.items():
        print(f"  {domain}: {len(tools)} tools")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    print("=== Generating full tool_config.yaml ===")
    generate_full_config(str(base_dir / "tool_config_full.yaml"))

    print("\n=== Generating domain tool descriptions ===")
    generate_domain_tool_descriptions(str(base_dir / "domain_tools.json"))
