# Third-Party Licenses

This document lists all third-party components used in BIOAgents,
their respective licenses, and any special restrictions.

> **Last updated**: 2026-02-13

---

## 1. Git Submodules (Included in Repository)

| Component | License | Copyright | URL |
|---|---|---|---|
| AgentGym-RL | MIT | 2025 FudanNLP-Agent | https://github.com/WooooDyy/AgentGym-RL |
| tau2-bench | MIT | 2025 Sierra Research | https://github.com/sierra-research/tau2-bench |

---

## 2. Python Dependencies (Installed via pip)

| Package | License | Copyright | Notes |
|---|---|---|---|
| torch (PyTorch) | BSD-3-Clause | Meta Platforms, Inc. | |
| transformers | Apache-2.0 | HuggingFace Inc. | |
| trl | Apache-2.0 | HuggingFace Inc. | |
| peft | Apache-2.0 | HuggingFace Inc. | |
| datasets | Apache-2.0 | HuggingFace Inc. | Library license; individual datasets have own licenses |
| accelerate | Apache-2.0 | HuggingFace Inc. | |
| vllm | Apache-2.0 | vLLM contributors | |
| deepspeed | Apache-2.0 | Microsoft Corporation | |
| gymnasium | MIT | Farama Foundation | |
| pydantic | MIT | Pydantic contributors | |
| numpy | BSD-3-Clause | NumPy Developers | |
| pandas | BSD-3-Clause | pandas Development Team | |
| scikit-learn | BSD-3-Clause | scikit-learn developers | |
| pillow | HPND (MIT-like) | Jeffrey A. Clark / Pillow contributors | |
| bert-score | MIT | Tianyi Zhang et al. | |
| rouge-score | Apache-2.0 | Google LLC | |
| nltk | Apache-2.0 | NLTK Project | |
| wandb (SDK) | MIT | Weights & Biases, Inc. | SDK only; W&B platform has separate commercial ToS |
| loguru | MIT | Delgan | |
| rich | MIT | Will McGugan | |
| httpx | BSD-3-Clause | Encode OSS Ltd. | |
| tqdm | MPL-2.0 / MIT | tqdm developers | Dual licensed |
| pyyaml | MIT | Ingy döt Net / Kirill Simonov | |
| flash-attn | BSD-3-Clause | Tri Dao et al. | Optional dependency |

All Python dependencies are compatible with Apache-2.0.

---

## 3. Pre-trained Models (Downloaded at Runtime)

| Model | License | Copyright | Restrictions |
|---|---|---|---|
| BiomedBERT (microsoft/BiomedNLP-BiomedBERT) | MIT | Microsoft Corporation | None |
| Qwen3 series (Qwen/Qwen3-*) | Apache-2.0 | Alibaba Cloud | |
| Qwen2.5-VL series | Apache-2.0 | Alibaba Cloud | |
| Lingshu-MedLLM | Varies | Check model card | Verify before use |

---

## 4. Data Sources (in databases/ — NOT redistributed via git)

### 4.1 Self-BioRAG Data (databases/critic/, databases/generator/)

| Source | License | Restrictions | Risk |
|---|---|---|---|
| Self-BioRAG (dmis-lab/self-biorag) | **No explicit license** | Built on LLaMA-2 (Meta Community License). Training data involves GPT-4 outputs (OpenAI ToS). | **HIGH** |

**Action**: These files are excluded from the repository via `.gitignore`.
Users must obtain this data directly from the original source and comply
with Meta's LLaMA-2 Community License and OpenAI's Terms of Service.

### 4.2 Instruction Data (databases/instruction/)

| Dataset | License | Copyright | Notes |
|---|---|---|---|
| MedInstruct-52k (AlpaCare) | Apache-2.0 | XZhang Lab | Data generated via GPT-4/ChatGPT; OpenAI ToS may apply to downstream model training |
| all_biomedical_instruction.json | Varies | Multiple sources | Aggregated from multiple instruction datasets |
| mol_instruction_qa.json | Varies | Check source | |
| self_instruct_biomedical.json | Varies | Check source | |

**Action**: Excluded from repository. Users must verify individual dataset
licenses before use in model training.

### 4.3 Retriever Data (databases/retriever/)

| Dataset | License | Copyright | Notes |
|---|---|---|---|
| MedCPT evidence | **Public Domain** (US Government Work) | NCBI/NLM | No restrictions; cite original authors |

### 4.4 Wikipedia Dumps (databases/wiki*/)

| Dataset | License | Copyright | Notes |
|---|---|---|---|
| Wikipedia 2018/2026 dumps | **CC-BY-SA 4.0** + **GFDL** | Wikimedia Foundation / Contributors | Share-alike clause: derived textual content must remain CC-BY-SA. Code using Wikipedia as a knowledge source is not affected. |

---

## 5. Evaluation Benchmark Datasets (Downloaded at Runtime)

| Benchmark | License | Copyright | URL | Notes |
|---|---|---|---|---|
| MedQA (USMLE) | MIT | Jin et al. | https://github.com/jind11/MedQA | |
| MedMCQA | MIT / Apache-2.0 | Pal et al. | https://medmcqa.github.io/ | |
| MMLU (medical) | MIT | Hendrycks et al. | https://github.com/hendrycks/test | |
| PubMedQA | MIT | Jin et al. | https://pubmedqa.github.io/ | |
| BioASQ | CC-BY-2.5 | BioASQ Consortium | http://bioasq.org/ | Requires registration; cite NLM |
| PathVQA | MIT | He et al. | https://github.com/UCSD-AI4H/PathVQA | Image copyrights belong to original publishers |
| VQA-RAD | CC0 1.0 (Public Domain) | Lau et al. | https://osf.io/89kps/ | No restrictions |
| SLAKE | CC-BY-4.0 | Liu et al. | https://www.med-vqa.com/slake/ | Attribution required |

All benchmark datasets are compatible with Apache-2.0 for evaluation use.
Note: Benchmark datasets are used for evaluation only and are not
redistributed as part of this repository.

---

## 6. Medical Data (NOT Included — Obtained Separately)

| Dataset | License | Restrictions |
|---|---|---|
| MIMIC-III | PhysioNet Credentialed Health Data License 1.5.0 | **Cannot redistribute**. Requires PhysioNet credentialing, CITI training, data use agreement. Research use only. |
| MIMIC-IV | PhysioNet Credentialed Health Data License 1.5.0 | Same as MIMIC-III |
| eICU | PhysioNet Credentialed Health Data License 1.5.0 | Same as MIMIC-III |

**Important**: No PhysioNet data is included in this repository. The synthetic
patient data in `data/domains/*/db.json` is **entirely fictional** and was
created specifically for this project. It does not contain or derive from any
real patient records.

---

## 7. AI-Generated Code

| Tool | Terms | Coverage |
|---|---|---|
| Anthropic Claude (Opus 4.6, Commercial API via Cursor) | Anthropic Commercial Terms (June 2025): Customer owns all outputs | Portions of codebase were AI-assisted; all code reviewed and modified by human authors |

Per Anthropic's Commercial Terms of Service (Section B), all right, title,
and interest in generated outputs are assigned to the customer. The generated
code is released under this project's Apache-2.0 license.

---

## 8. Research Paper References

The following papers informed the design of this system. Their methodologies
are referenced but their code is independently implemented:

| Paper | arXiv ID | Relevance |
|---|---|---|
| FairGRPO | 2510.19893 | Fairness-aware GRPO mechanism |
| DiagGym | 2510.24654 | Competitor — diagnostic agent gym |
| MedAgentGym | 2506.04405 | Competitor — code-centric medical agent gym |
| AgentGym | 2407.???? | Multi-environment agent RL framework |
| MedAgentBench | 2501.14654 | EHR agent benchmark reference |
| AgentClinic | 2405.07960 | Patient agent / cognitive bias reference |
| DoctorAgent-RL | 2505.19630 | Multi-agent RL for clinical dialogue |
| CARES | 2024 | Medical AI safety benchmark reference |

---

## License Compatibility Summary

```
Apache-2.0 (this project)
├── Compatible: MIT, BSD-3-Clause, Apache-2.0, CC0, CC-BY, MPL-2.0
├── Conditionally compatible: CC-BY-SA (text content only, share-alike)
└── NOT compatible for data redistribution:
    ├── PhysioNet Credentialed License (MIMIC data)
    ├── Meta LLaMA-2 Community License (Self-BioRAG weights)
    └── OpenAI ToS restrictions (GPT-4-generated training data)
```

The core BIOAgents source code, task definitions, and evaluation framework
are fully licensed under Apache-2.0 with no encumbrances. Third-party data
that carries restrictive licenses is excluded from the repository and must
be obtained independently by users.
