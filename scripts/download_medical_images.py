#!/usr/bin/env python3
"""Download Medical Imaging Datasets for BIOAgents RL Training.

Downloads curated medical imaging datasets that can be used as tool call
resources during RL training. When agents call tools like `analyze_medical_image`
or `get_image_report`, these datasets provide real medical images and metadata
for realistic simulation.

Selected from: https://github.com/sfikas/medical-imaging-datasets

Usage:
    python scripts/download_medical_images.py --all
    python scripts/download_medical_images.py --datasets vqa_rad slake medmnist
    python scripts/download_medical_images.py --datasets chest_xray --max-samples 200
    python scripts/download_medical_images.py --list
    python scripts/download_medical_images.py --catalog-only
"""

import argparse
import json
import os
import sys
import time
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "medical_images"
CATALOG_PATH = DATA_DIR / "catalog.json"

# ══════════════════════════════════════════════════════════════
#  Registry
# ══════════════════════════════════════════════════════════════

DATASET_REGISTRY = {
    "vqa_rad": {
        "name": "VQA-RAD",
        "description": "Radiology VQA — 315 images (X-ray, CT, MRI) with 3,515 QA pairs",
        "source": "HuggingFace: flaviagiammarino/vqa-rad",
        "modalities": ["xray", "ct", "mri"],
        "size_estimate": "~100MB",
        "license": "CC BY 4.0",
        "tool_use": ["analyze_medical_image", "get_image_report"],
    },
    "slake": {
        "name": "SLAKE",
        "description": "Multilingual Medical VQA — 642 images across organ systems",
        "source": "HuggingFace: BoKelvin/SLAKE",
        "modalities": ["xray", "ct", "mri"],
        "size_estimate": "~200MB",
        "license": "CC BY-NC-SA 4.0",
        "tool_use": ["analyze_medical_image", "get_patient_context"],
    },
    "pathvqa": {
        "name": "PathVQA",
        "description": "Pathology VQA — 4,998 pathology images with 32,799 QA pairs",
        "source": "HuggingFace: flaviagiammarino/path-vqa",
        "modalities": ["pathology"],
        "size_estimate": "~500MB",
        "license": "CC BY 4.0",
        "tool_use": ["analyze_medical_image", "search_imaging_knowledge"],
    },
    "pmc_vqa": {
        "name": "PMC-VQA",
        "description": "PubMedCentral VQA — Medical literature images with MC questions",
        "source": "HuggingFace: RadGenome/PMC-VQA",
        "modalities": ["mixed"],
        "size_estimate": "~1GB",
        "license": "CC BY 4.0",
        "tool_use": ["analyze_medical_image", "search_imaging_knowledge"],
    },
    "chest_xray": {
        "name": "Chest X-ray (Pneumonia)",
        "description": "Chest X-ray images for pneumonia detection",
        "source": "HuggingFace: hf-vision/chest-xray-pneumonia",
        "modalities": ["xray"],
        "size_estimate": "~1.2GB",
        "license": "CC BY 4.0",
        "tool_use": ["analyze_medical_image", "get_image_report"],
    },
    "brain_tumor": {
        "name": "Brain Tumor MRI",
        "description": "Brain MRI with 4 tumor types — glioma, meningioma, pituitary, no tumor",
        "source": "HuggingFace: masoud-ml/Brain-Tumor-MRI-Dataset",
        "modalities": ["mri"],
        "size_estimate": "~150MB",
        "license": "Public Domain",
        "tool_use": ["analyze_medical_image", "get_image_report"],
    },
    "skin_cancer": {
        "name": "Skin Cancer ISIC (HAM10000)",
        "description": "ISIC skin lesion dermoscopy — 10,015 images with 7 cancer types",
        "source": "HuggingFace: marmal88/skin_cancer",
        "modalities": ["dermoscopy"],
        "size_estimate": "~2GB",
        "license": "CC BY-NC 4.0",
        "tool_use": ["analyze_medical_image", "search_similar_cases"],
    },
    "medmnist": {
        "name": "MedMNIST v2",
        "description": "12 standardized medical image datasets (28x28): "
                       "path, chest, derma, oct, pneumonia, retina, breast, blood, "
                       "tissue, organA, organC, organS",
        "source": "medmnist pip package (https://medmnist.com/)",
        "modalities": ["pathology", "xray", "dermoscopy", "oct", "fundus",
                       "ultrasound", "microscopy", "ct"],
        "size_estimate": "~500MB total",
        "license": "CC BY 4.0",
        "tool_use": ["analyze_medical_image", "search_similar_cases"],
    },
}


def _print_registry():
    """Print available datasets."""
    print("\n" + "=" * 70)
    print("  Available Medical Imaging Datasets for BIOAgents")
    print("=" * 70)
    for key, info in DATASET_REGISTRY.items():
        print(f"\n  [{key}] {info['name']}")
        print(f"    {info['description']}")
        print(f"    Source: {info['source']}")
        print(f"    Modalities: {', '.join(info['modalities'])}")
        print(f"    Size: {info['size_estimate']}")
        print(f"    License: {info['license']}")
        print(f"    Tools: {', '.join(info['tool_use'])}")
    print(f"\n{'=' * 70}\n")


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_image(image, path: Path) -> bool:
    """Save a PIL image to file."""
    try:
        if hasattr(image, "save"):
            image.save(str(path))
            return True
    except Exception as e:
        print(f"    [WARN] Save failed {path.name}: {e}")
    return False


def _save_metadata(records: list, out_dir: Path) -> Path:
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return meta_path


# ══════════════════════════════════════════════════════════════
#  Download Functions
# ══════════════════════════════════════════════════════════════

def download_vqa_rad(max_samples: int = 500) -> dict:
    """Download VQA-RAD dataset from HuggingFace."""
    from datasets import load_dataset

    out_dir = _ensure_dir(DATA_DIR / "vqa_rad")
    img_dir = _ensure_dir(out_dir / "images")
    metadata = []

    print(f"\n[VQA-RAD] Downloading from HuggingFace...")
    for split in ["train", "test"]:
        try:
            ds = load_dataset("flaviagiammarino/vqa-rad", split=split)
            print(f"  {split}: {len(ds)} samples")

            for idx, item in enumerate(ds):
                if len(metadata) >= max_samples:
                    break

                image = item.get("image")
                if image is None:
                    continue

                img_name = f"vqarad_{split}_{idx:05d}.jpg"
                img_path = str(img_dir / img_name)
                _save_image(image, Path(img_path))

                metadata.append({
                    "id": f"vqa_rad_{split}_{idx:05d}",
                    "dataset": "VQA-RAD",
                    "image_path": img_path,
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "modality": "radiology",
                    "body_part": "mixed",
                    "split": split,
                })
        except Exception as e:
            print(f"  [ERROR] {split}: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Saved {len(metadata)} records")
    return {"dataset": "VQA-RAD", "count": len(metadata), "path": str(out_dir)}


def download_slake(max_samples: int = 500) -> dict:
    """Download SLAKE dataset from HuggingFace."""
    from datasets import load_dataset

    out_dir = _ensure_dir(DATA_DIR / "slake")
    img_dir = _ensure_dir(out_dir / "images")
    metadata = []

    print(f"\n[SLAKE] Downloading from HuggingFace...")
    for split in ["train", "test"]:
        try:
            ds = load_dataset("BoKelvin/SLAKE", split=split)
            print(f"  {split}: {len(ds)} samples")

            for idx, item in enumerate(ds):
                if len(metadata) >= max_samples:
                    break

                # English only
                if item.get("q_lang", "en") != "en":
                    continue

                image = item.get("image")
                img_name = f"slake_{split}_{idx:05d}.jpg"
                img_path = str(img_dir / img_name)
                if image:
                    _save_image(image, Path(img_path))

                content_type = str(item.get("content_type", "")).lower()
                modality = ("ct" if "ct" in content_type else
                            "mri" if "mri" in content_type else
                            "xray" if "xray" in content_type else "radiology")

                metadata.append({
                    "id": f"slake_{split}_{idx:05d}",
                    "dataset": "SLAKE",
                    "image_path": img_path if image else None,
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "modality": modality,
                    "body_part": str(item.get("content_type", "mixed")),
                    "split": split,
                })
        except Exception as e:
            print(f"  [ERROR] {split}: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Saved {len(metadata)} records")
    return {"dataset": "SLAKE", "count": len(metadata), "path": str(out_dir)}


def download_pathvqa(max_samples: int = 500) -> dict:
    """Download PathVQA dataset from HuggingFace."""
    from datasets import load_dataset

    out_dir = _ensure_dir(DATA_DIR / "pathvqa")
    img_dir = _ensure_dir(out_dir / "images")
    metadata = []

    print(f"\n[PathVQA] Downloading from HuggingFace...")
    for split in ["train", "test"]:
        try:
            ds = load_dataset("flaviagiammarino/path-vqa", split=split)
            n = min(len(ds), max_samples - len(metadata))
            print(f"  {split}: {len(ds)} samples (taking {n})")

            for idx in range(n):
                item = ds[idx]
                image = item.get("image")
                img_name = f"pathvqa_{split}_{idx:05d}.jpg"
                img_path = str(img_dir / img_name)
                if image:
                    _save_image(image, Path(img_path))

                metadata.append({
                    "id": f"pathvqa_{split}_{idx:05d}",
                    "dataset": "PathVQA",
                    "image_path": img_path if image else None,
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "modality": "pathology",
                    "body_part": "histopathology",
                    "split": split,
                })
        except Exception as e:
            print(f"  [ERROR] {split}: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Saved {len(metadata)} records")
    return {"dataset": "PathVQA", "count": len(metadata), "path": str(out_dir)}


def download_pmc_vqa(max_samples: int = 300) -> dict:
    """Download PMC-VQA dataset from HuggingFace."""
    from datasets import load_dataset

    out_dir = _ensure_dir(DATA_DIR / "pmc_vqa")
    img_dir = _ensure_dir(out_dir / "images")
    metadata = []

    print(f"\n[PMC-VQA] Downloading from HuggingFace...")
    try:
        ds = load_dataset("RadGenome/PMC-VQA", split="test")
        n = min(len(ds), max_samples)
        print(f"  test: {len(ds)} samples (taking {n})")

        for idx in range(n):
            item = ds[idx]
            image = item.get("image")
            img_name = f"pmcvqa_{idx:05d}.jpg"
            img_path = str(img_dir / img_name)
            if image:
                _save_image(image, Path(img_path))

            options = []
            for key in ["Choice A", "Choice B", "Choice C", "Choice D"]:
                opt = str(item.get(key, "")).strip()
                if opt:
                    options.append(opt)

            metadata.append({
                "id": f"pmc_vqa_{idx:05d}",
                "dataset": "PMC-VQA",
                "image_path": img_path if image else None,
                "question": str(item.get("Question", "")).strip(),
                "answer": str(item.get("Answer", "")).strip(),
                "options": options if options else None,
                "modality": "mixed",
                "body_part": "medical_literature",
                "split": "test",
            })
    except Exception as e:
        print(f"  [ERROR]: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Saved {len(metadata)} records")
    return {"dataset": "PMC-VQA", "count": len(metadata), "path": str(out_dir)}


def download_chest_xray(max_samples: int = 500) -> dict:
    """Download Chest X-ray pneumonia classification dataset."""
    from datasets import load_dataset

    out_dir = _ensure_dir(DATA_DIR / "chest_xray")
    img_dir = _ensure_dir(out_dir / "images")
    metadata = []

    label_map = {0: "NORMAL", 1: "PNEUMONIA"}

    print(f"\n[Chest X-ray] Downloading from hf-vision/chest-xray-pneumonia...")
    for split in ["train", "test"]:
        try:
            ds = load_dataset("hf-vision/chest-xray-pneumonia", split=split)
            n = min(len(ds), max_samples // 2)
            print(f"  {split}: {len(ds)} samples (taking {n})")

            for idx in range(n):
                item = ds[idx]
                image = item.get("image")
                label = item.get("label", 0)
                label_name = label_map.get(label, f"class_{label}")

                img_name = f"cxr_{split}_{idx:05d}_{label_name.lower()}.jpg"
                img_path = str(img_dir / img_name)
                if image:
                    _save_image(image, Path(img_path))

                findings = ("Pneumonia pattern detected — consolidation and/or infiltrates"
                            if label_name == "PNEUMONIA"
                            else "No acute cardiopulmonary findings")

                metadata.append({
                    "id": f"chest_xray_{split}_{idx:05d}",
                    "dataset": "Chest-Xray-Pneumonia",
                    "image_path": img_path if image else None,
                    "label": label_name,
                    "modality": "xray",
                    "body_part": "chest",
                    "findings": findings,
                    "split": split,
                })
        except Exception as e:
            print(f"  [ERROR] {split}: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Saved {len(metadata)} records")
    return {"dataset": "Chest-Xray-Pneumonia", "count": len(metadata), "path": str(out_dir)}


def download_brain_tumor(max_samples: int = 500) -> dict:
    """Download Brain Tumor MRI classification dataset."""
    from datasets import load_dataset

    out_dir = _ensure_dir(DATA_DIR / "brain_tumor")
    img_dir = _ensure_dir(out_dir / "images")
    metadata = []

    label_map = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
    findings_map = {
        "glioma": "Glioma tumor — irregular mass with heterogeneous enhancement",
        "meningioma": "Meningioma — well-circumscribed extra-axial mass with homogeneous enhancement",
        "pituitary": "Pituitary adenoma — sellar/suprasellar mass",
        "notumor": "No intracranial mass lesion. Normal brain parenchyma.",
    }

    print(f"\n[Brain Tumor MRI] Downloading from HuggingFace...")
    try:
        ds = load_dataset("AIOmarRehan/Brain_Tumor_MRI_Dataset", split="test")
        n = min(len(ds), max_samples)
        print(f"  test: {len(ds)} samples (taking {n})")

        for idx in range(n):
            item = ds[idx]
            image = item.get("image")
            label = item.get("label", 2)
            label_name = label_map.get(label, f"class_{label}")

            img_name = f"brain_{idx:05d}_{label_name}.jpg"
            img_path = str(img_dir / img_name)
            if image:
                _save_image(image, Path(img_path))

            metadata.append({
                "id": f"brain_tumor_{idx:05d}",
                "dataset": "Brain-Tumor-MRI",
                "image_path": img_path if image else None,
                "label": label_name,
                "modality": "mri",
                "body_part": "brain",
                "findings": findings_map.get(label_name, "No findings"),
                "split": "train",
            })
    except Exception as e:
        print(f"  [ERROR]: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Saved {len(metadata)} records")
    return {"dataset": "Brain-Tumor-MRI", "count": len(metadata), "path": str(out_dir)}


def download_skin_cancer(max_samples: int = 500) -> dict:
    """Download ISIC skin cancer dermoscopy dataset (HAM10000)."""
    from datasets import load_dataset

    out_dir = _ensure_dir(DATA_DIR / "skin_cancer")
    img_dir = _ensure_dir(out_dir / "images")
    metadata = []

    findings_map = {
        "akiec": "Actinic keratosis — rough, scaly patch. Pre-cancerous lesion.",
        "bcc": "Basal cell carcinoma — pearly/waxy bump with rolled borders.",
        "bkl": "Benign keratosis — well-demarcated pigmented lesion.",
        "df": "Dermatofibroma — firm, raised nodule. Benign.",
        "mel": "Melanoma — asymmetric lesion with irregular borders. ABCDE criteria positive.",
        "nv": "Melanocytic nevus — symmetric, well-circumscribed. Benign mole.",
        "vasc": "Vascular lesion — red/purple lesion with vascular pattern.",
    }

    print(f"\n[Skin Cancer] Downloading from marmal88/skin_cancer...")
    try:
        ds = load_dataset("marmal88/skin_cancer", split="train")
        n = min(len(ds), max_samples)
        print(f"  train: {len(ds)} samples (taking {n})")

        for idx in range(n):
            item = ds[idx]
            image = item.get("image")
            dx = str(item.get("dx", "nv")).lower()

            img_name = f"skin_{idx:05d}_{dx}.jpg"
            img_path = str(img_dir / img_name)
            if image:
                _save_image(image, Path(img_path))

            metadata.append({
                "id": f"skin_cancer_{idx:05d}",
                "dataset": "Skin-Cancer-ISIC",
                "image_path": img_path if image else None,
                "label": dx,
                "modality": "dermoscopy",
                "body_part": "skin",
                "findings": findings_map.get(dx, "Skin lesion identified"),
                "age": item.get("age"),
                "sex": item.get("sex"),
                "localization": item.get("localization"),
                "split": "train",
            })
    except Exception as e:
        print(f"  [ERROR]: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Saved {len(metadata)} records")
    return {"dataset": "Skin-Cancer-ISIC", "count": len(metadata), "path": str(out_dir)}


def download_medmnist(max_samples_per_subset: int = 100) -> dict:
    """Download MedMNIST v2 — 12 standardized medical image datasets via pip package."""
    out_dir = _ensure_dir(DATA_DIR / "medmnist")
    metadata = []

    subsets_info = {
        "PathMNIST": {"modality": "pathology", "body_part": "colon",
            "labels": {0: "Adipose", 1: "Background", 2: "Debris", 3: "Lymphocytes",
                       4: "Mucus", 5: "Smooth_muscle", 6: "Normal_colon", 7: "Stroma", 8: "Adenocarcinoma"}},
        "ChestMNIST": {"modality": "xray", "body_part": "chest",
            "labels": {i: f"finding_{i}" for i in range(14)}},
        "DermaMNIST": {"modality": "dermoscopy", "body_part": "skin",
            "labels": {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}},
        "OCTMNIST": {"modality": "oct", "body_part": "eye",
            "labels": {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}},
        "PneumoniaMNIST": {"modality": "xray", "body_part": "chest",
            "labels": {0: "Normal", 1: "Pneumonia"}},
        "RetinaMNIST": {"modality": "fundus", "body_part": "eye",
            "labels": {0: "Normal", 1: "DR1", 2: "DR2", 3: "DR3", 4: "DR4"}},
        "BreastMNIST": {"modality": "ultrasound", "body_part": "breast",
            "labels": {0: "Benign", 1: "Malignant"}},
        "BloodMNIST": {"modality": "microscopy", "body_part": "blood",
            "labels": {0: "Basophil", 1: "Eosinophil", 2: "Erythroblast", 3: "Ig",
                       4: "Lymphocyte", 5: "Monocyte", 6: "Neutrophil", 7: "Platelet"}},
        "TissueMNIST": {"modality": "microscopy", "body_part": "tissue",
            "labels": {i: f"tissue_{i}" for i in range(8)}},
        "OrganAMNIST": {"modality": "ct", "body_part": "abdomen_axial",
            "labels": {i: f"organ_{i}" for i in range(11)}},
        "OrganCMNIST": {"modality": "ct", "body_part": "abdomen_coronal",
            "labels": {i: f"organ_{i}" for i in range(11)}},
        "OrganSMNIST": {"modality": "ct", "body_part": "abdomen_sagittal",
            "labels": {i: f"organ_{i}" for i in range(11)}},
    }

    print(f"\n[MedMNIST v2] Downloading 12 subsets via medmnist package...")

    try:
        import medmnist
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        print("  [ERROR] medmnist not installed. Run: pip install medmnist")
        return {"dataset": "MedMNIST-v2", "count": 0, "path": str(out_dir)}

    cache_dir = str(out_dir / "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for class_name, info in subsets_info.items():
        subset_key = class_name.lower()
        try:
            DatasetClass = getattr(medmnist, class_name)
            dataset = DatasetClass(split="test", download=True, root=cache_dir, size=28)

            n = min(len(dataset), max_samples_per_subset)
            print(f"  {class_name}: {len(dataset)} samples (taking {n})")

            sub_dir = _ensure_dir(out_dir / subset_key / "images")

            for idx in range(n):
                img_pil, label_array = dataset[idx]
                label = int(label_array[0]) if hasattr(label_array, '__len__') else int(label_array)
                label_name = info["labels"].get(label, f"class_{label}")

                img_name = f"{subset_key}_{idx:05d}.png"
                img_path = str(sub_dir / img_name)

                if hasattr(img_pil, "save"):
                    img_pil.save(img_path)
                elif isinstance(img_pil, np.ndarray):
                    PILImage.fromarray(img_pil).save(img_path)

                metadata.append({
                    "id": f"medmnist_{subset_key}_{idx:05d}",
                    "dataset": f"MedMNIST-{class_name}",
                    "image_path": img_path,
                    "label": label_name,
                    "modality": info["modality"],
                    "body_part": info["body_part"],
                    "split": "test",
                    "subset": subset_key,
                })
        except Exception as e:
            print(f"  [ERROR] {class_name}: {e}")

    _save_metadata(metadata, out_dir)
    print(f"  Total MedMNIST: {len(metadata)} records")
    return {"dataset": "MedMNIST-v2", "count": len(metadata), "path": str(out_dir)}


# ══════════════════════════════════════════════════════════════
#  Catalog Builder
# ══════════════════════════════════════════════════════════════

def build_catalog() -> dict:
    """Build a unified catalog of all downloaded medical image datasets.

    This catalog is used by tool simulations to find images by modality/body_part.
    """
    catalog = {
        "created": datetime.now().isoformat(),
        "description": "Unified catalog of medical imaging datasets for BIOAgents RL training",
        "datasets": {},
        "by_modality": defaultdict(list),
        "by_body_part": defaultdict(list),
        "statistics": {},
    }

    total_images = 0
    total_with_qa = 0

    if not DATA_DIR.exists():
        return catalog

    for ds_dir in sorted(DATA_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue

        meta_path = ds_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path, "r") as f:
            records = json.load(f)

        ds_name = ds_dir.name
        catalog["datasets"][ds_name] = {
            "path": str(ds_dir),
            "count": len(records),
            "modalities": list(set(r.get("modality", "unknown") for r in records)),
            "body_parts": list(set(r.get("body_part", "unknown") for r in records)),
            "has_qa": any("question" in r for r in records),
            "has_labels": any("label" in r for r in records),
        }

        for record in records:
            modality = record.get("modality", "unknown")
            body_part = record.get("body_part", "unknown")
            img_path = record.get("image_path")

            if img_path and os.path.exists(img_path):
                total_images += 1
                entry = {
                    "id": record.get("id", ""),
                    "image_path": img_path,
                    "dataset": ds_name,
                    "modality": modality,
                    "body_part": body_part,
                }
                if "findings" in record:
                    entry["findings"] = record["findings"]
                if "label" in record:
                    entry["label"] = record["label"]
                if "question" in record:
                    entry["question"] = record["question"]
                    entry["answer"] = record.get("answer", "")
                    total_with_qa += 1

                catalog["by_modality"][modality].append(entry)
                catalog["by_body_part"][body_part].append(entry)

    catalog["statistics"] = {
        "total_images": total_images,
        "total_with_qa": total_with_qa,
        "datasets_count": len(catalog["datasets"]),
        "modalities": {k: len(v) for k, v in catalog["by_modality"].items()},
        "body_parts": {k: len(v) for k, v in catalog["by_body_part"].items()},
    }

    catalog["by_modality"] = dict(catalog["by_modality"])
    catalog["by_body_part"] = dict(catalog["by_body_part"])

    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Medical Image Catalog Built")
    print(f"{'=' * 60}")
    print(f"  Total images: {total_images}")
    print(f"  With QA pairs: {total_with_qa}")
    print(f"  Datasets: {len(catalog['datasets'])}")
    print(f"  Modalities: {catalog['statistics']['modalities']}")
    print(f"  Body parts: {catalog['statistics']['body_parts']}")
    print(f"  Catalog: {CATALOG_PATH}")

    return catalog


# ══════════════════════════════════════════════════════════════
#  Dispatcher
# ══════════════════════════════════════════════════════════════

DOWNLOAD_FUNCTIONS = {
    "vqa_rad": download_vqa_rad,
    "slake": download_slake,
    "pathvqa": download_pathvqa,
    "pmc_vqa": download_pmc_vqa,
    "chest_xray": download_chest_xray,
    "brain_tumor": download_brain_tumor,
    "skin_cancer": download_skin_cancer,
    "medmnist": download_medmnist,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download Medical Imaging Datasets for BIOAgents RL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--datasets", nargs="+", choices=list(DOWNLOAD_FUNCTIONS.keys()))
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per dataset")
    parser.add_argument("--catalog-only", action="store_true", help="Rebuild catalog only")

    args = parser.parse_args()

    if args.list:
        _print_registry()
        return

    if args.catalog_only:
        build_catalog()
        return

    if not args.all and not args.datasets:
        parser.print_help()
        return

    datasets_to_download = list(DOWNLOAD_FUNCTIONS.keys()) if args.all else args.datasets

    print(f"\n{'#' * 60}")
    print(f"  BIOAgents Medical Image Downloader")
    print(f"  Datasets: {', '.join(datasets_to_download)}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output: {DATA_DIR}")
    print(f"{'#' * 60}")

    results = []
    for ds_name in datasets_to_download:
        fn = DOWNLOAD_FUNCTIONS[ds_name]
        try:
            t0 = time.time()
            if ds_name == "medmnist":
                result = fn(max_samples_per_subset=max(args.max_samples // 12, 50))
            else:
                result = fn(max_samples=args.max_samples)
            elapsed = time.time() - t0
            result["elapsed_sec"] = round(elapsed, 1)
            results.append(result)
            print(f"  [{ds_name}] Done in {elapsed:.1f}s — {result['count']} samples")
        except Exception as e:
            print(f"  [{ds_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"dataset": ds_name, "count": 0, "error": str(e)})

    catalog = build_catalog()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "catalog_stats": catalog.get("statistics", {}),
    }
    summary_path = DATA_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Download Complete!")
    print(f"{'=' * 60}")
    total = sum(r.get("count", 0) for r in results)
    print(f"  Total samples: {total}")
    print(f"  Summary: {summary_path}")
    print(f"  Catalog: {CATALOG_PATH}")


if __name__ == "__main__":
    main()
