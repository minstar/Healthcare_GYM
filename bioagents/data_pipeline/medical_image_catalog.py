"""Medical Image Catalog — Unified access to downloaded medical imaging datasets.

This module provides easy access to the medical images downloaded by
`scripts/download_medical_images.py`. During RL training, when agents call
tools like `analyze_medical_image`, this catalog can provide real image paths
and associated metadata for realistic tool simulation.

Usage:
    from bioagents.data_pipeline.medical_image_catalog import MedicalImageCatalog

    catalog = MedicalImageCatalog()

    # Get images by modality (for visual_diagnosis / radiology_report domains)
    xray_images = catalog.get_by_modality("xray", limit=10)
    mri_images = catalog.get_by_modality("mri", limit=5)

    # Get images by body part
    chest_images = catalog.get_by_body_part("chest", limit=10)
    brain_images = catalog.get_by_body_part("brain", limit=5)

    # Get a random image for tool simulation
    img = catalog.random_image(modality="dermoscopy")

    # Get VQA pairs for evaluation
    vqa_pairs = catalog.get_vqa_pairs(dataset="VQA-RAD", limit=50)

    # Print stats
    catalog.print_stats()
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

# ── Project root paths ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CATALOG_PATH = _PROJECT_ROOT / "datasets" / "medical_images" / "catalog.json"
_DATA_DIR = _PROJECT_ROOT / "datasets" / "medical_images"


class MedicalImageCatalog:
    """Unified access to downloaded medical imaging datasets.

    Loads the catalog.json built by download_medical_images.py and provides
    query methods for RL training tool simulations.
    """

    def __init__(self, catalog_path: Optional[str] = None):
        """Initialize the catalog.

        Args:
            catalog_path: Path to catalog.json. If None, uses default location.
        """
        self.catalog_path = Path(catalog_path) if catalog_path else _CATALOG_PATH
        self._catalog = None
        self._all_images = []
        self._by_modality = {}
        self._by_body_part = {}
        self._by_dataset = {}
        self._loaded = False

    def _load(self):
        """Lazy load the catalog."""
        if self._loaded:
            return

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Medical image catalog not found at {self.catalog_path}. "
                "Run: python scripts/download_medical_images.py --all"
            )

        with open(self.catalog_path, "r") as f:
            self._catalog = json.load(f)

        # Index by modality
        for modality, images in self._catalog.get("by_modality", {}).items():
            self._by_modality[modality] = images
            self._all_images.extend(images)

        # Index by body part
        for body_part, images in self._catalog.get("by_body_part", {}).items():
            self._by_body_part[body_part] = images

        # Index by dataset
        for img in self._all_images:
            ds = img.get("dataset", "unknown")
            if ds not in self._by_dataset:
                self._by_dataset[ds] = []
            self._by_dataset[ds].append(img)

        self._loaded = True

    @property
    def stats(self) -> dict:
        """Get catalog statistics."""
        self._load()
        return self._catalog.get("statistics", {})

    @property
    def modalities(self) -> list:
        """Get list of available modalities."""
        self._load()
        return list(self._by_modality.keys())

    @property
    def body_parts(self) -> list:
        """Get list of available body parts."""
        self._load()
        return list(self._by_body_part.keys())

    @property
    def datasets(self) -> list:
        """Get list of available dataset names."""
        self._load()
        return list(self._by_dataset.keys())

    def get_by_modality(
        self,
        modality: str,
        limit: Optional[int] = None,
        shuffle: bool = False,
    ) -> list[dict]:
        """Get images by imaging modality.

        Args:
            modality: One of xray, ct, mri, pathology, dermoscopy, oct, fundus,
                      ultrasound, microscopy, radiology, mixed.
            limit: Max number of images to return.
            shuffle: Whether to shuffle results.

        Returns:
            List of image metadata dicts.
        """
        self._load()
        images = self._by_modality.get(modality, [])
        if shuffle:
            images = random.sample(images, min(len(images), limit or len(images)))
        if limit:
            images = images[:limit]
        return images

    def get_by_body_part(
        self,
        body_part: str,
        limit: Optional[int] = None,
        shuffle: bool = False,
    ) -> list[dict]:
        """Get images by body part.

        Args:
            body_part: E.g., chest, brain, skin, eye, breast, etc.
            limit: Max number of images to return.
            shuffle: Whether to shuffle results.

        Returns:
            List of image metadata dicts.
        """
        self._load()
        images = self._by_body_part.get(body_part, [])
        if shuffle:
            images = random.sample(images, min(len(images), limit or len(images)))
        if limit:
            images = images[:limit]
        return images

    def get_by_dataset(
        self,
        dataset: str,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Get images from a specific dataset."""
        self._load()
        images = self._by_dataset.get(dataset, [])
        if limit:
            images = images[:limit]
        return images

    def random_image(
        self,
        modality: Optional[str] = None,
        body_part: Optional[str] = None,
    ) -> Optional[dict]:
        """Get a random image, optionally filtered by modality/body_part.

        Args:
            modality: Optional modality filter.
            body_part: Optional body part filter.

        Returns:
            Image metadata dict, or None if no matching images.
        """
        self._load()

        pool = self._all_images
        if modality:
            pool = self._by_modality.get(modality, [])
        if body_part:
            pool = [img for img in pool if img.get("body_part") == body_part]

        if not pool:
            return None

        return random.choice(pool)

    def get_vqa_pairs(
        self,
        dataset: Optional[str] = None,
        modality: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Get image-question-answer pairs for VQA evaluation.

        Args:
            dataset: Optional dataset filter (e.g., "vqa_rad", "slake").
            modality: Optional modality filter.
            limit: Max number of pairs.

        Returns:
            List of VQA pair dicts with image_path, question, answer.
        """
        self._load()

        pool = self._all_images
        if dataset:
            pool = self._by_dataset.get(dataset, [])
        if modality:
            pool = [img for img in pool if img.get("modality") == modality]

        # Filter to only those with QA pairs
        pairs = [img for img in pool if "question" in img and img.get("question")]

        if limit:
            pairs = pairs[:limit]

        return pairs

    def get_tool_simulation_data(
        self,
        domain: str,
        num_images: int = 10,
    ) -> list[dict]:
        """Get images suitable for a specific BIOAgents domain.

        Maps domains to appropriate modalities and returns enriched metadata
        suitable for tool simulation (analyze_medical_image, get_image_report, etc).

        Args:
            domain: BIOAgents domain name.
            num_images: Number of images to return.

        Returns:
            List of enriched image metadata for tool simulation.
        """
        self._load()

        domain_modality_map = {
            "visual_diagnosis": ["xray", "ct", "mri", "pathology", "dermoscopy",
                                 "fundus", "ultrasound", "radiology"],
            "radiology_report": ["xray", "ct", "mri", "radiology"],
            "clinical_diagnosis": ["xray", "ct", "mri", "radiology"],
            "triage_emergency": ["xray", "ct"],
            "ehr_management": ["xray", "ct", "mri"],
        }

        modalities = domain_modality_map.get(domain, list(self._by_modality.keys()))

        pool = []
        for mod in modalities:
            pool.extend(self._by_modality.get(mod, []))

        if not pool:
            return []

        selected = random.sample(pool, min(num_images, len(pool)))

        # Enrich with tool-compatible fields
        enriched = []
        for img in selected:
            entry = {
                "image_id": img["id"],
                "image_path": img["image_path"],
                "modality": img["modality"],
                "body_part": img["body_part"],
                "dataset_source": img["dataset"],
            }
            if "findings" in img:
                entry["findings_text"] = img["findings"]
            if "label" in img:
                entry["classification"] = img["label"]
            if "question" in img:
                entry["vqa_question"] = img["question"]
                entry["vqa_answer"] = img.get("answer", "")

            enriched.append(entry)

        return enriched

    def print_stats(self):
        """Print catalog statistics."""
        self._load()
        stats = self.stats
        print(f"\n{'=' * 50}")
        print(f"  Medical Image Catalog Statistics")
        print(f"{'=' * 50}")
        print(f"  Total images: {stats.get('total_images', 0)}")
        print(f"  With QA pairs: {stats.get('total_with_qa', 0)}")
        print(f"  Datasets: {stats.get('datasets_count', 0)}")
        print(f"\n  By Modality:")
        for mod, count in sorted(stats.get("modalities", {}).items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"    {mod}: {count}")
        print(f"\n  By Body Part:")
        for bp, count in sorted(stats.get("body_parts", {}).items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"    {bp}: {count}")
        print(f"{'=' * 50}")


# ── Convenience singleton ──────────────────────────────────────
_default_catalog = None


def get_catalog() -> MedicalImageCatalog:
    """Get the default catalog singleton."""
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = MedicalImageCatalog()
    return _default_catalog


if __name__ == "__main__":
    catalog = MedicalImageCatalog()
    catalog.print_stats()

    print(f"\nModalities: {catalog.modalities}")
    print(f"Body Parts: {catalog.body_parts}")
    print(f"Datasets: {catalog.datasets}")

    # Test getting images for visual_diagnosis domain
    vis_images = catalog.get_tool_simulation_data("visual_diagnosis", num_images=5)
    print(f"\nVisual Diagnosis simulation data ({len(vis_images)} images):")
    for img in vis_images:
        print(f"  {img['image_id']}: {img['modality']}/{img['body_part']}")

    # Test getting VQA pairs
    vqa = catalog.get_vqa_pairs(limit=3)
    print(f"\nVQA pairs ({len(vqa)}):")
    for pair in vqa:
        print(f"  Q: {pair.get('question', '')[:60]}...")
        print(f"  A: {pair.get('answer', '')[:60]}...")
