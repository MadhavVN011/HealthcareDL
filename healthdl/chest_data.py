"""Verified CT archive loading and patient-folder partitions for research training."""

from collections import Counter
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from .data import file_hash

ARCHIVE_SHA256 = "4b9bd9674a058c59274bf6d8a46efb8d0cf473b0434eeb5bbe9d5423e7145d87"
SOURCE_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SZDUQX"


@dataclass
class ChestDataset:
    names: list[str]
    labels: np.ndarray
    hashes: list[str]
    groups: list[str]
    categories: list[str]
    raw_images: int


def load_chest_dataset(archive_path):
    if file_hash(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Unexpected CT archive; verify the publisher release before training")
    names, labels, hashes, groups, categories = [], [], [], [], []
    seen, raw_images = {}, 0
    with ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            parts = Path(name).parts
            if len(parts) != 4 or parts[0] != "New_Data_CoV2" or parts[1] not in {"Covid", "Healthy", "Others"}:
                raise ValueError("Unexpected CT image path or category")
            raw_images += 1
            category = parts[1]
            # Folder numbers restart in each category. Namespace before hashing.
            group = hashlib.sha256("/".join(parts[1:3]).encode()).hexdigest()
            label = int(category == "Covid")
            with Image.open(BytesIO(archive.read(name))) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen:
                prior_label, prior_group = seen[digest]
                if prior_label != label:
                    raise ValueError("Conflicting labels on identical CT images")
                if prior_group != group:
                    raise ValueError("Identical CT images cross patient folders; resolve groups before splitting")
                continue
            seen[digest] = (label, group)
            names.append(name)
            labels.append(label)
            hashes.append(digest)
            groups.append(group)
            categories.append(category)
    # The publisher's 4,173 archive entries include two non-image files.
    if raw_images != 4171 or len(set(groups)) != 210:
        raise ValueError("Archive image/patient-folder totals differ from the publisher release")
    return ChestDataset(names, np.array(labels, dtype=np.int64), hashes, groups, categories, raw_images)


def split_chest_dataset(dataset, seed=20):
    """Stratify patient folders by original category, never individual slices."""
    group_categories = {}
    for group, category in zip(dataset.groups, dataset.categories):
        if group in group_categories and group_categories[group] != category:
            raise ValueError("A patient folder has multiple categories")
        group_categories[group] = category
    unique_groups = sorted(group_categories)
    strata = [group_categories[group] for group in unique_groups]
    train, remainder = train_test_split(unique_groups, test_size=0.4, random_state=seed, stratify=strata)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed,
                                      stratify=[group_categories[group] for group in remainder])
    partitions = {"train": train, "validation": validation, "test": test}
    return {name: np.array([i for i, group in enumerate(dataset.groups) if group in set(values)], dtype=np.int64)
            for name, values in partitions.items()}


def partition_summary(dataset, split):
    return {name: {"images": len(indexes), "patient_folders": len({dataset.groups[i] for i in indexes}),
                   "categories": dict(Counter(dataset.categories[i] for i in indexes))}
            for name, indexes in split.items()}
