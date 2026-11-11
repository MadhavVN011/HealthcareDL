"""One feature/label schema shared by training and inference."""

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "conditions.json"
SCHEMA_VERSION = 1


def normalize_symptom(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    aliases = {
        "fluid_overload_1": "fluid_overload",
        "spotting_urination": "spotting_urination",
        "foul_smell_ofurine": "foul_smell_of_urine",
        "cold_hands_and_feet": "cold_hands_and_feets",
        "swollen_extremities": "swollen_extremeties",
        "scarring": "scurring",
    }
    return aliases.get(value, value)


def normalize_disease(value: str) -> str:
    return " ".join(value.lower().split())


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    catalog = {}
    names = set()
    for entry in entries:
        if entry["id"] in catalog or not entry["description"]:
            raise ValueError("Duplicate or incomplete condition entry")
        for name in [entry["label"], *entry["aliases"]]:
            key = normalize_disease(name)
            if key in names:
                raise ValueError(f"Duplicate condition alias: {name}")
            names.add(key)
        if not entry["source"].startswith("https://medlineplus.gov/"):
            raise ValueError("Condition references must use MedlinePlus")
        catalog[entry["id"]] = entry
    return catalog


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pattern_id(row: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(row, dtype=np.uint8).tobytes()).hexdigest()


@dataclass
class Dataset:
    features: list[str]
    class_ids: list[str]
    x: np.ndarray
    y: np.ndarray
    raw_rows: int
    sha256: str
    merged_columns: dict[str, list[int]]


def load_dataset(path: Path = ROOT / "Training.csv") -> Dataset:
    catalog = load_catalog()
    aliases = {
        normalize_disease(name): key
        for key, item in catalog.items()
        for name in [item["label"], *item["aliases"]]
    }
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header[-1] != "prognosis":
            raise ValueError("Expected prognosis as the last column")
        rows = list(reader)
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("Empty or malformed training dataset")
    groups = {}
    for index, name in enumerate(header[:-1]):
        key = normalize_symptom(name)
        if not key:
            raise ValueError("Empty symptom column")
        groups.setdefault(key, []).append(index)
    raw = np.array([row[:-1] for row in rows], dtype=np.float32)
    if not np.isin(raw, [0, 1]).all():
        raise ValueError("Symptom features must be binary and finite")
    # OR duplicate columns; pandas' automatic .1 suffix would hide the real signal.
    features = list(groups)
    x = np.stack([raw[:, indices].max(axis=1) for indices in groups.values()], axis=1)
    unknown_labels = sorted({row[-1] for row in rows if normalize_disease(row[-1]) not in aliases})
    if unknown_labels:
        raise ValueError(f"Unknown condition labels: {unknown_labels}")
    labels = [aliases[normalize_disease(row[-1])] for row in rows]
    class_ids = sorted(set(labels))
    if set(class_ids) != set(catalog):
        raise ValueError("Training labels and condition catalog do not agree")
    label_indices = {key: index for index, key in enumerate(class_ids)}
    seen = {}
    unique_x, unique_y = [], []
    for vector, label in zip(x, labels):
        key = pattern_id(vector)
        if key in seen:
            if seen[key] != label:
                raise ValueError("The same symptom pattern has conflicting labels")
            continue
        seen[key] = label
        unique_x.append(vector)
        unique_y.append(label_indices[label])
    return Dataset(
        features, class_ids, np.array(unique_x, dtype=np.float32),
        np.array(unique_y, dtype=np.int64), len(rows), file_hash(path),
        {name: indices for name, indices in groups.items() if len(indices) > 1},
    )


def split_dataset(dataset: Dataset, seed: int = 20) -> dict[str, np.ndarray]:
    """Approximately 60/20/20 per class, after deduplication; no shared patterns."""
    rng = np.random.default_rng(seed)
    partitions = {name: [] for name in ["train", "validation", "test"]}
    for class_index in range(len(dataset.class_ids)):
        indices = rng.permutation(np.flatnonzero(dataset.y == class_index))
        if len(indices) < 5:
            raise ValueError("Need at least five unique patterns per class")
        holdout = max(1, round(len(indices) * 0.2))
        partitions["test"].extend(indices[:holdout])
        partitions["validation"].extend(indices[holdout:holdout * 2])
        partitions["train"].extend(indices[holdout * 2:])
    return {name: rng.permutation(indices) for name, indices in partitions.items()}
