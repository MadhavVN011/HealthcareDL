"""Evaluate supplied external examples without silently accepting contamination."""
import csv
import json
from pathlib import Path

import numpy as np

from .data import ROOT, file_hash, load_catalog, normalize_disease, normalize_symptom, pattern_id
from .inference import Predictor
from .training import metrics


def evaluate_external(data_path: Path, artifact_dir: Path = ROOT / "artifacts", group_column: str | None = None):
    predictor = Predictor(artifact_dir)
    report = json.loads((Path(artifact_dir) / "report.json").read_text())
    catalog = load_catalog()
    aliases = {normalize_disease(name): key for key, item in catalog.items()
               for name in [key, item["label"], *item["aliases"]]}
    with Path(data_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = list(reader)
    if not header or not rows or any(len(row) != len(header) for row in rows) or header.count("prognosis") != 1:
        raise ValueError("External data requires complete rows and one prognosis column")
    if group_column and header.count(group_column) != 1:
        raise ValueError("The requested patient/group column must occur exactly once")
    if group_column == "prognosis":
        raise ValueError("The label cannot also be a patient/group identifier")
    groups = {}
    for index, column in enumerate(header):
        if column == "prognosis" or column == group_column:
            continue
        if not column and all(not row[index] for row in rows):
            continue  # Publisher's unused trailing column.
        groups.setdefault(normalize_symptom(column), []).append(index)
    if set(groups) != set(predictor.features):
        raise ValueError("External feature schema differs from the model; missing or unknown columns cannot be silently mapped")
    x = np.stack([np.array([[row[i] for i in groups[name]] for row in rows], dtype=np.float32).max(axis=1)
                  for name in predictor.features], axis=1)
    # Validate original values too: merging must not hide a negative or nonbinary column.
    raw = np.array([[row[i] for positions in groups.values() for i in positions] for row in rows], dtype=np.float32)
    if not np.isin(raw, [0, 1]).all():
        raise ValueError("External symptom features must be binary and finite")
    try:
        y = np.array([predictor.class_ids.index(aliases[normalize_disease(row[header.index("prognosis")])]) for row in rows])
    except (KeyError, ValueError) as error:
        raise ValueError("External data contains unrecognized disease labels") from error
    patient_count = None
    if group_column:
        ids = [row[header.index(group_column)].strip() for row in rows]
        if any(not value for value in ids):
            raise ValueError("Patient/group identifiers must not be empty")
        patient_count = len(set(ids))
    known = {value for values in report["split_pattern_ids"].values() for value in values}
    overlap = sum(pattern_id(row) in known for row in x)
    result = {
        "source_sha256": file_hash(data_path), "model_dataset_sha256": report["dataset_sha256"],
        "rows": len(rows), "patient_groups_supplied": patient_count,
        "unique_patterns": len({pattern_id(row) for row in x}), "overlapping_rows": overlap,
        "scope": "External pattern evaluation; clinical provenance and patient independence are not established by this command.",
    }
    if overlap:
        result.update(status="rejected_overlap", message="Data overlaps the original development patterns. Independent metrics were not calculated.")
        return result
    probabilities = predictor.model.probabilities(x)
    result.update(status="evaluated", classifier=metrics(y, probabilities, predictor.class_ids))
    statuses = {}
    for row in x:
        status = (predictor.predict([name for name, present in zip(predictor.features, row) if present])["status"]
                  if row.any() else "invalid_empty")
        statuses[status] = statuses.get(status, 0) + 1
    result["serving_status_counts"] = statuses
    result["category_support"] = {key: int((y == i).sum()) for i, key in enumerate(predictor.class_ids)}
    return result
