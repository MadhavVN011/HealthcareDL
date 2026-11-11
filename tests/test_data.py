import csv
import json

import numpy as np
import pytest

from healthdl.data import ROOT, load_catalog, load_dataset, normalize_disease, normalize_symptom, pattern_id, split_dataset


def test_deduplication_and_duplicate_column_signal(dataset):
    assert dataset.raw_rows == 4920
    assert dataset.x.shape == (304, 131)
    assert len(dataset.class_ids) == 41
    assert dataset.merged_columns == {"fluid_overload": [45, 117]}
    with (ROOT / "Training.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        rows = list(reader)
    assert sum(int(row[45]) for row in rows) == 0
    assert sum(int(row[117]) for row in rows) == 114
    assert dataset.x[:, dataset.features.index("fluid_overload")].sum() > 0
    assert len({pattern_id(row) for row in dataset.x}) == len(dataset.x)


def test_partitions_are_disjoint_complete_and_stratified(dataset):
    partitions = split_dataset(dataset)
    sets = {name: {pattern_id(dataset.x[i]) for i in indices} for name, indices in partitions.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert sum(map(len, sets.values())) == len(dataset.x)
    for indices in partitions.values():
        assert set(dataset.y[indices]) == set(range(41))
    assert all(np.array_equal(indices, split_dataset(dataset)[name]) for name, indices in partitions.items())


def test_catalog_resolves_all_original_labels(dataset):
    catalog = load_catalog()
    aliases = {normalize_disease(name): key for key, entry in catalog.items()
               for name in [entry["label"], *entry["aliases"]]}
    for old in ["Diabetes ", "Hypertension ", "Peptic ulcer diseae", "(vertigo) Paroymsal  Positional Vertigo"]:
        entry = catalog[aliases[normalize_disease(old)]]
        assert entry["description"]
        assert entry["source"].startswith("https://medlineplus.gov/")
    assert set(dataset.class_ids) == set(catalog)


@pytest.mark.parametrize("raw,expected", [
    (" High Fever ", "high_fever"), ("dischromic _patches", "dischromic_patches"),
    ("foul_smell_of urine", "foul_smell_of_urine"), ("fluid_overload.1", "fluid_overload"),
])
def test_feature_normalization(raw, expected):
    assert normalize_symptom(raw) == expected


@pytest.mark.parametrize("problem", ["conflict", "nonbinary", "unknown_label"])
def test_invalid_training_data_is_rejected(tmp_path, problem):
    with (ROOT / "Training.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    if problem == "conflict":
        rows.append([*rows[1][:-1], "Allergy"])
    elif problem == "nonbinary":
        rows[1][0] = "NaN"
    else:
        rows[1][-1] = "Unrecognized condition"
    path = tmp_path / "training.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_exported_reference_uses_only_training_partition(dataset):
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    with np.load(ROOT / "artifacts/training_patterns.npz", allow_pickle=False) as reference:
        patterns = {pattern_id(row) for row in reference["patterns"]}
    assert patterns == set(report["split_pattern_ids"]["train"])
    assert not patterns & set(report["split_pattern_ids"]["test"])
    assert not patterns & set(report["split_pattern_ids"]["validation"])
