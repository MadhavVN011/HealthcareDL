import csv

import pytest

from healthdl.data import ROOT
from healthdl.evaluation import evaluate_external


def write_external(tmp_path, dataset, *, missing=False, nonbinary=False):
    # Generated fixture exercises schema/metrics only; it is not clinical evidence.
    x = dataset.x[:3].copy()
    if nonbinary:
        x[0, 0] = 2
    path = tmp_path / "external.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        features = dataset.features[:-1] if missing else dataset.features
        writer.writerow([*features, "prognosis", "patient_id"])
        for index, row in enumerate(x):
            values = row[:-1] if missing else row
            writer.writerow([*values, dataset.class_ids[dataset.y[index]], f"private-patient-{index}"])
    return path


def test_original_data_is_rejected_as_external():
    result = evaluate_external(ROOT / "Training.csv")
    assert result["status"] == "rejected_overlap"
    assert result["overlapping_rows"] == 4920
    assert "classifier" not in result


def test_synthetic_training_variants_are_also_rejected_as_external(tmp_path, dataset):
    from healthdl.data import pattern_id, split_dataset
    from healthdl.training import augment_training
    extra, labels, _ = augment_training(dataset, split_dataset(dataset)["train"])
    assert pattern_id(extra[0]) not in {pattern_id(row) for row in dataset.x}
    path = tmp_path / "synthetic.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*dataset.features, "prognosis"])
        writer.writerow([*extra[0], dataset.class_ids[labels[0]]])
    result = evaluate_external(path)
    assert result["status"] == "rejected_overlap"
    assert result["overlapping_rows"] == 1
    assert "classifier" not in result


def test_external_schema_and_binary_validation(tmp_path, dataset):
    with pytest.raises(ValueError, match="schema"):
        evaluate_external(write_external(tmp_path, dataset, missing=True), group_column="patient_id")
    with pytest.raises(ValueError, match="binary"):
        evaluate_external(write_external(tmp_path, dataset, nonbinary=True), group_column="patient_id")


def test_patient_identifiers_are_not_exported(tmp_path, dataset):
    path = write_external(tmp_path, dataset)
    # Explicitly resolve canonical IDs to labels as expected in a source CSV.
    from healthdl.data import load_catalog
    catalog = load_catalog()
    text = path.read_text()
    for key, item in catalog.items():
        text = text.replace("," + key + ",", "," + item["label"] + ",")
    path.write_text(text)
    result = evaluate_external(path, group_column="patient_id")
    assert result["patient_groups_supplied"] == 3
    assert result["status"] == "rejected_overlap"
    assert "private-patient" not in str(result)


def test_nonoverlapping_examples_produce_research_metrics(tmp_path, dataset):
    from healthdl.data import pattern_id
    import numpy as np
    import warnings
    path = tmp_path / "new-fixture.csv"
    vector = np.zeros(len(dataset.features), dtype=int)
    vector[[0, 2, 5, 10, 15]] = 1
    assert pattern_id(vector) not in {pattern_id(row) for row in dataset.x}
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*dataset.features, "prognosis", "patient_id"])
        writer.writerow([*vector, "Acne", "private-example"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        result = evaluate_external(path, group_column="patient_id")
    assert result["status"] == "evaluated"
    assert result["overlapping_rows"] == 0
    assert result["classifier"]["per_class"]["acne"]["support"] == 1
    assert result["patient_groups_supplied"] == 1
    assert "private-example" not in str(result)
