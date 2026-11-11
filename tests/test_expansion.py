import hashlib
import json
from types import SimpleNamespace

import numpy as np

from healthdl.data import ROOT, pattern_id, split_dataset
from healthdl.training import augment_training
from healthdl.chest_data import ChestDataset, split_chest_dataset


def test_augmentation_is_derived_only_from_training_without_original_pattern_overlap(dataset):
    split = split_dataset(dataset)
    extra_x, extra_y, audit = augment_training(dataset, split["train"])
    assert len(extra_x) == audit["synthetic_training_patterns"] > len(split["train"])
    assert extra_x.sum(1).min() >= 3
    original = {pattern_id(row) for row in dataset.x}
    assert not original & set(audit["synthetic_pattern_ids"])
    for row, label in zip(extra_x, extra_y):
        possible = dataset.x[split["train"]][dataset.y[split["train"]] == label]
        assert any(np.all(row <= parent) and int((parent - row).sum()) == 1 for parent in possible)


def test_ambiguous_augmentation_and_existing_holdout_patterns_are_excluded():
    dataset = SimpleNamespace(
        features=list("abcde"),
        x=np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 1], [0, 1, 1, 1, 0]], dtype=np.float32),
        y=np.array([0, 1, 0]),
    )
    extra, _, audit = augment_training(dataset, np.array([0, 1]))
    keys = {pattern_id(row) for row in extra}
    assert pattern_id(np.array([1, 1, 1, 0, 0])) not in keys
    assert pattern_id(dataset.x[2]) not in keys
    assert audit["ambiguous_patterns_discarded"] == 1
    assert audit["original_pattern_candidates_discarded"] == 1


def test_exported_augmentation_preserves_baseline_partitions_and_is_honestly_counted():
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    protocol = json.loads((ROOT / "data/expansion_protocol.json").read_text())
    for name, keys in report["split_pattern_ids"].items():
        digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
        assert digest == protocol["symptoms"]["baseline_partition_membership_sha256"][name]
    augmentation = report["augmentation"]
    assert augmentation["new_patient_records"] == 0
    assert augmentation["training_examples_total"] == report["split_sizes"]["train"] + augmentation["synthetic_training_patterns"]
    assert 0 < augmentation["synthetic_weight_fraction"] <= 0.5
    assert not set(augmentation["synthetic_pattern_ids"]) & {key for keys in report["split_pattern_ids"].values() for key in keys}
    rejected = protocol["rejected_candidates"][0]
    assert rejected["new_binary_patterns"] == 0
    assert rejected["shared_with_original"] == rejected["unique_binary_patterns"] == report["unique_rows"]


def test_group_split_keeps_every_patient_folder_in_one_partition():
    groups = [f"{category}-{patient}" for category in ["Covid", "Healthy", "Others"] for patient in range(10) for _ in range(3)]
    categories = [group.split("-")[0] for group in groups]
    dataset = ChestDataset([], np.array([int(c == "Covid") for c in categories]), [], groups, categories, len(groups))
    split = split_chest_dataset(dataset)
    memberships = [{groups[i] for i in indexes} for indexes in split.values()]
    assert all(not left & right for i, left in enumerate(memberships) for right in memberships[i + 1:])
    assert sum(map(len, split.values())) == len(groups)
    assert [len(values) for values in memberships] == [18, 6, 6]
    assert all(set(categories[i] for i in indexes) == {"Covid", "Healthy", "Others"} for indexes in split.values())


def test_exported_ct_has_disjoint_patient_folders_and_more_unique_images():
    report = json.loads((ROOT / "artifacts/chest_ct/report.json").read_text())
    protocol = json.loads((ROOT / "data/expansion_protocol.json").read_text())
    groups = [set(values) for values in report["split_group_ids"].values()]
    assert all(not left & right for i, left in enumerate(groups) for right in groups[i + 1:])
    assert sum(map(len, groups)) == report["dataset"]["patient_folders"] == 210
    assert report["dataset"]["unique_images"] > protocol["chest_ct"]["old_unique_images"]
    assert report["dataset"]["raw_images"] - report["dataset"]["duplicates_removed"] == report["dataset"]["unique_images"]
    for name, summary in report["partition_summary"].items():
        assert summary["patient_folders"] == len(report["split_group_ids"][name])
        assert summary["images"] == report["split_sizes"][name]
        assert sum(summary["categories"].values()) == summary["images"]
