"""Reproducible CPU training, validation-only checkpoint selection, held-out report."""

import copy
from collections import Counter
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, log_loss, top_k_accuracy_score,
)
from sklearn.svm import SVC
from torch import nn

from .data import ROOT, SCHEMA_VERSION, file_hash, load_dataset, pattern_id, split_dataset
from .model import SymptomNetwork
from .inference import Predictor
from .runtime import NumpyNetwork


@dataclass
class TrainConfig:
    data_path: Path = ROOT / "Training.csv"
    output_dir: Path = ROOT / "artifacts"
    seed: int = 20
    epochs: int = 400
    patience: int = 40
    batch_size: int = 32
    learning_rate: float = 0.003
    augment_missing_symptoms: bool = True


def augment_training(dataset, train_indices):
    """Delete one positive symptom from training rows; never fabricate new patients."""
    original = {pattern_id(row) for row in dataset.x}
    candidates, ambiguous = {}, set()
    excluded_original = 0
    for index in train_indices:
        row, label = dataset.x[index], int(dataset.y[index])
        if row.sum() <= 3:
            continue
        for feature in np.flatnonzero(row):
            masked = row.copy()
            masked[feature] = 0
            key = pattern_id(masked)
            if key in original:
                excluded_original += 1
                continue
            if key in candidates and candidates[key][1] != label:
                ambiguous.add(key)
            else:
                candidates[key] = (masked, label)
    accepted = sorted(set(candidates) - ambiguous)
    extra_x = np.array([candidates[key][0] for key in accepted], dtype=np.float32).reshape(-1, len(dataset.features))
    extra_y = np.array([candidates[key][1] for key in accepted], dtype=np.int64)
    return extra_x, extra_y, {
        "kind": "training_only_single_positive_symptom_deletion", "minimum_remaining_symptoms": 3,
        "original_training_patterns": len(train_indices), "synthetic_training_patterns": len(accepted),
        "ambiguous_patterns_discarded": len(ambiguous), "original_pattern_candidates_discarded": excluded_original,
        "synthetic_pattern_ids": accepted, "new_patient_records": 0,
        "note": "Synthetic variants are not new observations or independent validation. All original patterns are excluded from augmentation.",
    }


def metrics(y, probabilities, class_ids):
    predicted = probabilities.argmax(axis=1)
    labels = np.arange(len(class_ids))
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "top_3_accuracy": float(top_k_accuracy_score(y, probabilities, k=3, labels=labels)),
        "log_loss": float(log_loss(y, probabilities, labels=labels)),
        "multiclass_brier_score": float(np.square(probabilities - np.eye(len(labels))[y]).sum(axis=1).mean()),
        "per_class": classification_report(
            y, predicted, labels=labels, target_names=class_ids,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
    }


def train(config: TrainConfig | None = None) -> dict:
    config = config or TrainConfig()
    if min(config.epochs, config.patience, config.batch_size) < 1:
        raise ValueError("Epochs, patience and batch size must be positive")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = load_dataset(config.data_path)
    split = split_dataset(dataset, config.seed)
    x = torch.from_numpy(dataset.x)
    y = torch.from_numpy(dataset.y)
    train_indices = torch.from_numpy(split["train"])
    val_indices = torch.from_numpy(split["validation"])
    if config.augment_missing_symptoms:
        extra_x, extra_y, augmentation = augment_training(dataset, split["train"])
    else:
        extra_x = np.empty((0, len(dataset.features)), dtype=np.float32)
        extra_y = np.empty(0, dtype=np.int64)
        augmentation = {"kind": "disabled", "original_training_patterns": len(train_indices),
                        "synthetic_training_patterns": 0, "synthetic_pattern_ids": [], "new_patient_records": 0}
    fit_x = torch.cat([x[train_indices], torch.from_numpy(extra_x)])
    fit_y = torch.cat([y[train_indices], torch.from_numpy(extra_y)])
    weights = torch.ones(len(fit_y))
    # Within each category, synthetic rows collectively carry at most the weight
    # of the original training rows, regardless of symptom-list length.
    for label in range(len(dataset.class_ids)):
        extra_indexes = torch.nonzero(fit_y[len(train_indices):] == label, as_tuple=True)[0] + len(train_indices)
        if len(extra_indexes):
            weights[extra_indexes] = int((y[train_indices] == label).sum()) / len(extra_indexes)
    weights /= weights.mean()
    augmentation["training_examples_total"] = len(fit_y)
    augmentation["synthetic_weight_fraction"] = float(weights[len(train_indices):].sum() / weights.sum())
    model = SymptomNetwork(len(dataset.features), len(dataset.class_ids))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    history, best_state = [], None
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(fit_y))
        for indices in order.split(config.batch_size):
            optimizer.zero_grad()
            logits = model(fit_x[indices])
            loss = ((nn.functional.cross_entropy(logits, fit_y[indices], reduction="none") * weights[indices]).mean()
                    if len(extra_x) else criterion(logits, fit_y[indices]))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_logits = model(fit_x)
            val_logits = model(x[val_indices])
            train_loss = ((nn.functional.cross_entropy(train_logits, fit_y, reduction="none") * weights).mean().item()
                          if len(extra_x) else criterion(train_logits, fit_y).item())
            val_loss = criterion(val_logits, y[val_indices]).item()
            val_accuracy = (val_logits.argmax(1) == y[val_indices]).float().mean().item()
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError("Training diverged; refusing to export invalid weights")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": val_loss, "validation_accuracy": val_accuracy})
        if val_loss < best_loss - 0.0001:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_prob = model(x[val_indices]).softmax(1).numpy()
        test_prob = model(x[split["test"]]).softmax(1).numpy()
        # Synthetic missing-symptom stress check, never used to select the model.
        masked = dataset.x[split["test"]].copy()
        rng = np.random.default_rng(config.seed)
        for row in masked:
            row[rng.choice(np.flatnonzero(row))] = 0
        masked_prob = model(torch.from_numpy(masked)).softmax(1).numpy()
    # A comparison only: the application serves the neural network.
    baseline = SVC(kernel="linear").fit(dataset.x[split["train"]], dataset.y[split["train"]])
    baseline_pred = baseline.predict(dataset.x[split["test"]])
    hashes = {name: [pattern_id(dataset.x[i]) for i in indices] for name, indices in split.items()}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(hashes[left]) & set(hashes[right]):
            raise ValueError("Symptom patterns overlap between partitions")
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": dataset.sha256,
        "raw_rows": dataset.raw_rows, "unique_rows": len(dataset.x),
        "duplicate_rows_removed": dataset.raw_rows - len(dataset.x),
        "feature_count": len(dataset.features), "class_count": len(dataset.class_ids),
        "merged_columns": dataset.merged_columns,
        "split_sizes": {name: len(indices) for name, indices in split.items()},
        "pattern_overlap": 0, "split_pattern_ids": hashes,
        "augmentation": augmentation,
        "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "validation": metrics(dataset.y[split["validation"]], val_prob, dataset.class_ids),
        "test": metrics(dataset.y[split["test"]], test_prob, dataset.class_ids),
        "missing_one_symptom_test": metrics(dataset.y[split["test"]], masked_prob, dataset.class_ids),
        "svc_baseline_test": {
            "accuracy": float(accuracy_score(dataset.y[split["test"]], baseline_pred)),
            "macro_f1": float(f1_score(dataset.y[split["test"]], baseline_pred, average="macro")),
        },
        "limitations": [
            "Public file matches and publisher license declarations are recorded in data/SOURCES.md; original clinical collection and patient-level provenance remain unverified.",
            "Pattern-disjoint partitions are not independent clinical or patient-level validation.",
            "Related symptom templates may remain across partitions.",
            "Softmax scores are uncalibrated; no clinical accuracy or safety is established.",
            "Missing-symptom testing is synthetic, not a clinical evaluation.",
            "Training augmentation deletes one observed symptom from original training patterns; it adds no patient records or clinical evidence.",
        ],
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "model.pt"
    reference_path = output / "training_patterns.npz"
    torch.save(model.state_dict(), weights_path)
    inference_path = output / "inference.npz"
    parameters = {}
    for layer, index in enumerate([0, 3, 6]):
        parameters[f"w{layer}"] = model.state_dict()[f"layers.{index}.weight"].numpy()
        parameters[f"b{layer}"] = model.state_dict()[f"layers.{index}.bias"].numpy()
    np.savez_compressed(inference_path, **parameters)
    exported = NumpyNetwork(inference_path, len(dataset.features), len(dataset.class_ids))
    with torch.inference_mode():
        original_logits = model(x).numpy()
    exported_logits = exported.forward(dataset.x)
    np.testing.assert_allclose(exported_logits, original_logits, atol=1e-5, rtol=1e-5)
    report["inference_export_max_abs_error"] = float(np.abs(exported_logits - original_logits).max())
    np.savez_compressed(reference_path, patterns=dataset.x[split["train"]])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "mlp_128_64_dropout_015", "features": dataset.features,
        "class_ids": dataset.class_ids, "dataset_sha256": dataset.sha256,
        "weights_sha256": file_hash(weights_path), "reference_sha256": file_hash(reference_path),
        "inference_sha256": file_hash(inference_path),
        "catalog_sha256": file_hash(ROOT / "data" / "conditions.json"),
        "policy": {"min_symptoms": int(dataset.x[split["train"]].sum(1).min()),
                   "max_symptoms": int(dataset.x[split["train"]].sum(1).max()),
                   "min_jaccard": 0.5, "min_score": 0.7, "min_margin": 0.2},
        "policy_note": "Demonstration heuristics, not validated clinical thresholds or calibrated probabilities.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Report the actual serving policy separately from the raw classifier metrics.
    predictor = Predictor(output)
    results = [predictor.predict([name for name, present in zip(dataset.features, row) if present])
               for row in dataset.x[split["test"]]]
    accepted = [i for i, result in enumerate(results) if result["status"] == "match"]
    correct = sum(results[i]["condition"]["id"] == dataset.class_ids[dataset.y[split["test"]][i]]
                  for i in accepted)
    report["serving_policy_test"] = {
        "status_counts": dict(Counter(result["status"] for result in results)),
        "match_coverage": len(accepted) / len(results),
        "accuracy_among_matches": correct / len(accepted) if accepted else None,
        "note": "Urgent messages and abstentions are not disease predictions; these heuristics are not clinically validated.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
