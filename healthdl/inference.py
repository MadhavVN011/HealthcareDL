"""Load a neural state dictionary with its schema and apply explicit demo policies."""

import difflib
import json
import re
from pathlib import Path

import numpy as np

from .data import CATALOG_PATH, SCHEMA_VERSION, file_hash, load_catalog, normalize_symptom
from .runtime import NumpyNetwork

EMERGENCY_SYMPTOMS = {
    "chest_pain", "breathlessness", "weakness_of_one_body_side", "slurred_speech",
    "coma", "stomach_bleeding", "acute_liver_failure",
}
EMERGENCY_MESSAGE = (
    "These symptoms can require urgent medical assessment. If they are happening now, "
    "seek emergency care and do not wait for a result from this demo."
)


class InputError(ValueError):
    pass


def parse_symptoms(raw, features: list[str]) -> list[str]:
    if isinstance(raw, str):
        if len(raw) > 3000:
            raise InputError("The symptom list is too long.")
        tokens = re.split(r"[,;\n]", raw.strip().strip("[]"))
    elif isinstance(raw, list) and len(raw) <= len(features):
        tokens = raw
    else:
        raise InputError("Provide symptoms as comma-separated text or a list of names.")
    if any(not isinstance(token, str) or len(token) > 120 for token in tokens):
        raise InputError("Each symptom must be a short text name.")
    names = list(dict.fromkeys(normalize_symptom(token) for token in tokens if token.strip(" '\"")))
    if not names:
        raise InputError("Choose at least one symptom from the available list.")
    unknown = [name for name in names if name not in features]
    if unknown:
        suggestions = difflib.get_close_matches(unknown[0], features, n=3, cutoff=0.55)
        hint = " Suggested names: " + ", ".join(s.replace("_", " ") for s in suggestions) + "." if suggestions else ""
        raise InputError("One or more symptom names are not recognized. Use the available symptom list." + hint)
    return names


class Predictor:
    def __init__(self, artifact_dir: Path):
        directory = Path(artifact_dir)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION or metadata["architecture"] != "mlp_128_64_dropout_015":
            raise ValueError("Unsupported model schema or architecture; retrain the model")
        self.features = metadata["features"]
        self.class_ids = metadata["class_ids"]
        self.catalog = load_catalog()
        if (not self.features or len(set(self.features)) != len(self.features)
                or any(not isinstance(name, str) or normalize_symptom(name) != name for name in self.features)
                or len(set(self.class_ids)) != len(self.class_ids)
                or set(self.class_ids) != set(self.catalog)):
            raise ValueError("Invalid feature or class schema")
        paths = {"inference_sha256": directory / "inference.npz",
                 "reference_sha256": directory / "training_patterns.npz",
                 "catalog_sha256": CATALOG_PATH}
        if any(file_hash(path) != metadata[key] for key, path in paths.items()):
            raise ValueError("Model artifact checksum mismatch; retrain or restore the artifact bundle")
        self.policy = metadata["policy"]
        if not (isinstance(self.policy["min_symptoms"], int)
                and isinstance(self.policy["max_symptoms"], int)
                and 1 <= self.policy["min_symptoms"] <= self.policy["max_symptoms"] <= len(self.features)
                and all(np.isfinite(self.policy[key]) and 0 <= self.policy[key] <= 1
                        for key in ["min_jaccard", "min_score", "min_margin"])):
            raise ValueError("Invalid inference policy")
        with np.load(paths["reference_sha256"], allow_pickle=False) as reference:
            self.patterns = reference["patterns"].astype(np.float32)
        if (self.patterns.ndim != 2 or self.patterns.shape[1] != len(self.features)
                or not len(self.patterns) or not np.isin(self.patterns, [0, 1]).all()
                or not (self.patterns.sum(1) >= 1).all()):
            raise ValueError("Invalid reference patterns")
        self.model = NumpyNetwork(paths["inference_sha256"], len(self.features), len(self.class_ids))

    def predict(self, raw) -> dict:
        symptoms = parse_symptoms(raw, self.features)
        if EMERGENCY_SYMPTOMS.intersection(symptoms):
            return {"status": "urgent", "message": EMERGENCY_MESSAGE, "symptoms": symptoms}
        count = len(symptoms)
        if count < self.policy["min_symptoms"]:
            return {"status": "insufficient", "message": (
                f"You selected {count} unique {'symptom' if count == 1 else 'symptoms'}. "
                f"This demo needs at least {self.policy['min_symptoms']} distinct, recognized symptoms "
                "to compare a pattern. For a walkthrough, run a fictional demo example. "
                "Only include symptoms you actually have when exploring your own list."
            ), "symptoms": symptoms}
        vector = np.array([name in symptoms for name in self.features], dtype=np.float32)
        union = np.maximum(self.patterns, vector).sum(axis=1)
        similarity = float((np.minimum(self.patterns, vector).sum(axis=1) / union).max())
        if count > self.policy["max_symptoms"] or similarity < self.policy["min_jaccard"]:
            return {"status": "unsupported", "message": (
                "This combination is outside the patterns this demo can support. "
                "A healthcare professional can help assess these symptoms."
            ), "symptoms": symptoms}
        scores = self.model.probabilities(vector[np.newaxis, :])[0]
        order = np.argsort(scores)[::-1]
        if scores[order[0]] < self.policy["min_score"] or scores[order[0]] - scores[order[1]] < self.policy["min_margin"]:
            return {"status": "uncertain", "message": (
                "The demo cannot select a clear dataset match for these symptoms. "
                "This result cannot rule out a medical condition."
            ), "symptoms": symptoms}
        condition = self.catalog[self.class_ids[order[0]]]
        if condition["id"] in {"heart_attack", "brain_hemorrhage"}:
            return {"status": "urgent", "message": EMERGENCY_MESSAGE, "symptoms": symptoms}
        return {"status": "match", "symptoms": symptoms, "condition": condition,
                "model_score": float(scores[order[0]]),
                "message": "A match to an example in the learning dataset. This is not a diagnosis."}
