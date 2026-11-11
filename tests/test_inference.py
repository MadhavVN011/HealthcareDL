import json
import shutil

import pytest
import torch

from healthdl.data import ROOT
from healthdl.inference import InputError, Predictor, parse_symptoms
from healthdl.model import SymptomNetwork
from healthdl.runtime import NumpyNetwork
from healthdl.data import load_dataset
import numpy as np


@pytest.mark.parametrize("raw", [None, "", " ", [], {}, [None], [1], [["itching"]], "aching", "itching,unknown"])
def test_invalid_inputs_are_rejected(predictor, raw):
    with pytest.raises(InputError):
        predictor.predict(raw)


def test_normalization_deduplication_and_trailing_comma(predictor):
    assert parse_symptoms("[ 'Itching', 'HIGH FEVER', 'itching', ]", predictor.features) == ["itching", "high_fever"]
    result = predictor.predict("itching, Itching, ITCHING")
    assert result["status"] == "insufficient"
    assert "1 unique symptom" in result["message"]
    assert "at least 3" in result["message"]
    assert "model_score" not in result


def test_sparse_and_urgent_inputs_bypass_network(predictor, monkeypatch):
    def fail(*args):
        raise AssertionError("Network should not run for this input")
    monkeypatch.setattr(predictor.model, "forward", fail)
    assert predictor.predict("cough")["status"] == "insufficient"
    assert predictor.predict("chest pain")["status"] == "urgent"
    assert predictor.predict("slurred speech")["status"] == "urgent"
    assert predictor.predict("acidity,cold hands and feet,blackheads")["status"] == "unsupported"


def test_known_pattern_uses_neural_network_and_canonical_content(predictor):
    result = predictor.predict("itching,skin rash,nodal skin eruptions")
    assert result["status"] == "match"
    assert result["condition"]["id"] == "fungal_infection"
    assert "medications" not in result
    assert "probability" not in result
    assert isinstance(predictor.model, NumpyNetwork)
    assert predictor.model.weights["w0"].shape == (128, 131)
    vector = np.array([[name in result["symptoms"] for name in predictor.features]], dtype=np.float32)
    assert result["model_score"] == float(predictor.model.probabilities(vector).max())


def test_export_matches_pytorch_on_every_unique_pattern(predictor):
    dataset = load_dataset()
    original = SymptomNetwork(131, 41)
    original.load_state_dict(torch.load(ROOT / "artifacts/model.pt", weights_only=True))
    original.eval()
    with torch.inference_mode():
        expected = original(torch.from_numpy(dataset.x)).numpy()
    np.testing.assert_allclose(predictor.model.forward(dataset.x), expected, rtol=1e-5, atol=1e-5)


def test_ambiguous_model_abstains(predictor):
    for parameter in predictor.model.weights.values():
        parameter.fill(0)
    assert predictor.predict("itching,skin rash,nodal skin eruptions")["status"] == "uncertain"


@pytest.mark.parametrize("problem", ["weights", "schema", "classes", "policy"])
def test_artifact_corruption_or_schema_drift_is_rejected(tmp_path, problem):
    target = tmp_path / "artifacts"
    shutil.copytree(ROOT / "artifacts", target)
    if problem == "weights":
        with (target / "inference.npz").open("ab") as stream:
            stream.write(b"changed")
    else:
        path = target / "metadata.json"
        metadata = json.loads(path.read_text())
        if problem == "schema":
            metadata["schema_version"] = 999
        elif problem == "classes":
            metadata["class_ids"][0] = "unknown"
        else:
            metadata["policy"]["min_score"] = float("nan")
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        Predictor(target)
