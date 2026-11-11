import json

import torch

from healthdl.data import file_hash
from healthdl.inference import Predictor
from healthdl.training import TrainConfig, train


def test_training_is_reproducible_and_exports_reloadable_weights(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = train(TrainConfig(output_dir=first, epochs=5, patience=3))
    repeat = train(TrainConfig(output_dir=second, epochs=5, patience=3))
    assert report["history"] == repeat["history"]
    state = torch.load(first / "model.pt", weights_only=True)
    other = torch.load(second / "model.pt", weights_only=True)
    assert all(torch.equal(state[key], other[key]) for key in state)
    assert report["pattern_overlap"] == 0
    assert report["best_epoch"] <= report["epochs_run"]
    assert report["test"]["per_class"]["fungal_infection"]["support"] >= 1
    assert "svc_baseline_test" in report
    assert "serving_policy_test" in report
    assert "missing_one_symptom_test" in report
    metadata = json.loads((first / "metadata.json").read_text())
    assert metadata["weights_sha256"] == file_hash(first / "model.pt")
    predictor = Predictor(first)
    assert (predictor.model.weights["w0"] == state["layers.0.weight"].numpy()).all()
    assert report["inference_export_max_abs_error"] < 1e-5
