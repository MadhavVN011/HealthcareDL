"""Release check usable with runtime dependencies only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from healthdl.data import file_hash
from healthdl.inference import Predictor


def main():
    predictor = Predictor(ROOT / "artifacts")
    metadata = json.loads((ROOT / "artifacts/metadata.json").read_text())
    torch_path = ROOT / "artifacts/model.pt"
    if torch_path.exists() and file_hash(torch_path) != metadata["weights_sha256"]:
        raise ValueError("Canonical PyTorch checkpoint checksum mismatch")
    result = predictor.predict("itching, skin rash, nodal skin eruptions")
    if result.get("condition", {}).get("id") != "fungal_infection":
        raise ValueError("Release smoke prediction failed")
    print("Model schema, exported weights, reference patterns, and smoke prediction passed.")


if __name__ == "__main__":
    main()
