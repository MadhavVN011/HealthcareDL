"""Check the browser inference assets without importing the imaging training stack."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check():
    manifest = json.loads((ROOT / "public/static/vendor/manifest.json").read_text())
    for row in manifest:
        path = ROOT / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Medical dependency checksum mismatch: {row['file']}")
    folder = ROOT / "public/static/models/chest-ct"
    report = json.loads((folder / "report.json").read_text())
    if hashlib.sha256((folder / "model.onnx").read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("CT export/report checksum mismatch")
    if report["classes"] != ["comparator", "covid_associated"] or report["dataset"]["patient_disjoint"] is not False:
        raise ValueError("Unexpected CT research schema or evaluation claim")
    for example in report["examples"]:
        if not (folder / example["file"]).is_file():
            raise ValueError("Missing CT walkthrough image")
    print("Medical dependency checksums, CT export, model schema, and examples passed.")


if __name__ == "__main__":
    check()
