import json
import subprocess
import sys
import tomllib

from healthdl.data import ROOT


def test_runtime_does_not_import_training_stack():
    result = subprocess.run([sys.executable, "-c", "import main,sys; assert not {'torch','sklearn','pandas'} & set(sys.modules); assert main.app.test_client().get('/health').status_code == 200"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_vercel_entrypoint_and_runtime_assets():
    config = json.loads((ROOT / "vercel.json").read_text())
    assert config["framework"] == "flask"
    assert "main.py" in config["functions"]
    assert (ROOT / "public/static/style.css").is_file()
    assert (ROOT / "public/static/app.js").is_file()
    runtime = (ROOT / "requirements.txt").read_text().lower()
    assert "torch" not in runtime and "scikit" not in runtime
    assert "torch" in (ROOT / "requirements-training.txt").read_text()


def test_vercel_pyproject_declares_the_runtime_requirements():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = {line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()}
    assert set(manifest["project"]["dependencies"]) == requirements
    assert manifest["project"]["requires-python"] == ">=3.12,<3.13"


def test_published_provenance_is_evidence_not_approval():
    evidence = json.loads((ROOT / "data/provenance.json").read_text())
    assert evidence["findings"][0]["comparison"] == "table_exact_after_empty_column_removal"
    assert evidence["findings"][1]["overlap_rows"] == 41
    assert "clinical" in " ".join(evidence["limits"]).lower()
    review = (ROOT / "docs/CLINICAL_REVIEW.md").read_text()
    assert "pending qualified reviewer" in review
