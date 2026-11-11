import pytest
import json
import shutil

from healthdl.data import ROOT
from healthdl.examples import DEMO_EXAMPLES
from main import create_app


@pytest.mark.parametrize("path", ["/", "/predict", "/about", "/blog", "/developer", "/contact", "/model", "/data", "/privacy", "/static/mark.svg", "/static/style.css", "/static/app.js", "/static/theme.js", "/health"])
def test_routes_and_assets(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("data", [{}, {"symptoms": ""}, {"symptoms": " "}, {"symptoms": "aching"}])
def test_invalid_form_has_readable_error(client, data):
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert b'role="alert"' in response.data


@pytest.mark.parametrize("raw", ["high fever", "Itching", "itching,"])
def test_previously_crashing_inputs_are_handled(client, raw):
    response = client.post("/predict", data={"symptoms": raw})
    assert response.status_code == 200
    assert b"More symptoms needed" in response.data
    assert b"at least 3 distinct, recognized symptoms" in response.data


def test_form_result_and_health_input_not_logged(client, capsys, caplog):
    response = client.post("/predict", data={"symptoms": "itching,skin rash,nodal skin eruptions"})
    assert response.status_code == 200
    assert b"Dataset match: Fungal infection" in response.data
    assert b"medlineplus.gov/fungalinfections.html" in response.data
    assert "nodal_skin_eruptions" not in caplog.text
    assert "nodal" not in capsys.readouterr().out


@pytest.mark.parametrize("payload", [None, [], {}, {"symptoms": None}, {"symptoms": [7]}, {"symptoms": {"itching": True}}])
def test_api_invalid_payloads(client, payload):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert response.json["status"] == "invalid"


def test_api_malformed_and_wrong_content_type(client):
    assert client.post("/api/predict", data='{', content_type="application/json").status_code == 400
    assert client.post("/api/predict", data="symptoms=itching").status_code == 400


def test_api_valid_list_and_urgent_result(client):
    result = client.post("/api/predict", json={"symptoms": ["itching", "skin rash", "nodal skin eruptions"]})
    assert result.status_code == 200
    assert result.json["condition"]["id"] == "fungal_infection"
    assert 0.7 <= result.json["model_score"] <= 1
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    assert result.json["evaluation"] == {
        "accuracy": report["test"]["accuracy"], "test_patterns": report["split_sizes"]["test"],
        "scope": "held_out_dataset_patterns",
    }
    urgent = client.post("/api/predict", json={"symptoms": ["chest pain"]})
    assert urgent.json["status"] == "urgent"
    assert "condition" not in urgent.json
    assert "model_score" not in urgent.json
    assert "evaluation" not in urgent.json


@pytest.mark.parametrize("example", DEMO_EXAMPLES, ids=lambda item: item["id"])
def test_demo_examples_replace_input_and_run_model(client, predictor, example):
    expected = predictor.predict(list(example["symptoms"]))
    assert expected["status"] == "match"
    assert expected["condition"]["id"] == example["expected_id"]
    response = client.post("/predict", data={"example": example["id"], "symptoms": "chest pain"})
    assert response.status_code == 200
    assert f'Dataset match: {expected["condition"]["label"]}'.encode() in response.data
    assert b"Fictional input" in response.data
    assert b"Held-out test accuracy" in response.data
    assert b"Model score" in response.data
    assert b"61 unique dataset test patterns" in response.data
    assert b"100.0<small>%</small>" in response.data


def test_unknown_demo_example_is_rejected(client):
    response = client.post("/predict", data={"example": "unknown", "symptoms": "itching"})
    assert response.status_code == 400
    assert b"Choose one of the available demo examples" in response.data


@pytest.mark.parametrize("accuracy", [0.75, None, "missing", float("nan"), 1.2])
def test_displayed_accuracy_comes_from_report_or_is_unavailable(tmp_path, accuracy):
    directory = tmp_path / "artifacts"
    shutil.copytree(ROOT / "artifacts", directory)
    path = directory / "report.json"
    if accuracy == "missing":
        path.unlink()
    else:
        report = json.loads(path.read_text())
        report["test"]["accuracy"] = accuracy
        path.write_text(json.dumps(report))
    client = create_app({"TESTING": True, "ARTIFACT_DIR": directory}).test_client()
    response = client.post("/predict", data={"example": "skin"})
    api_result = client.post("/api/predict", json={"symptoms": list(DEMO_EXAMPLES[0]["symptoms"])}).json
    assert response.status_code == 200
    if accuracy == 0.75:
        assert b"75.0<small>%</small>" in response.data
        assert api_result["evaluation"]["accuracy"] == 0.75
    else:
        assert b'class="metric-unavailable">Unavailable' in response.data
        assert "evaluation" not in api_result


def test_body_limits_and_escaping(client):
    assert client.post("/api/predict", data="x" * 9000, content_type="application/json").status_code == 413
    assert client.post("/predict", data={"symptoms": "x" * 9000}).status_code == 413
    response = client.post("/predict", data={"symptoms": "<script>alert(1)</script>"})
    assert response.status_code == 400
    assert b"<script>alert(1)</script>" not in response.data
    assert b"&lt;script&gt;" in response.data


def test_headers_and_debug_defaults(client, app):
    response = client.get("/")
    assert not app.debug
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "Set-Cookie" not in response.headers


def test_missing_model_is_visible_and_does_not_crash(tmp_path):
    app = create_app({"TESTING": True, "ARTIFACT_DIR": tmp_path})
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert b"temporarily unavailable" in client.get("/").data
    assert client.get("/health").status_code == 503
    assert client.post("/predict", data={"symptoms": "itching"}).status_code == 503
    assert client.post("/api/predict", json={"symptoms": ["itching"]}).status_code == 503


def test_paths_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = create_app({"TESTING": True}).test_client()
    assert client.get("/health").status_code == 200
    assert client.get("/static/mark.svg").status_code == 200


def test_https_headers_are_explicitly_configurable():
    client = create_app({"TESTING": True, "ENABLE_HSTS": True}).test_client()
    assert "max-age=31536000" == client.get("/").headers["Strict-Transport-Security"]
