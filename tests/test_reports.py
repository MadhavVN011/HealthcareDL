import hashlib
import json
import shutil
import subprocess
from urllib.parse import urlparse

import numpy as np
import pytest

from healthdl.data import ROOT


@pytest.mark.parametrize("path", ["/reports", "/static/medical-core.mjs", "/static/medical-imaging.mjs", "/static/medical-reports.mjs", "/static/models/chest-ct/report.json", "/static/examples/blood-report.pdf"])
def test_report_routes(client, path):
    assert client.get(path).status_code == 200


def test_only_document_workspace_enables_wasm_and_local_previews(client):
    ordinary = client.get("/").headers["Content-Security-Policy"]
    workspace = client.get("/reports").headers["Content-Security-Policy"]
    assert "wasm-unsafe-eval" not in ordinary
    assert "'wasm-unsafe-eval'" in workspace
    assert "'unsafe-eval'" not in workspace
    assert "img-src 'self' blob: data:" in workspace
    assert "worker-src 'self'" in workspace
    assert "object-src 'none'" in workspace
    assert client.post("/reports").status_code == 405


def test_chest_bundle_and_split_provenance():
    public = ROOT / "public/static/models/chest-ct"
    report = json.loads((ROOT / "artifacts/chest_ct/report.json").read_text())
    summary = json.loads((public / "report.json").read_text())
    assert json.loads((ROOT / "artifacts/chest_ct_runtime.json").read_text()) == summary
    assert hashlib.sha256((public / "model.onnx").read_bytes()).hexdigest() == summary["model_sha256"]
    assert summary["dataset"]["patient_disjoint"] is True
    assert summary["dataset"]["license"] == "CC0-1.0"
    partitions = [set(values) for values in report["split_hashes"].values()]
    assert all(not left & right for i, left in enumerate(partitions) for right in partitions[i+1:])
    assert sum(map(len, partitions)) == report["dataset"]["unique_images"] == 4169
    matrix = np.array(report["test"]["confusion_matrix"])
    assert matrix.trace() / matrix.sum() == report["test"]["accuracy"]
    assert matrix.sum() == report["test"]["count"] == 833
    assert summary["test"] == report["test"]
    assert report["onnx_max_abs_error"] < 0.0001


def test_ct_controls_work_without_public_files_in_server_bundle(app, monkeypatch, tmp_path):
    import main
    monkeypatch.setattr(main, "ROOT", tmp_path)
    response = app.test_client().get("/reports")
    assert response.status_code == 200
    assert b'data-unavailable="true"' not in response.data
    assert b"83.7%" in response.data


def test_missing_ct_runtime_report_disables_only_ct_controls(app, tmp_path):
    app.config["CT_REPORT_PATH"] = tmp_path / "missing.json"
    response = app.test_client().get("/reports")
    assert response.status_code == 200
    assert b'data-unavailable="true"' in response.data
    assert b'Try fictional example' in response.data


def test_vendored_medical_assets_have_pinned_checksums_and_licenses():
    manifest = json.loads((ROOT / "public/static/vendor/manifest.json").read_text())
    for row in manifest:
        path = ROOT / row["file"]
        assert path.is_file(), row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["file"]
        assert row["license"] in {"Apache-2.0", "MIT"}
    assert (ROOT / "public/static/vendor/tesseract/tesseract-core-relaxedsimd-lstm.wasm.js").is_file()
    assert (ROOT / "public/static/vendor/onnx/ort-wasm-simd-threaded.wasm").is_file()


LAB_CASES = [
    ("Hemoglobin 10.8 g/dL 12.0 - 15.5", "below"),
    ("Glucose 100 mg/dL 70 - 100", "within"),
    ("Glucose 100 mg/dL <100", "above"),
    ("Glucose 100 mg/dL <=100", "within"),
    ("HDL 40 mg/dL >40", "below"),
    ("HDL 40 mg/dL >=40", "within"),
    ("WBC 6.1 x10^9/L 4 - 11", "within"),
    ("Platelets 275,000 cells/uL 150,000 - 450,000", "within"),
    ("Creatinine 1.3 mg/dL 0.5 - 1.2", "above"),
    ("Hemoglobin 110 g/L 12 - 16 g/dL", "review"),
    ("Hemoglobin 10.5 12 - 16", "review"),
    ("Hemoglobin 10.5 g/dL", "review"),
    ("Hemoglobin 10.5 g/dL male 13-17 female 12-15", "review"),
    ("Glucose 5,2 mmol/L 3.9 - 5.5", "review"),
    ("Glucose 5.2 mmol/L 3,9 - 5,5", "review"),
    ("Glucose 100 mg/dL 120 - 70", "review"),
    ("Glucose >=200 mg/dL 70 - 99", "above"),
    ("Glucose <200 mg/dL 70 - 99", "review"),
    ("Glucose <20 mg/dL 70 - 99", "below"),
    ("Glucose >70 mg/dL 70 - 99", "review"),
    ("TSH 2 mIU/L 0.4 - 4", "within"),
    ("TSH 2 mIU/L 0.4 - 4 uIU/mL", "review"),
    ("Hemoglobin A1c 5.4 % 4 - 5.6", "within"),
    ("Glucose 100 mg/dL 70 - 99 150", "review"),
    ("Glucose 1 mg/dL -1 - 2", "review"),
]


@pytest.fixture(scope="module")
def parsed_lab_cases():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the standalone JavaScript parser checks")
    program = "import {interpretLabText} from " + json.dumps((ROOT / "public/static/medical-core.mjs").as_uri()) + "; let raw=''; for await (const chunk of process.stdin) raw+=chunk; console.log(JSON.stringify(JSON.parse(raw).map(interpretLabText)));"
    cases = [text for text, _ in LAB_CASES] + ["Hemoglobin 10.5 g/dL 12-16\nHemoglobin 14 g/dL 12-16", "<script>alert('x')</script>\nAn unknown test 3 mg/L 1-2"]
    result = subprocess.run([node, "--input-type=module", "-e", program], input=json.dumps(cases), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("index,case", list(enumerate(LAB_CASES)), ids=[text for text, _ in LAB_CASES])
def test_range_comparisons_handle_bounds_units_and_ambiguity(parsed_lab_cases, index, case):
    text, expected = case
    result = parsed_lab_cases[index]
    assert len(result["rows"]) == 1, text
    assert result["rows"][0]["status"] == expected, result


def test_duplicate_tests_and_unknown_rows_do_not_get_invented_interpretations(parsed_lab_cases):
    duplicate, unknown = parsed_lab_cases[-2:]
    assert duplicate["review"] == 2 and duplicate["compared"] == 0
    assert unknown["rows"] == [] and unknown["compared"] == 0


@pytest.mark.parametrize("extension,method", [("pdf", "PDF text extraction"), ("png", "Local image OCR"), ("txt", "Text report")])
def test_actual_report_files_are_processed_locally(live_url, browser_page, extension, method):
    from playwright.sync_api import expect
    page = browser_page
    posts, external, errors = [], [], []
    page.on("request", lambda request: posts.append(request.url) if request.method != "GET" else None)
    page.on("request", lambda request: external.append(request.url) if urlparse(request.url).scheme in {"http", "https"} and not request.url.startswith(live_url) else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(live_url + "/reports")
    page.get_by_role("button", name="Try fictional example").click()
    page.set_input_files("#blood-file", str(ROOT / f"public/static/examples/blood-report.{extension}"))
    assert page.input_value("#blood-text") == ""
    page.get_by_role("button", name="Analyze blood report").click()
    expect(page.locator("#blood-results")).to_be_visible(timeout=60000)
    expect(page.locator("#blood-method")).to_have_text(method)
    assert page.locator("#blood-rows tr").count() >= 3
    expect(page.locator("#blood-error")).to_be_hidden()
    assert page.locator("#blood-source").text_content()
    page.get_by_role("button", name="Clear report and results").click()
    assert page.locator("#blood-source").text_content() == ""
    assert page.input_value("#blood-file") == ""
    assert page.locator("#blood-rows tr").count() == 0
    assert not posts and not external and not errors


def test_ct_browser_preprocessing_and_inference_match_pytorch_export(live_url, browser_page):
    page = browser_page
    page.goto(live_url + "/reports")
    report = json.loads((ROOT / "public/static/models/chest-ct/report.json").read_text())
    for example in report["examples"]:
        actual = page.evaluate("""async (file) => {
          const {classifyCtPixels} = await import('/static/medical-imaging.mjs');
          const response = await fetch('/static/models/chest-ct/' + file);
          const image = await createImageBitmap(await response.blob());
          const canvas = document.createElement('canvas');
          canvas.width = image.width; canvas.height = image.height;
          const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
          const result = await classifyCtPixels(ctx.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height);
          image.close(); return result.logits;
        }""", example["file"])
        np.testing.assert_allclose(actual, example["logits"], atol=0.001, rtol=0.0001)


def test_ct_file_classification_requires_confirmation_and_does_not_upload(live_url, browser_page):
    from playwright.sync_api import expect
    page = browser_page
    page.set_viewport_size({"width": 375, "height": 812})
    page.emulate_media(color_scheme="dark")
    requests, errors = [], []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(live_url + "/reports")
    page.set_input_files("#ct-file", str(ROOT / "public/static/models/chest-ct/example-0.png"))
    page.get_by_role("button", name="Analyze CT slice").click()
    assert page.locator("#ct-results").is_hidden()
    page.check("#ct-confirm")
    page.get_by_role("button", name="Analyze CT slice").click()
    expect(page.locator("#ct-results")).to_be_visible(timeout=60000)
    expect(page.locator("#ct-result-label")).to_have_text("Comparator dataset pattern")
    expect(page.locator("#ct-result-accuracy")).to_have_text("83.7%")
    assert "833" in page.locator("#ct-result-support").inner_text()
    assert "42" in page.locator("#ct-result-support").inner_text()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.get_by_role("button", name="Clear image and results").click()
    assert page.locator("#ct-preview").get_attribute("src") is None
    assert page.locator("#ct-results").is_hidden()
    assert not errors
    assert all(method == "GET" and (url.startswith(live_url) or url.startswith("blob:")) for method, url in requests)


def test_invalid_medical_files_fail_without_a_prediction(live_url, browser_page):
    from playwright.sync_api import expect
    page = browser_page
    page.goto(live_url + "/reports")
    page.set_input_files("#blood-file", {"name": "fake.pdf", "mimeType": "application/pdf", "buffer": b"not a PDF"})
    page.get_by_role("button", name="Analyze blood report").click()
    expect(page.locator("#blood-error")).to_be_visible()
    assert page.locator("#blood-results").is_hidden()
    page.set_input_files("#ct-file", {"name": "scan.dcm", "mimeType": "application/dicom", "buffer": b"0" * 128 + b"DICM"})
    page.check("#ct-confirm")
    page.get_by_role("button", name="Analyze CT slice").click()
    expect(page.locator("#ct-error")).to_be_visible()
    assert page.locator("#ct-results").is_hidden()


def test_scanned_pdf_uses_local_ocr(live_url, browser_page, tmp_path):
    from PIL import Image
    from playwright.sync_api import expect
    path = tmp_path / "scanned-report.pdf"
    with Image.open(ROOT / "public/static/examples/blood-report.png") as image:
        image.convert("RGB").save(path, "PDF")
    page = browser_page
    page.goto(live_url + "/reports")
    page.set_input_files("#blood-file", str(path))
    page.get_by_role("button", name="Analyze blood report").click()
    expect(page.locator("#blood-results")).to_be_visible(timeout=60000)
    expect(page.locator("#blood-method")).to_have_text("PDF text + local OCR")
    assert page.locator("#blood-rows tr").count() >= 3
    expect(page.locator("#blood-error")).to_be_hidden()


def test_long_pdf_is_rejected_instead_of_partially_interpreted(live_url, browser_page, tmp_path):
    from PIL import Image
    from playwright.sync_api import expect
    path = tmp_path / "long-report.pdf"
    with Image.new("RGB", (100, 100), "white") as image:
        image.save(path, "PDF", save_all=True, append_images=[image] * 5)
    page = browser_page
    page.goto(live_url + "/reports")
    page.set_input_files("#blood-file", str(path))
    page.get_by_role("button", name="Analyze blood report").click()
    expect(page.locator("#blood-error")).to_contain_text("at most 5 pages")
    expect(page.locator("#blood-results")).to_be_hidden()


@pytest.mark.parametrize("problem", ["checksum", "accuracy"])
def test_inconsistent_ct_model_bundle_is_rejected(live_url, browser_page, problem):
    from playwright.sync_api import expect
    page = browser_page
    report = json.loads((ROOT / "public/static/models/chest-ct/report.json").read_text())
    if problem == "checksum":
        report["model_sha256"] = "0" * 64
    else:
        report["test"]["accuracy"] = 1.2
    page.route("**/models/chest-ct/report.json", lambda route: route.fulfill(json=report))
    page.goto(live_url + "/reports")
    page.get_by_role("button", name="Comparator sample", exact=True).click()
    expect(page.locator("#ct-error")).to_contain_text("integrity check", timeout=30000)
    expect(page.locator("#ct-results")).to_be_hidden()


def test_pasted_source_text_is_not_executed_as_html(live_url, browser_page):
    from playwright.sync_api import expect
    page = browser_page
    page.goto(live_url + "/reports")
    page.get_by_text("Paste report text instead", exact=True).click()
    page.fill("#blood-text", 'Hemoglobin 10.8 g/dL 12-15.5\n<img src=x onerror="window.medicalInjection=true">')
    page.get_by_role("button", name="Analyze blood report").click()
    expect(page.locator("#blood-results")).to_be_visible()
    assert page.locator("#blood-source img").count() == 0
    assert page.evaluate("window.medicalInjection === undefined")
