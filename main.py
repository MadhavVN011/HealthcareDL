"""Flask interface for the trained PyTorch symptom classifier."""

from pathlib import Path
from pickle import UnpicklingError
import json
import math
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from healthdl.data import ROOT
from healthdl.examples import DEMO_EXAMPLES
from healthdl.inference import InputError, Predictor


def create_app(config=None):
    application = Flask(__name__, static_folder="public/static", static_url_path="/static")
    application.config.from_mapping(
        ARTIFACT_DIR=ROOT / "artifacts", MAX_CONTENT_LENGTH=8192,
        DEBUG=False, TESTING=False,
        ENABLE_HSTS=os.environ.get("HEALTHDL_HTTPS", "0") == "1",
    )
    if config:
        application.config.update(config)
    predictor = None
    try:
        predictor = Predictor(Path(application.config["ARTIFACT_DIR"]))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, UnpicklingError):
        application.logger.error("Model unavailable. Restore the artifact bundle or run python train.py.")
    application.extensions["predictor"] = predictor
    try:
        report = json.loads((Path(application.config["ARTIFACT_DIR"]) / "report.json").read_text())
    except (OSError, ValueError):
        report = None
    evaluation = None
    try:
        accuracy = report["test"]["accuracy"]
        test_patterns = report["split_sizes"]["test"]
        if (type(accuracy) in (int, float) and math.isfinite(accuracy) and 0 <= accuracy <= 1
                and type(test_patterns) is int and test_patterns > 0):
            evaluation = {"accuracy": accuracy, "test_patterns": test_patterns,
                          "scope": "held_out_dataset_patterns"}
    except (KeyError, TypeError):
        pass

    def predict(raw):
        result = application.extensions["predictor"].predict(raw)
        if result["status"] == "match" and evaluation is not None:
            result["evaluation"] = dict(evaluation)
        return result

    @application.context_processor
    def project_context():
        return {"model_ready": application.extensions["predictor"] is not None,
                "project_report": report, "app_version": "1.0"}

    def page(result=None, message=None, submitted="", status=200, example=None):
        engine = application.extensions["predictor"]
        return render_template(
            "index.html", result=result, message=message, submitted=submitted,
            symptoms=sorted(engine.features) if engine else [], available=engine is not None,
            demo_examples=DEMO_EXAMPLES, active_example=example,
            minimum_symptoms=engine.policy["min_symptoms"] if engine else None,
        ), status

    @application.after_request
    def response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=(self)"
        response.headers["X-Frame-Options"] = "DENY"
        if application.config["ENABLE_HSTS"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        # Only the document workspace needs WebAssembly and local file previews.
        # Files and extracted medical content are never submitted to this server.
        medical = request.endpoint == "medical_reports" or request.path.startswith("/static/vendor/")
        scripts = "'self' 'wasm-unsafe-eval'" if medical else "'self'"
        images = "'self' blob: data:" if medical else "'self'"
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; script-src {scripts}; style-src 'self'; img-src {images}; "
            "worker-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @application.get("/reports")
    def medical_reports():
        try:
            ct_report = json.loads((ROOT / "public/static/models/chest-ct/report.json").read_text())
        except (OSError, ValueError):
            ct_report = None
        return render_template("reports.html", ct_report=ct_report)

    @application.get("/")
    @application.get("/predict")
    def index():
        return page()

    @application.post("/predict")
    def predict_form():
        engine = application.extensions["predictor"]
        if engine is None:
            return page(message="The demo is temporarily unavailable. Please try again later.", status=503)
        raw = request.form.get("symptoms", "")
        try:
            example = None
            if "example" in request.form:
                example = next((item for item in DEMO_EXAMPLES if item["id"] == request.form["example"]), None)
                if example is None:
                    raise InputError("Choose one of the available demo examples.")
                raw = ", ".join(name.replace("_", " ") for name in example["symptoms"])
            return page(result=predict(raw), submitted=raw, example=example)
        except InputError as error:
            return page(message=str(error), submitted=raw, status=400)

    @application.post("/api/predict")
    def predict_api():
        engine = application.extensions["predictor"]
        if engine is None:
            return jsonify(status="unavailable", message="The demo is temporarily unavailable."), 503
        if not request.is_json:
            return jsonify(status="invalid", message="Send a JSON object with a symptoms field."), 400
        try:
            payload = request.get_json()
        except BadRequest:
            return jsonify(status="invalid", message="The request body is not valid JSON."), 400
        if not isinstance(payload, dict):
            return jsonify(status="invalid", message="Send a JSON object with a symptoms field."), 400
        try:
            return jsonify(predict(payload.get("symptoms")))
        except InputError as error:
            return jsonify(status="invalid", message=str(error)), 400

    @application.get("/health")
    def health():
        ready = application.extensions["predictor"] is not None
        return jsonify(status="ready" if ready else "unavailable"), 200 if ready else 503

    @application.get("/model")
    def model_details():
        curves = {"training": "", "validation": ""}
        if report and report.get("history"):
            history = report["history"]
            maximum = max(max(row["train_loss"], row["validation_loss"]) for row in history)
            for name, key in [("training", "train_loss"), ("validation", "validation_loss")]:
                curves[name] = " ".join(
                    f"{40 + 640 * i / max(1, len(history) - 1):.2f},{210 - 175 * row[key] / max(maximum, 0.001):.2f}"
                    for i, row in enumerate(history)
                )
        return render_template("model.html", report=report, curves=curves)

    @application.get("/data")
    def data_details():
        provenance_path = ROOT / "data" / "provenance.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
        return render_template("data.html", provenance=provenance)

    @application.get("/privacy")
    def privacy():
        return render_template("privacy.html")

    @application.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        if request.path.startswith("/api/"):
            return jsonify(status="invalid", message="The request is too large."), 413
        return page(message="The symptom list is too long.", status=413)

    for route in ["about", "blog", "developer", "contact"]:
        application.add_url_rule(
            f"/{route}", endpoint=route,
            view_func=lambda template=route: render_template(f"{template}.html"),
        )
    return application


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
