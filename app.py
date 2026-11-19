"""InsightAI – Student Outcome Predictor (Flask + scikit-learn)."""
import logging
import math
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("insightai")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # inputs are tiny; reject big bodies

# ---------------------------------------------------------------- model ----
# Looks for MODEL_PATH first, then common file names in the project folder.
CANDIDATES = [os.environ.get("MODEL_PATH"), "logistic.pkl", "logistic(2).pkl"]


def load_model():
    for name in filter(None, CANDIDATES):
        path = Path(name) if Path(name).is_absolute() else BASE_DIR / name
        if path.is_file():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(path, "rb") as fh:
                        model = pickle.load(fh)  # only load files you trust
                log.info("Loaded model from %s", path.name)
                return model
            except Exception:
                log.exception("Could not load %s", path)
    log.error("No model file found. Place logistic.pkl next to app.py.")
    return None


MODEL = load_model()

# ---------------------------------------------------------------- fields ---
# kind "code": categorical field – must be the numeric code from the training
# data (whole number). kind "number": any finite number.
FIELDS = [
    # name, label, hint, kind, group
    ("Major_Category", "Major category", "Numeric code for the student's major, exactly as encoded in the training data.", "code", "Student profile"),
    ("Year_of_Study", "Year of study", "Value as used in the training data (e.g. 1, 2, 3, 4).", "code", "Student profile"),
    ("Pre_Semester_GPA", "Pre-semester GPA", "GPA before the semester, on the same scale as the training data.", "number", "Student profile"),
    ("Post_Semester_GPA", "Post-semester GPA", "GPA after the semester, on the same scale as the training data.", "number", "Student profile"),
    ("Weekly_GenAI_Hours", "Weekly GenAI hours", "Hours per week spent using generative AI tools.", "number", "AI usage"),
    ("Primary_Use_Case", "Primary use case", "Numeric code for the main way AI is used, from the training data.", "code", "AI usage"),
    ("Prompt_Engineering_Skill", "Prompt engineering skill", "Skill score exactly as recorded in the training data.", "number", "AI usage"),
    ("Tool_Diversity", "Tool diversity", "Number or score of distinct AI tools used.", "number", "AI usage"),
    ("Paid_Subscription", "Paid subscription", "Numeric code for paid plan status, from the training data (often 0 or 1).", "code", "AI usage"),
    ("Perceived_AI_Dependency", "Perceived AI dependency", "Dependency score exactly as recorded in the training data.", "number", "AI usage"),
    ("Traditional_Study_Hours", "Traditional study hours", "Hours per week studying without AI tools.", "number", "Study habits & context"),
    ("Institutional_Policy", "Institutional policy", "Numeric code for the institution's AI policy, from the training data.", "code", "Study habits & context"),
    ("Anxiety_Level_During_Exams", "Exam anxiety level", "Anxiety score exactly as recorded in the training data.", "number", "Study habits & context"),
    ("Skill_Retention_Score", "Skill retention score", "Retention score exactly as recorded in the training data.", "number", "Study habits & context"),
]
# The model expects THIS order (matches model.feature_names_in_).
MODEL_ORDER = [
    "Major_Category", "Year_of_Study", "Pre_Semester_GPA", "Weekly_GenAI_Hours",
    "Primary_Use_Case", "Prompt_Engineering_Skill", "Tool_Diversity",
    "Paid_Subscription", "Traditional_Study_Hours", "Perceived_AI_Dependency",
    "Institutional_Policy", "Anxiety_Level_During_Exams", "Post_Semester_GPA",
    "Skill_Retention_Score",
]
META = {f[0]: f for f in FIELDS}


def grouped_fields():
    groups = {}
    for name, label, hint, kind, group in FIELDS:
        groups.setdefault(group, []).append(
            {"name": name, "label": label, "hint": hint, "kind": kind}
        )
    return groups


def parse_inputs(payload):
    """Return (values_in_model_order, errors_by_field)."""
    values, errors = [], {}
    for name in MODEL_ORDER:
        label, kind = META[name][1], META[name][3]
        raw = payload.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = f"{label} is required."
            continue
        try:
            num = float(str(raw).strip())
        except ValueError:
            errors[name] = f"{label} must be a number."
            continue
        if not math.isfinite(num):
            errors[name] = f"{label} must be a finite number."
        elif kind == "code" and num != int(num):
            errors[name] = f"{label} must be a whole-number code from the training data."
        else:
            values.append(num)
    return values, errors


# ---------------------------------------------------------------- routes ---
@app.route("/")
def index():
    model_info = None
    if MODEL is not None:
        model_info = {
            "type": type(MODEL).__name__,
            "features": int(getattr(MODEL, "n_features_in_", len(MODEL_ORDER))),
            "classes": [int(c) for c in getattr(MODEL, "classes_", [])],
        }
    return render_template("index.html", groups=grouped_fields(), model=model_info)


@app.post("/api/predict")
def predict():
    if MODEL is None:
        return jsonify(error="The model file is not available on the server."), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the form values as JSON."), 400

    values, errors = parse_inputs(payload)
    if errors:
        return jsonify(error="Please fix the highlighted fields.", fields=errors), 422

    try:
        x = np.array([values], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # feature-name warning for ndarray input
            pred = MODEL.predict(x)[0]
            probs = None
            if hasattr(MODEL, "predict_proba"):
                p = MODEL.predict_proba(x)[0]
                probs = [
                    {"class": int(c), "probability": round(float(v), 4)}
                    for c, v in zip(MODEL.classes_, p)
                ]
        return jsonify(predicted_class=int(pred), probabilities=probs)
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="Something went wrong while predicting. Please check your values and try again."), 500


@app.get("/healthz")
def healthz():
    return jsonify(status="ok", model_loaded=MODEL is not None)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Request too large."), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    return resp


if __name__ == "__main__":  # local dev only; Render uses `gunicorn app:app`
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
