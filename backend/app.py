"""
app.py
-------
Flask REST API that serves the trained Loan Approval model.

Endpoints:
  GET  /api/health        -> simple health check
  POST /api/predict       -> takes applicant details, returns prediction
  GET  /api/feature-info  -> returns metadata (used by frontend dropdowns)

Run with:
  python app.py
The API will start at http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "..", "ml")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ---- Load trained artifacts once at startup ----
model = joblib.load(os.path.join(ML_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(ML_DIR, "scaler.joblib"))
encoders = joblib.load(os.path.join(ML_DIR, "encoders.joblib"))
meta = joblib.load(os.path.join(ML_DIR, "feature_meta.joblib"))

NUMERIC_COLS = meta["numeric_cols"]
CATEGORICAL_COLS = meta["categorical_cols"]
FEATURE_ORDER = meta["feature_order"]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/api/feature-info", methods=["GET"])
def feature_info():
    """Tells the frontend what categorical options are valid (built from the encoders)."""
    options = {col: list(le.classes_) for col, le in encoders.items()}
    return jsonify({"categorical_options": options, "numeric_fields": NUMERIC_COLS})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        row = {}
        for col in NUMERIC_COLS:
            row[col] = float(data[col])
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            value = data[col]
            if value not in le.classes_:
                return jsonify({"error": f"Invalid value '{value}' for field '{col}'. "
                                          f"Expected one of {list(le.classes_)}"}), 400
            row[col] = int(le.transform([value])[0])
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    # Build the feature vector in the exact order the model was trained on
    ordered = [row[c] for c in FEATURE_ORDER]
    X = np.array(ordered, dtype=float).reshape(1, -1)

    # Scale only the numeric columns (same as training)
    numeric_idx = [FEATURE_ORDER.index(c) for c in NUMERIC_COLS]
    X_numeric_scaled = scaler.transform(X[:, numeric_idx])
    X[:, numeric_idx] = X_numeric_scaled

    prob_approved = float(model.predict_proba(X)[0][1])
    prediction = int(prob_approved >= 0.5)

    # Simple "risk factor" explanation using model coefficients (interview talking point:
    # this is a lightweight explainability layer on top of the prediction)
    risk_factors = []
    if data.get("DebtToIncomeRatio", 0) and float(data["DebtToIncomeRatio"]) > 0.5:
        risk_factors.append("High debt-to-income ratio")
    if float(data.get("CreditScore", 700)) < 600:
        risk_factors.append("Below-average credit score")
    if int(data.get("ExistingLoans", 0)) >= 3:
        risk_factors.append("Multiple existing loans")
    if not risk_factors:
        risk_factors.append("No major risk flags detected")

    return jsonify({
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": round(prob_approved, 4),
        "risk_factors": risk_factors,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
