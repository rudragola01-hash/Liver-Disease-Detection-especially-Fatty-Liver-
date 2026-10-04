"""Load the saved model and score one patient."""
from __future__ import annotations

import joblib
import pandas as pd

import config


def load_bundle(path=None) -> dict:
    path = path or config.MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python main.py` first to train one.")
    return joblib.load(path)


def risk_band(prob: float, threshold: float) -> str:
    """Human-friendly band. 'High' means the tuned screening threshold is crossed."""
    if prob >= max(threshold, 0.60):
        return "High"
    if prob >= threshold or prob >= 0.35:
        return "Moderate"
    return "Low"


def predict_patient(patient: dict, bundle: dict | None = None) -> dict:
    """patient = {'Age': 45, 'Gender': 'Male', 'Total_Bilirubin': 1.2, ...}"""
    bundle = bundle or load_bundle()
    row = pd.DataFrame([patient])[config.RAW_FEATURES]
    prob = float(bundle["pipeline"].predict_proba(row)[0, 1])
    thr = float(bundle["threshold"])
    return {
        "probability": prob,
        "screen_positive": prob >= thr,
        "threshold": thr,
        "risk_band": risk_band(prob, thr),
        "model_name": bundle["name"],
    }
