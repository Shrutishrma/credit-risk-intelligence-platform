import joblib
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
threshold = joblib.load(THRESHOLD_PATH)


def predict_risk(applicant: pd.DataFrame):
    """
    Predict default probability and risk band for an applicant.
    """

    X = applicant.drop(
        columns=["TARGET", "SK_ID_CURR"],
        errors="ignore"
    )

    X_processed = preprocessor.transform(X)

    probability = model.predict_proba(
        X_processed
    )[0, 1]

    if probability < 0.30:
        risk_band = "Low"
    elif probability < threshold:
        risk_band = "Medium"
    else:
        risk_band = "High"

    return {
        "probability": probability,
        "risk_score": probability * 100,
        "risk_band": risk_band
    }