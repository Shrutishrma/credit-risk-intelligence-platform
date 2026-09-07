import joblib
import shap
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

explainer = shap.TreeExplainer(model)


def explain_prediction(applicant: pd.DataFrame):
    """
    Generate SHAP values for one applicant.
    """

    X = applicant.drop(
        columns=["TARGET", "SK_ID_CURR"],
        errors="ignore"
    )

    X_processed = preprocessor.transform(X)

    shap_values = explainer.shap_values(X_processed)

    # Handle different SHAP versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values[0]

    feature_names = preprocessor.get_feature_names_out()

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })

    explanation["impact"] = explanation["shap_value"].apply(
        lambda x: "Increases risk" if x > 0 else "Reduces risk"
    )

    explanation["absolute_impact"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_impact",
        ascending=False
    )

    return explanation.head(10)