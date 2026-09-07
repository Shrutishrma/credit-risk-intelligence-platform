import joblib
import shap
import pandas as pd
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

explainer = shap.TreeExplainer(model)


def _clean_feature_name(name: str) -> str:
    """Remove ColumnTransformer / pipeline prefixes."""
    name = str(name)
    return name.split("__", 1)[-1]


def _display_name(name: str) -> str:
    """Convert technical feature names into readable labels."""
    clean = _clean_feature_name(name)
    clean = re_sub_category_suffix(clean)
    return clean.replace("_", " ").strip().title()


def re_sub_category_suffix(name: str) -> str:
    """
    Keep the original categorical feature name while preserving a category
    label when the transformed name is one-hot encoded.

    Examples:
        CODE_GENDER_F -> CODE_GENDER [Female]
        NAME_FAMILY_STATUS_Married -> NAME_FAMILY_STATUS [Married]
    """
    return name


def _find_original_feature(transformed_name: str, original_columns) -> tuple[str | None, str | None]:
    """
    Map a transformed feature back to an original applicant column.

    Returns:
        (original_column, category)
        category is populated for likely one-hot features.
    """
    clean = _clean_feature_name(transformed_name)

    if clean in original_columns:
        return clean, None

    # Longest-match first so columns containing underscores are handled safely.
    candidates = sorted(
        (str(col) for col in original_columns),
        key=len,
        reverse=True,
    )

    for col in candidates:
        prefix = f"{col}_"
        if clean.startswith(prefix):
            category = clean[len(prefix):]
            return col, category

    return None, None


def _raw_value(value):
    """Return a compact business-friendly representation of an applicant value."""
    if pd.isna(value):
        return "Missing in applicant record"

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if np.isfinite(value):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return "Missing in applicant record"

    return str(value)


def _pretty_value(value) -> str:
    """Make common encoded category values readable."""
    text = _raw_value(value)
    if text in {"F", "Female"}:
        return "Female"
    if text in {"M", "Male"}:
        return "Male"
    return text.replace("_", " ").title() if isinstance(value, str) else text


def _pretty_feature(original_feature: str, category: str | None, raw_value) -> str:
    """Create a business-facing feature label."""
    label = original_feature.replace("_", " ").strip().title()

    if category is not None:
        category_text = category.replace("_", " ").strip().title()

        if category.upper() == "F":
            category_text = "Female"
        elif category.upper() == "M":
            category_text = "Male"

        return f"{label} — {category_text}"

    return label


def explain_prediction(applicant: pd.DataFrame):
    """
    Generate SHAP values for one applicant and map transformed model features
    back to the applicant's original columns.

    One-hot encoded features are shown only when the category is active for
    this applicant. This prevents confusing outputs such as showing both
    Gender F and Gender M for the same applicant.
    """
    X = applicant.drop(
        columns=["TARGET", "SK_ID_CURR"],
        errors="ignore"
    )

    X_processed = preprocessor.transform(X)

    if hasattr(X_processed, "toarray"):
        processed_row = X_processed.toarray()[0]
    else:
        processed_row = np.asarray(X_processed)[0]

    shap_values = explainer.shap_values(X_processed)

    # Handle different SHAP versions.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)[0]

    feature_names = preprocessor.get_feature_names_out()
    row = X.iloc[0]
    original_columns = list(X.columns)

    rows = []

    for transformed_name, shap_value in zip(feature_names, shap_values):
        original_feature, category = _find_original_feature(
            transformed_name,
            original_columns
        )

        if original_feature is None:
            # Unknown transformed column; retain it rather than silently dropping.
            raw = "Not available"
            active = True
            display_feature = _display_name(transformed_name)
        else:
            raw_value = row.get(original_feature)
            raw = _pretty_value(raw_value)
            display_feature = _pretty_feature(
                original_feature,
                category,
                raw_value,
            )

            # For one-hot features, only explain the active category for this
            # applicant. Inactive categories are technically valid SHAP inputs,
            # but are confusing as business-facing reasons.
            if category is not None:
                try:
                    transformed_index = list(feature_names).index(transformed_name)
                    transformed_row_value = float(processed_row[transformed_index])
                    active = transformed_row_value > 0.5
                except Exception:
                    active = False
            else:
                active = True

        rows.append({
            "feature": transformed_name,
            "display_feature": display_feature,
            "original_feature": original_feature or transformed_name,
            "category": category,
            "applicant_value": raw,
            "shap_value": float(shap_value),
            "impact": "Increases risk" if shap_value > 0 else "Reduces risk",
            "absolute_impact": abs(float(shap_value)),
            "active": active,
        })

    explanation = pd.DataFrame(rows)

    # Do not show inactive one-hot categories such as Gender M when the
    # applicant is female.
    explanation = explanation[explanation["active"]].copy()

    explanation = explanation.sort_values(
        "absolute_impact",
        ascending=False
    )

    return explanation.head(10).reset_index(drop=True)
