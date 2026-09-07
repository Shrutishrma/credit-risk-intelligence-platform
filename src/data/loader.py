from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_applicant_data.parquet"
)


def load_data():
    """
    Load the final applicant-level dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    # Create readable target label for the application
    if "DEFAULT_STATUS" not in df.columns and "TARGET" in df.columns:
        df["DEFAULT_STATUS"] = df["TARGET"].map({
            0: "No Default",
            1: "Default"
        })

    return df