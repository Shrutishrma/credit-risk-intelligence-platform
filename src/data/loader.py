from pathlib import Path
import os
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
DATA_PATH = DATA_DIR / "final_applicant_data.parquet"


def _get_secret(name: str):
    """Read a setting from environment variables or Streamlit Secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _download_private_data():
    """
    Download the processed applicant dataset from a PRIVATE Hugging Face
    dataset repository when it is not available locally.

    The access token is never hardcoded and should be supplied through
    environment variables or Streamlit Secrets.
    """
    repo_id = _get_secret("HF_DATASET_REPO")
    token = _get_secret("HF_TOKEN")
    filename = _get_secret("HF_DATASET_FILE") or "final_applicant_data.parquet"

    if not repo_id or not token:
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "For deployed use, configure HF_DATASET_REPO and HF_TOKEN "
            "in the application secrets."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the private dataset."
        ) from exc

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=str(DATA_DIR),
    )

    return Path(downloaded_path)


def load_data():
    """
    Load the final applicant-level dataset.

    Local/Docker:
        Uses data/processed/final_applicant_data.parquet.

    Streamlit Cloud:
        Downloads the same processed file from a private Hugging Face dataset
        repository if the local file does not exist.
    """
    if not DATA_PATH.exists():
        _download_private_data()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found after download attempt: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    # Create readable target label for the application.
    if "DEFAULT_STATUS" not in df.columns and "TARGET" in df.columns:
        df["DEFAULT_STATUS"] = df["TARGET"].map({
            0: "No Default",
            1: "Default"
        })

    return df
