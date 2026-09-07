import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

RULES_PATH = BASE_DIR / "models" / "business_rules.json"


def load_business_rules():
    """
    Load the business-readable credit risk rules.
    """

    with open(RULES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)