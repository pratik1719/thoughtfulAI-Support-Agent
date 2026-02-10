import json
import re
from typing import Any, Dict


def normalize(text: str) -> str:
    """Normalize text for better matching."""
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file safely."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
