"""Text normalization helpers for graph aliases and SHAP tokens."""

from __future__ import annotations

import re
import unicodedata


def normalize_key(text: str) -> str:
    """Normalize free text into a stable alias key."""
    normalized = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]+", "", normalized)
    return re.sub(r"\s+", " ", normalized)
