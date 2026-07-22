"""
Resolve runtime training settings (max_length recommendation, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_max_length(
    config_max_length: int,
    recommendation_path: str | Path | None = None,
    project_root: Path | None = None,
) -> tuple[int, str]:
    """
    Prefer Notebook 3 recommendation JSON when present; else config value.
    """
    if recommendation_path is not None:
        path = Path(recommendation_path)
        if project_root is not None and not path.is_absolute():
            path = Path(project_root) / path
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                payload: dict[str, Any] = json.load(fh)
            recommended = int(
                payload.get("recommended_max_length")
                or payload.get("token_length_report", {}).get("recommended_max_length")
                or config_max_length
            )
            logger.info(
                "Using max_length=%s from recommendation file %s", recommended, path
            )
            return recommended, f"recommendation_file:{path}"

    logger.info("Using max_length=%s from config", config_max_length)
    return int(config_max_length), "config"
