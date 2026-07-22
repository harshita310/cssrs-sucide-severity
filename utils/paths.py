"""
Filesystem path helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.config import AttributeDict, get_project_root


def ensure_directories(cfg: AttributeDict | None = None, extra: Iterable[Path] | None = None) -> None:
    """
    Create all project output directories if they do not already exist.

    Parameters
    ----------
    cfg:
        Loaded configuration. If None, only default structural dirs are created.
    extra:
        Additional paths to ensure.
    """
    root = get_project_root()
    defaults = [
        root / "DATA" / "RAW",
        root / "DATA" / "processed",
        root / "NOTEBOOKS",
        root / "saved_model",
        root / "RESULTS" / "plots",
        root / "RESULTS" / "metrics",
        root / "configs",
        root / "utils",
    ]

    for path in defaults:
        path.mkdir(parents=True, exist_ok=True)

    if cfg is not None:
        for key in ("save_path", "results_path", "plots_path", "metrics_path"):
            p = cfg.paths.get(key)
            if p is not None:
                Path(p).mkdir(parents=True, exist_ok=True)

    if extra:
        for path in extra:
            Path(path).mkdir(parents=True, exist_ok=True)
