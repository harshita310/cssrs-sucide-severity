"""
Data I/O helpers shared across notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Columns reserved for LLM benchmarking — never used as training targets
BENCHMARK_LABEL_COLUMNS: tuple[str, ...] = (
    "gpt_label",
    "claude_label",
    "gemini_label",
    "llama_label",
    "mistral_label",
)


def load_raw_dataset(
    path: str | Path,
    text_column: str = "content",
    label_column: str = "severity",
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load the CSSR-S Reddit CSV with schema validation.

    Parameters
    ----------
    path:
        Path to the raw CSV.
    text_column:
        Expected text field (default: ``content``).
    label_column:
        Expected human annotation field (default: ``severity``).
    required_columns:
        Optional extra required columns.

    Returns
    -------
    pd.DataFrame
        Raw dataframe (labels are NEVER modified).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Place the CSSR-S CSV under DATA/raw/."
        )

    df = pd.read_csv(path)
    logger.info("Loaded raw dataset from %s | shape=%s", path, df.shape)

    required = list(required_columns or [])
    required.extend([text_column, label_column])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def training_columns(
    df: pd.DataFrame,
    text_column: str = "content",
    label_column: str = "severity",
    ignore_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Return a view with only columns needed for supervised training.

    Benchmark LLM label columns are dropped here; they remain in the raw file
    for later evaluation notebooks.
    """
    ignore = set(ignore_columns or BENCHMARK_LABEL_COLUMNS)
    keep = [c for c in df.columns if c not in ignore]
    # Ensure text + label are present and ordered first for clarity
    ordered = [text_column, label_column] + [
        c for c in keep if c not in {text_column, label_column}
    ]
    return df[ordered].copy()


def save_processed_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Persist the processed CSV (UTF-8). Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved processed dataset → %s | shape=%s", path, df.shape)
    return path


def load_processed_dataset(
    path: str | Path,
    text_column: str = "content",
    label_column: str = "severity",
) -> pd.DataFrame:
    """
    Load a previously saved processed CSV with schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run NOTEBOOKS/02_Preprocessing.ipynb first."
        )

    df = pd.read_csv(path)
    logger.info("Loaded processed dataset from %s | shape=%s", path, df.shape)

    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(
                f"Processed file missing '{col}'. Columns={list(df.columns)}"
            )
    return df
