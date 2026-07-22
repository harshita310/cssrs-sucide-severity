"""
Exploratory data analysis helpers for MentalBERT-CSSR (Notebook 1).

Functions here are intentionally pure / side-effect-light so notebooks remain
readable while avoiding duplicated analysis logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Approximate whitespace tokenizer for EDA (not the MentalBERT tokenizer).
# True subword token counts are computed in Notebook 3.


def add_text_length_features(
    df: pd.DataFrame,
    text_column: str = "content",
) -> pd.DataFrame:
    """
    Append character, word, and whitespace-token length columns.

    Notes
    -----
    ``token_count_approx`` uses a simple whitespace split. It is a cheap
    proxy for EDA only. Official max_length recommendations come from the
    MentalBERT tokenizer in Notebook 3.
    """
    out = df.copy()
    text = out[text_column].fillna("").astype(str)

    out["char_count"] = text.str.len()
    out["word_count"] = text.str.split().str.len()
    # Proxy "token" count: non-empty whitespace-separated pieces
    out["token_count_approx"] = out["word_count"]
    # Sentence-ish units (period/question/exclamation) — descriptive only
    out["sentence_punct_count"] = text.str.count(r"[.!?]+")

    return out


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages per column."""
    missing = df.isna().sum()
    summary = pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_pct": (missing.values / len(df) * 100).round(3),
            "dtype": [str(df[c].dtype) for c in missing.index],
        }
    )
    return summary.sort_values("missing_count", ascending=False).reset_index(drop=True)


def summarize_duplicates(
    df: pd.DataFrame,
    text_column: str = "content",
) -> dict[str, Any]:
    """Summarize exact row duplicates and duplicate text bodies."""
    return {
        "n_rows": int(len(df)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_text_rows": int(df[text_column].duplicated().sum()),
        "unique_texts": int(df[text_column].nunique(dropna=False)),
    }


def class_distribution(
    df: pd.DataFrame,
    label_column: str = "severity",
) -> pd.DataFrame:
    """Severity class counts, percentages, and imbalance ratio vs. majority."""
    counts = df[label_column].value_counts().sort_index()
    total = counts.sum()
    majority = counts.max()
    dist = pd.DataFrame(
        {
            "severity": counts.index.astype(int),
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(3),
            "imbalance_ratio_vs_majority": (majority / counts.values).round(3),
        }
    )
    return dist.reset_index(drop=True)


def length_statistics(
    series: pd.Series,
    name: str,
    percentiles: Sequence[float] = (0.50, 0.75, 0.90, 0.95, 0.99),
) -> dict[str, Any]:
    """Compute descriptive statistics for a numeric length series."""
    s = series.dropna().astype(float)
    stats: dict[str, Any] = {
        "feature": name,
        "count": int(s.count()),
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "mean": float(s.mean()) if len(s) else np.nan,
        "std": float(s.std()) if len(s) else np.nan,
        "median": float(s.median()) if len(s) else np.nan,
    }
    for p in percentiles:
        key = f"p{int(p * 100)}"
        stats[key] = float(s.quantile(p)) if len(s) else np.nan
    return stats


def length_statistics_table(
    df: pd.DataFrame,
    columns: Sequence[str],
    percentiles: Sequence[float],
) -> pd.DataFrame:
    """Build a tidy table of length statistics for multiple columns."""
    rows = [length_statistics(df[c], c, percentiles) for c in columns]
    return pd.DataFrame(rows)


def length_by_class(
    df: pd.DataFrame,
    length_column: str,
    label_column: str = "severity",
) -> pd.DataFrame:
    """Per-class length aggregates (mean / median / min / max / p95)."""
    grouped = (
        df.groupby(label_column)[length_column]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max",
            p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    grouped[label_column] = grouped[label_column].astype(int)
    return grouped.sort_values(label_column).reset_index(drop=True)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure and close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)
    return path


def plot_severity_distribution(
    dist: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
    title: str = "C-SSRS Severity Class Distribution",
) -> Path:
    """Bar plot of severity counts."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=dist, x="severity", y="count", ax=ax, color="#2c5f7c")
    ax.set_xlabel("Severity (human annotation)")
    ax.set_ylabel("Number of posts")
    ax.set_title(title)
    for i, row in dist.iterrows():
        ax.text(
            i,
            row["count"] + max(dist["count"]) * 0.01,
            f"{int(row['count'])}\n({row['percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_length_histogram(
    series: pd.Series,
    output_path: Path,
    xlabel: str,
    title: str,
    dpi: int = 150,
    bins: int = 40,
) -> Path:
    """Histogram of a length feature with mean / p95 reference lines."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series.dropna(), bins=bins, color="#2c5f7c", edgecolor="white", alpha=0.9)
    mean_v = series.mean()
    p95_v = series.quantile(0.95)
    ax.axvline(mean_v, color="#c45c26", linestyle="--", linewidth=1.5, label=f"mean={mean_v:.1f}")
    ax.axvline(p95_v, color="#1b7a4e", linestyle=":", linewidth=1.5, label=f"p95={p95_v:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_length_boxplot_by_class(
    df: pd.DataFrame,
    length_column: str,
    label_column: str,
    output_path: Path,
    ylabel: str,
    title: str,
    dpi: int = 150,
) -> Path:
    """Boxplot of length feature stratified by severity."""
    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(df[label_column].dropna().unique())
    sns.boxplot(
        data=df,
        x=label_column,
        y=length_column,
        order=order,
        ax=ax,
        color="#7a9eaf",
    )
    ax.set_xlabel("Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_imbalance_ratio(
    dist: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Bar plot of imbalance ratio relative to the majority class."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=dist,
        x="severity",
        y="imbalance_ratio_vs_majority",
        ax=ax,
        color="#8b4513",
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="majority = 1.0")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Imbalance ratio (majority / class)")
    ax.set_title("Class Imbalance Relative to Majority Severity")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def build_eda_report(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    percentiles: Sequence[float],
) -> dict[str, Any]:
    """
    Assemble a serialisable EDA summary dictionary for metrics export.
    """
    enriched = add_text_length_features(df, text_column=text_column)
    dist = class_distribution(enriched, label_column=label_column)
    length_cols = ["char_count", "word_count", "token_count_approx"]
    length_table = length_statistics_table(enriched, length_cols, percentiles)

    return {
        "n_rows": int(len(enriched)),
        "n_columns": int(enriched.shape[1]),
        "columns": list(df.columns),
        "missing": summarize_missing(df).to_dict(orient="records"),
        "duplicates": summarize_duplicates(df, text_column=text_column),
        "class_distribution": dist.to_dict(orient="records"),
        "length_statistics": length_table.to_dict(orient="records"),
        "label_column": label_column,
        "text_column": text_column,
        "notes": {
            "token_count_approx": (
                "Whitespace proxy only. MentalBERT subword token lengths "
                "are analysed in Notebook 3."
            ),
            "labels": "Human severity annotations are never modified.",
            "ignored_for_training": list(BENCHMARK_SAFE_IGNORE),
        },
    }


BENCHMARK_SAFE_IGNORE: tuple[str, ...] = (
    "gpt_label",
    "claude_label",
    "gemini_label",
    "llama_label",
    "mistral_label",
)
