"""
MentalBERT tokenization utilities (Notebook 3).

Always uses ``mental/mental-bert-base-uncased``. Never substitute another encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import BertTokenizer, PreTrainedTokenizerBase

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Hard constraint for this research line
MENTALBERT_MODEL_NAME = "mental/mental-bert-base-uncased"
# Standard BERT absolute-position ceiling
MODEL_MAX_POSITIONS = 512
# Candidate sequence lengths commonly used with BERT-base
CANDIDATE_MAX_LENGTHS: tuple[int, ...] = (64, 128, 160, 192, 256, 320, 384, 512)


@dataclass
class TokenLengthReport:
    """Serialisable summary of MentalBERT subword length analysis."""

    model_name: str
    truncation_side: str
    n_texts: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    std_tokens: float
    median_tokens: float
    percentiles: dict[str, float] = field(default_factory=dict)
    coverage_by_candidate: dict[str, float] = field(default_factory=dict)
    recommended_max_length: int = 256
    recommendation_rationale: str = ""
    n_exceeding_recommended: int = 0
    pct_exceeding_recommended: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_mentalbert_tokenizer(
    model_name: str = MENTALBERT_MODEL_NAME,
    truncation_side: str = "left",
) -> PreTrainedTokenizerBase:
    """
    Load the MentalBERT tokenizer and set truncation side.

    Parameters
    ----------
    model_name:
        Must be ``mental/mental-bert-base-uncased`` for this project.
    truncation_side:
        ``\"left\"`` keeps the *end* of the post (recent intent / climax).

    Notes
    -----
    Uses ``BertTokenizer`` (not ``AutoTokenizer``) because newer ``transformers``
    versions probe ``additional_chat_templates`` via the Hub API; a stale or
    invalid HF OAuth session can 401 even when the BERT vocab is already cached.
    MentalBERT is a standard WordPiece BERT tokenizer — ``BertTokenizer`` is
    the correct, stable loader for this research line.
    """
    if model_name != MENTALBERT_MODEL_NAME:
        raise ValueError(
            f"Encoder locked to {MENTALBERT_MODEL_NAME!r}. Got {model_name!r}."
        )
    if truncation_side not in {"left", "right"}:
        raise ValueError(f"Invalid truncation_side: {truncation_side}")

    try:
        tokenizer = BertTokenizer.from_pretrained(
            model_name, local_files_only=True, token=False
        )
    except Exception as local_err:
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name, token=False)
        except Exception as online_err:
            raise RuntimeError(
                "Failed to load mental/mental-bert-base-uncased tokenizer. "
                "The model may be gated: accept the terms on Hugging Face, then "
                "run `huggingface-cli login`. "
                f"Errors: local={local_err!r}; online={online_err!r}"
            ) from online_err

    tokenizer.truncation_side = truncation_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        logger.warning("pad_token was None; set to %s", tokenizer.pad_token)

    logger.info(
        "Loaded tokenizer | model=%s vocab=%s truncation_side=%s",
        model_name,
        getattr(tokenizer, "vocab_size", "?"),
        tokenizer.truncation_side,
    )
    return tokenizer


def encode_length(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = True,
) -> int:
    """Return MentalBERT subword length for one string (no truncation)."""
    encoded = tokenizer.encode(
        text if isinstance(text, str) else str(text),
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    return int(len(encoded))


def compute_token_lengths(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = True,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute subword sequence lengths for a corpus.

    Uses batched ``tokenizer(...)`` without truncation for speed.
    """
    lengths: list[int] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = [str(t) if t is not None else "" for t in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.int32)


def coverage_at_max_length(lengths: np.ndarray, max_length: int) -> float:
    """Fraction of sequences with length <= max_length."""
    if len(lengths) == 0:
        return 0.0
    return float((lengths <= max_length).mean())


def recommend_max_length(
    lengths: np.ndarray,
    *,
    candidates: Sequence[int] = CANDIDATE_MAX_LENGTHS,
    model_limit: int = MODEL_MAX_POSITIONS,
    target_coverage: float = 0.99,
    prefer_near_percentile: float = 0.95,
) -> tuple[int, str, dict[str, float]]:
    """
    Choose a research-practical ``max_length``.

    Policy
    ------
    1. Compute coverage for each candidate <= model_limit.
    2. Prefer the **smallest** candidate that covers ``target_coverage`` (default 99%).
    3. If none reach target coverage, pick the smallest candidate >= p95,
       else fall back to ``model_limit``.
    4. Never exceed MentalBERT / BERT position limit (512).

    Returns
    -------
    recommended, rationale, coverage_by_candidate
    """
    candidates = sorted({int(c) for c in candidates if int(c) <= model_limit})
    coverage = {str(c): coverage_at_max_length(lengths, c) for c in candidates}

    p95 = float(np.quantile(lengths, prefer_near_percentile)) if len(lengths) else 0.0
    p99 = float(np.quantile(lengths, 0.99)) if len(lengths) else 0.0

    qualifying = [c for c in candidates if coverage[str(c)] >= target_coverage]
    if qualifying:
        recommended = min(qualifying)
        rationale = (
            f"Smallest candidate covering >={target_coverage:.0%} of sequences "
            f"(p95={p95:.1f}, p99={p99:.1f}, coverage@{recommended}="
            f"{coverage[str(recommended)]:.4f}). "
            f"Left truncation retains the post ending when length > {recommended}."
        )
        return recommended, rationale, coverage

    # Fallback: smallest candidate at or above p95
    above_p95 = [c for c in candidates if c >= p95]
    if above_p95:
        recommended = min(above_p95)
        rationale = (
            f"No candidate reached {target_coverage:.0%} coverage; "
            f"selected smallest candidate >= p95 ({p95:.1f}) → {recommended} "
            f"(coverage={coverage[str(recommended)]:.4f}, p99={p99:.1f})."
        )
        return recommended, rationale, coverage

    recommended = model_limit
    rationale = (
        f"Lengths exceed typical candidates (p95={p95:.1f}, p99={p99:.1f}); "
        f"using model limit {model_limit}."
    )
    coverage[str(recommended)] = coverage_at_max_length(lengths, recommended)
    return recommended, rationale, coverage


def build_token_length_report(
    lengths: np.ndarray,
    *,
    model_name: str,
    truncation_side: str,
    percentiles: Sequence[float] = (0.50, 0.75, 0.90, 0.95, 0.99),
    candidates: Sequence[int] = CANDIDATE_MAX_LENGTHS,
    target_coverage: float = 0.99,
) -> TokenLengthReport:
    """Aggregate statistics + recommendation into a TokenLengthReport."""
    lengths = np.asarray(lengths, dtype=np.int32)
    pct = {
        f"p{int(p * 100)}": float(np.quantile(lengths, p)) if len(lengths) else float("nan")
        for p in percentiles
    }
    recommended, rationale, coverage = recommend_max_length(
        lengths,
        candidates=candidates,
        target_coverage=target_coverage,
    )
    n_ex = int((lengths > recommended).sum()) if len(lengths) else 0
    pct_ex = float(n_ex / len(lengths) * 100) if len(lengths) else 0.0

    return TokenLengthReport(
        model_name=model_name,
        truncation_side=truncation_side,
        n_texts=int(len(lengths)),
        min_tokens=int(lengths.min()) if len(lengths) else 0,
        max_tokens=int(lengths.max()) if len(lengths) else 0,
        mean_tokens=float(lengths.mean()) if len(lengths) else 0.0,
        std_tokens=float(lengths.std()) if len(lengths) else 0.0,
        median_tokens=float(np.median(lengths)) if len(lengths) else 0.0,
        percentiles=pct,
        coverage_by_candidate=coverage,
        recommended_max_length=int(recommended),
        recommendation_rationale=rationale,
        n_exceeding_recommended=n_ex,
        pct_exceeding_recommended=pct_ex,
    )


def token_lengths_by_class(
    df: pd.DataFrame,
    lengths: np.ndarray,
    label_column: str = "severity",
) -> pd.DataFrame:
    """Per-severity token-length aggregates."""
    tmp = df[[label_column]].copy()
    tmp["token_length"] = lengths
    grouped = (
        tmp.groupby(label_column)["token_length"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max",
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
        )
        .reset_index()
        .sort_values(label_column)
    )
    return grouped.reset_index(drop=True)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)
    return path


def plot_token_length_histogram(
    lengths: np.ndarray,
    output_path: Path,
    *,
    recommended_max_length: int | None = None,
    dpi: int = 150,
    bins: int = 40,
) -> Path:
    """Histogram of MentalBERT sequence lengths with p95 / recommendation lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=bins, color="#2c5f7c", edgecolor="white", alpha=0.9)
    p95 = float(np.quantile(lengths, 0.95))
    p99 = float(np.quantile(lengths, 0.99))
    ax.axvline(p95, color="#1b7a4e", linestyle=":", linewidth=1.5, label=f"p95={p95:.1f}")
    ax.axvline(p99, color="#c45c26", linestyle="--", linewidth=1.5, label=f"p99={p99:.1f}")
    if recommended_max_length is not None:
        ax.axvline(
            recommended_max_length,
            color="#6b2d5c",
            linestyle="-.",
            linewidth=1.8,
            label=f"recommended max_length={recommended_max_length}",
        )
    ax.set_xlabel("Sequence length (MentalBERT subword tokens, incl. special tokens)")
    ax.set_ylabel("Frequency")
    ax.set_title("MentalBERT Token Sequence Length Distribution")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_token_length_boxplot_by_class(
    df: pd.DataFrame,
    lengths: np.ndarray,
    label_column: str,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Boxplot of token lengths stratified by severity."""
    plot_df = df[[label_column]].copy()
    plot_df["token_length"] = lengths
    order = sorted(plot_df[label_column].dropna().unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x=label_column, y="token_length", order=order, ax=ax, color="#7a9eaf")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Token length")
    ax.set_title("MentalBERT Token Length by Severity")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def plot_coverage_curve(
    coverage_by_candidate: dict[str, float],
    output_path: Path,
    *,
    recommended_max_length: int | None = None,
    dpi: int = 150,
) -> Path:
    """Bar/line coverage of corpus vs candidate max_length values."""
    xs = sorted(int(k) for k in coverage_by_candidate)
    ys = [coverage_by_candidate[str(x)] * 100 for x in xs]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, marker="o", color="#2c5f7c", linewidth=2)
    ax.axhline(99.0, color="gray", linestyle="--", linewidth=1, label="99% coverage")
    ax.axhline(95.0, color="gray", linestyle=":", linewidth=1, label="95% coverage")
    if recommended_max_length is not None:
        ax.axvline(
            recommended_max_length,
            color="#6b2d5c",
            linestyle="-.",
            label=f"recommended={recommended_max_length}",
        )
    ax.set_xlabel("Candidate max_length")
    ax.set_ylabel("% of sequences fully covered (no truncation)")
    ax.set_title("Coverage vs Candidate max_length")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=dpi)


def demonstrate_left_truncation(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, Any]:
    """
    Show that left truncation keeps the *end* of a long post, while right
    truncation keeps the beginning.
    """
    full_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    prev = tokenizer.truncation_side

    tokenizer.truncation_side = "left"
    left_ids = tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=max_length
    )
    tokenizer.truncation_side = "right"
    right_ids = tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=max_length
    )
    tokenizer.truncation_side = prev

    left_decoded = tokenizer.decode(left_ids, skip_special_tokens=True)
    right_decoded = tokenizer.decode(right_ids, skip_special_tokens=True)
    full_decoded = tokenizer.decode(full_ids, skip_special_tokens=True)

    # Heuristic: left truncation should align with the suffix of the full decode
    left_is_suffix = full_decoded.endswith(left_decoded) if left_decoded else False
    right_is_prefix = full_decoded.startswith(right_decoded) if right_decoded else False

    return {
        "full_length": len(full_ids),
        "max_length": max_length,
        "truncated": len(full_ids) > max_length,
        "left_truncation_decoded": left_decoded,
        "right_truncation_decoded": right_decoded,
        "left_decoded_is_suffix_of_full": left_is_suffix,
        "right_decoded_is_prefix_of_full": right_is_prefix,
    }
