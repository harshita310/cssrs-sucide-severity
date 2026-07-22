"""
Conservative text preprocessing for MentalBERT-CSSR (Notebook 2).

Design principles
-----------------
1. Preserve clinical / affective signal: negations, emotion words, punctuation
   that carries intensity (``!``, ``?``, ``...``), and emoji.
2. Never modify ``severity`` labels.
3. Clean only what is necessary for model hygiene:
   Unicode normalisation, control/invalid characters, repeated whitespace,
   null rows, and duplicates.
4. Do **not** lowercase, stem, lemmatise, remove stopwords, or strip
   meaning-bearing punctuation — MentalBERT's uncased tokenizer handles
   casing; aggressive NLP cleaning would destroy suicide-relevant cues.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import pandas as pd

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Control characters except tab / newline / carriage return (handled separately)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
# Unicode replacement character and other non-characters often seen in bad scrapes
_INVALID_UNICODE_RE = re.compile(r"[\uFFFE\uFFFF\uFFFD]")
# Zero-width and BOM-like characters that add no linguistic signal
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
# Any whitespace run (space, tab, newline, unicode spaces) → single space
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PreprocessReport:
    """Audit trail for a preprocessing run (paper / reproducibility artefact)."""

    n_input_rows: int = 0
    n_output_rows: int = 0
    n_dropped_null_text: int = 0
    n_dropped_null_label: int = 0
    n_dropped_blank_text: int = 0
    n_dropped_invalid_label: int = 0
    n_dropped_exact_duplicates: int = 0
    n_dropped_duplicate_text: int = 0
    n_texts_changed_by_cleaning: int = 0
    valid_severity_labels: list[int] = field(default_factory=lambda: list(range(7)))
    operations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Normalise Unicode to a canonical form (default NFC).

    NFC keeps visually identical characters consistent without aggressively
    decomposing characters that may appear in clinical slang / emoji sequences.
    """
    return unicodedata.normalize(form, text)


def remove_invalid_characters(text: str) -> str:
    """
    Strip control chars, replacement chars, and zero-width glue.

    Keeps letters, digits, punctuation, emoji, and combining marks.
    """
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _INVALID_UNICODE_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    # Drop other non-printable categories except standard space separators
    cleaned_chars: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category.startswith("C") and ch not in "\t\n\r":
            # Other control / surrogate / private-use leftovers
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars)


def collapse_whitespace(text: str) -> str:
    """Replace any whitespace run with a single ASCII space and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(
    text: str | float | None,
    *,
    unicode_form: str = "NFC",
    unescape_html: bool = True,
) -> str:
    """
    Apply the conservative cleaning pipeline to a single string.

    Steps
    -----
    1. Coerce to string (null-safe caller should drop nulls first).
    2. Optional HTML entity unescape (``&amp;`` → ``&``) — meaning-preserving.
    3. Unicode normalisation (NFC).
    4. Remove invalid / control / zero-width characters.
    5. Collapse repeated whitespace.

    Explicitly **not** done: lowercasing, stopword removal, negation stripping,
    punctuation deletion, emoji removal, URL wiping, stemming.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    cleaned = str(text)

    if unescape_html:
        cleaned = html.unescape(cleaned)

    cleaned = normalize_unicode(cleaned, form=unicode_form)
    cleaned = remove_invalid_characters(cleaned)
    cleaned = collapse_whitespace(cleaned)
    return cleaned


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    text_column: str = "content",
    label_column: str = "severity",
    valid_labels: Sequence[int] | None = None,
    unicode_form: str = "NFC",
    unescape_html: bool = True,
    drop_exact_duplicates: bool = True,
    drop_duplicate_text: bool = True,
    keep_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, PreprocessReport]:
    """
    Full-frame preprocessing with a detailed drop audit.

    Parameters
    ----------
    df:
        Raw (or raw-subset) dataframe. Labels are never remapped.
    text_column / label_column:
        Schema fields.
    valid_labels:
        Allowed severity values (default ``0..6``). Rows outside this set
        are dropped as invalid — labels themselves are not rewritten.
    keep_columns:
        If provided, restrict output to these columns (order preserved).
        Useful to persist LLM benchmark columns alongside cleaned text for
        later comparison without using them as training targets.

    Returns
    -------
    processed : pd.DataFrame
    report : PreprocessReport
    """
    if text_column not in df.columns:
        raise KeyError(f"Missing text column: {text_column}")
    if label_column not in df.columns:
        raise KeyError(f"Missing label column: {label_column}")

    labels = list(valid_labels) if valid_labels is not None else list(range(7))
    report = PreprocessReport(
        n_input_rows=int(len(df)),
        valid_severity_labels=labels,
        operations=[
            "html_unescape" if unescape_html else "skip_html_unescape",
            f"unicode_normalize_{unicode_form}",
            "remove_invalid_characters",
            "collapse_whitespace",
            "drop_null_blank_text_or_label",
            "drop_invalid_severity_labels",
            "drop_exact_duplicates" if drop_exact_duplicates else "keep_exact_duplicates",
            "drop_duplicate_text" if drop_duplicate_text else "keep_duplicate_text",
            "labels_never_modified",
        ],
    )

    out = df.copy()
    original_text = out[text_column].astype(str)

    # --- text cleaning (vectorised via apply; volume is small / research-grade) ---
    out[text_column] = out[text_column].map(
        lambda x: clean_text(x, unicode_form=unicode_form, unescape_html=unescape_html)
    )
    report.n_texts_changed_by_cleaning = int((original_text.fillna("") != out[text_column]).sum())

    # --- null / blank drops ---
    null_text = out[text_column].isna()
    # After clean_text, NaN becomes ""; treat blank separately for clearer audit
    report.n_dropped_null_text = int(df[text_column].isna().sum())

    null_label = out[label_column].isna()
    report.n_dropped_null_label = int(null_label.sum())

    blank_text = out[text_column].str.len().eq(0)
    # blanks that were not already counted as raw nulls
    report.n_dropped_blank_text = int((blank_text & df[text_column].notna()).sum())

    mask_keep = (~null_label) & (~blank_text)
    out = out.loc[mask_keep].copy()

    # --- invalid severity values (do not remap; drop) ---
    # Coerce to numeric for membership test without changing stored dtype yet
    severity_numeric = pd.to_numeric(out[label_column], errors="coerce")
    valid_mask = severity_numeric.isin(labels)
    report.n_dropped_invalid_label = int((~valid_mask).sum())
    out = out.loc[valid_mask].copy()
    # Store severity as int — same values, cleaner dtype for modelling
    out[label_column] = severity_numeric.loc[valid_mask].astype(int)

    # --- duplicates ---
    if drop_exact_duplicates:
        before = len(out)
        out = out.drop_duplicates(keep="first")
        report.n_dropped_exact_duplicates = before - len(out)

    if drop_duplicate_text:
        before = len(out)
        out = out.drop_duplicates(subset=[text_column], keep="first")
        report.n_dropped_duplicate_text = before - len(out)

    out = out.reset_index(drop=True)

    if keep_columns is not None:
        missing = [c for c in keep_columns if c not in out.columns]
        if missing:
            raise KeyError(f"keep_columns not present after preprocess: {missing}")
        out = out.loc[:, list(keep_columns)].copy()

    report.n_output_rows = int(len(out))
    logger.info(
        "Preprocessing complete | in=%s out=%s dropped=%s texts_changed=%s",
        report.n_input_rows,
        report.n_output_rows,
        report.n_input_rows - report.n_output_rows,
        report.n_texts_changed_by_cleaning,
    )
    return out, report


def before_after_examples(
    df_before: pd.DataFrame,
    text_column: str = "content",
    n: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample rows where cleaning changed the text (for notebook QA).
    """
    raw_as_str = df_before[text_column].fillna("").astype(str)
    cleaned = raw_as_str.map(clean_text)
    changed = raw_as_str != cleaned
    candidates = df_before.loc[changed].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["raw_text", "cleaned_text"])

    sample = candidates.sample(n=min(n, len(candidates)), random_state=seed)
    return pd.DataFrame(
        {
            "raw_text": sample[text_column].fillna("").astype(str).values,
            "cleaned_text": sample[text_column].map(clean_text).values,
        }
    )
