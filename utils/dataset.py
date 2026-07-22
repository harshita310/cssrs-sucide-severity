"""
PyTorch Dataset / DataLoader construction for CSSR-S severity classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class CSSRSeverityDataset(Dataset):
    """
    Tokenises posts on-the-fly with MentalBERT WordPiece.

    Truncation side must already be set on ``tokenizer`` (project default: left).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 192,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels length mismatch")
        self.texts = [str(t) for t in texts]
        self.labels = [int(y) for y in labels]
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def stratified_split(
    df: pd.DataFrame,
    *,
    text_column: str = "content",
    label_column: str = "severity",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train / validation / test frames with optional stratification.

    Ratios must sum to 1.0 (within floating tolerance). If stratification is
    requested but a split is impossible (class count < 2 in a partition), the
    function retries that split **without** stratification and logs a warning.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def _split(frame: pd.DataFrame, test_size: float, do_stratify: bool):
        strat = frame[label_column].astype(int) if do_stratify else None
        try:
            return train_test_split(
                frame,
                test_size=test_size,
                random_state=seed,
                stratify=strat,
            )
        except ValueError as err:
            if not do_stratify:
                raise
            logger.warning(
                "Stratified split failed (%s). Retrying without stratification.",
                err,
            )
            return train_test_split(
                frame,
                test_size=test_size,
                random_state=seed,
                stratify=None,
            )

    train_df, temp_df = _split(df, test_size=(1.0 - train_ratio), do_stratify=stratify)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = _split(temp_df, test_size=relative_test, do_stratify=stratify)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split sizes | train=%s val=%s test=%s | stratify=%s",
        len(train_df),
        len(val_df),
        len(test_df),
        stratify,
    )
    return train_df, val_df, test_df


def create_dataloader(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    *,
    text_column: str = "content",
    label_column: str = "severity",
    max_length: int = 192,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_weighted_sampler: bool = False,
    num_labels: int = 7,
) -> DataLoader:
    """
    Build a DataLoader over a split dataframe.

    When ``use_weighted_sampler=True`` (training only), each example is
    sampled inversely proportional to its class frequency so rare severity
    labels appear more often per epoch.
    """
    dataset = CSSRSeverityDataset(
        texts=df[text_column].astype(str).tolist(),
        labels=df[label_column].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    sampler = None
    if use_weighted_sampler:
        labels = np.asarray(dataset.labels, dtype=int)
        sampler = build_weighted_sampler(labels, num_labels=num_labels)
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_balanced_class_weights(
    labels: list[int] | np.ndarray,
    num_labels: int = 7,
) -> torch.Tensor:
    """
    sklearn-style balanced class weights: ``n_samples / (n_classes * count_c)``.

    Missing classes receive weight 1.0 (should not happen with stratified splits).
    """
    from sklearn.utils.class_weight import compute_class_weight

    y = np.asarray(labels, dtype=int)
    classes = np.arange(num_labels)
    present = np.unique(y)
    weights = np.ones(num_labels, dtype=np.float64)
    if len(present) > 0:
        cw = compute_class_weight(class_weight="balanced", classes=present, y=y)
        for cls, w in zip(present, cw):
            weights[int(cls)] = float(w)
    tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info("Balanced class weights: %s", {i: round(float(w), 4) for i, w in enumerate(weights)})
    return tensor


def build_weighted_sampler(
    labels: np.ndarray,
    num_labels: int = 7,
) -> torch.utils.data.WeightedRandomSampler:
    """Inverse-frequency WeightedRandomSampler over training indices."""
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    class_w = 1.0 / counts
    sample_w = class_w[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
    logger.info("WeightedRandomSampler enabled | class_counts=%s", counts.astype(int).tolist())
    return sampler


def split_label_counts(df: pd.DataFrame, label_column: str = "severity") -> dict[str, Any]:
    """Return sorted class counts for logging / experiment tracking."""
    counts = df[label_column].astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}
