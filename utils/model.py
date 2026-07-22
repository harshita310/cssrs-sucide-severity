"""
MentalBERT encoder for C-SSRS severity classification.

Encoder is locked to ``mental/mental-bert-base-uncased``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from utils.logging_utils import get_logger
from utils.tokenization import MENTALBERT_MODEL_NAME

logger = get_logger(__name__)


def load_mentalbert_for_classification(
    model_name: str = MENTALBERT_MODEL_NAME,
    num_labels: int = 7,
    dropout: float = 0.1,
) -> BertForSequenceClassification:
    """
    Load MentalBERT with a 7-way classification head.

    Uses ``BertForSequenceClassification`` against the MentalBERT checkpoint
    (same Hub repo / weights). Prefers local cache first because the Hub repo
    is gated; falls back to online download when authenticated.
    """
    if model_name != MENTALBERT_MODEL_NAME:
        raise ValueError(
            f"Encoder locked to {MENTALBERT_MODEL_NAME!r}. Got {model_name!r}."
        )

    config = BertConfig.from_pretrained(model_name, local_files_only=True, token=False)
    config.num_labels = num_labels
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    config.classifier_dropout = dropout

    load_errors: list[str] = []
    model: BertForSequenceClassification | None = None

    for kwargs in (
        {"local_files_only": True, "token": False},
        {"token": False},
    ):
        try:
            model = BertForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
            logger.info("Loaded MentalBERT with kwargs=%s", kwargs)
            break
        except Exception as err:
            load_errors.append(repr(err))
            logger.warning("MentalBERT load attempt failed (%s): %s", kwargs, err)

    if model is None:
        raise RuntimeError(
            "Failed to load mental/mental-bert-base-uncased. "
            "Accept the model terms at "
            "https://huggingface.co/mental/mental-bert-base-uncased "
            "then run `huggingface-cli login`. "
            f"Errors: {load_errors}"
        )

    logger.info(
        "Loaded MentalBERT classifier | num_labels=%s dropout=%s params=%s",
        num_labels,
        dropout,
        sum(p.numel() for p in model.parameters()),
    )
    return model


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int,
    best_metric: float,
    history: list[dict[str, Any]] | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> Path:
    """Save a full training checkpoint (model + optional optimizer/scheduler)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "history": history or [],
        "config_snapshot": config_snapshot or {},
        "model_name": MENTALBERT_MODEL_NAME,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)
    logger.info("Saved checkpoint → %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model (and optionally optimizer/scheduler) from checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Loaded checkpoint ← %s | epoch=%s best_metric=%s",
        path,
        checkpoint.get("epoch"),
        checkpoint.get("best_metric"),
    )
    return checkpoint
