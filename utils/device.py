"""
Device / AMP helpers for MentalBERT-CSSR training.
"""

from __future__ import annotations

from typing import Any

import torch

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str = "cuda") -> torch.device:
    """
    Resolve the runtime device.

    Uses CUDA when available and requested; otherwise falls back to CPU.
    """
    requested = (requested or "cuda").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Using CUDA device: %s | capability=%s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
        )
        return device

    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable — falling back to CPU.")

    device = torch.device("cpu")
    logger.info("Using device: cpu")
    return device


def amp_enabled(use_mixed_precision: bool, device: torch.device) -> bool:
    """Mixed precision is only meaningful on CUDA."""
    enabled = bool(use_mixed_precision) and device.type == "cuda"
    if use_mixed_precision and device.type != "cuda":
        logger.warning("USE_MIXED_PRECISION=True but device is CPU — AMP disabled.")
    return enabled


def get_grad_scaler(enabled: bool) -> Any:
    """
    Return a GradScaler when AMP is enabled.

    Uses ``torch.cuda.amp.GradScaler`` as required by the project brief.
    """
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()
