"""
Loss functions for imbalanced C-SSRS severity classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropyLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017) with optional class weights.

    ``FL = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Helps rare high-severity classes by down-weighting easy majority examples.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight.detach().float())
        else:
            self.weight = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smoothed log-probs path via CE; then apply focal factor on hard examples
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        with torch.no_grad():
            pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Cross entropy plus an ordinal distance penalty.

    The labels stay as 7 separate classes. The extra term encourages the
    probability mass to stay close to the true severity level, so predicting
    class 5 for true class 6 is penalized less than predicting class 0.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        distance_weight: float = 0.2,
        num_labels: int = 7,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.distance_weight = float(distance_weight)
        self.num_labels = int(num_labels)
        if weight is not None:
            self.register_buffer("weight", weight.detach().float())
        else:
            self.weight = None  # type: ignore[assignment]
        self.register_buffer(
            "class_values",
            torch.arange(self.num_labels, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
        probs = torch.softmax(logits, dim=-1)
        expected = torch.sum(probs * self.class_values.to(logits.device), dim=-1)
        ordinal_penalty = F.smooth_l1_loss(expected, targets.float())
        return ce + self.distance_weight * ordinal_penalty


def build_criterion(
    *,
    loss_type: str = "focal",
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    ordinal_distance_weight: float = 0.0,
    num_labels: int = 7,
) -> nn.Module:
    """Factory for training criterion."""
    loss_type = loss_type.lower().strip()
    if loss_type == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if loss_type in {"ordinal_ce", "ce_ordinal"}:
        return OrdinalCrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            distance_weight=ordinal_distance_weight,
            num_labels=num_labels,
        )
    if loss_type == "focal":
        return FocalCrossEntropyLoss(
            weight=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    raise ValueError(
        f"Unknown loss_type={loss_type!r}; use 'ce', 'ordinal_ce', or 'focal'."
    )
