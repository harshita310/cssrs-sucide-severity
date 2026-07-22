"""
Classification metrics for MentalBERT-CSSR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _within_k_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
) -> float:
    """Fraction of predictions within ``k`` ordinal severity levels of truth."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred) <= k))


def _high_severity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: int = 4,
) -> dict[str, float]:
    """Binary collapse: severity >= threshold vs below (clinical high-risk band)."""
    true_bin = (y_true >= threshold).astype(int)
    pred_bin = (y_pred >= threshold).astype(int)
    tp = int(np.sum((true_bin == 1) & (pred_bin == 1)))
    fp = int(np.sum((true_bin == 0) & (pred_bin == 1)))
    fn = int(np.sum((true_bin == 1) & (pred_bin == 0)))
    tn = int(np.sum((true_bin == 0) & (pred_bin == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "high_severity_threshold": float(threshold),
        "high_severity_precision": float(precision),
        "high_severity_recall": float(recall),
        "high_severity_f1": float(f1),
        "high_severity_support": float(int(true_bin.sum())),
    }


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[int] | None = None,
    target_names: Sequence[str] | None = None,
    y_prob: np.ndarray | None = None,
    high_severity_threshold: int = 4,
) -> dict[str, Any]:
    """
    Compute accuracy + macro / micro / weighted precision, recall, F1.

    When ``y_prob`` is supplied (n_samples, n_classes), also computes log-loss,
    macro OvR AUC, and top-2 / top-3 accuracy.

    Ordinal severity extras: balanced accuracy, Cohen's kappa, quadratic weighted
    kappa, Matthews correlation, MAE, within-1/2 accuracy, high-severity F1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_list = [int(x) for x in labels]

    acc = float(accuracy_score(y_true, y_pred))

    def _prf(average: str) -> tuple[float, float, float]:
        p, r, f, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            zero_division=0,
        )
        return float(p), float(r), float(f)

    macro_p, macro_r, macro_f1 = _prf("macro")
    micro_p, micro_r, micro_f1 = _prf("micro")
    weighted_p, weighted_r, weighted_f1 = _prf("weighted")

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    cohen_kappa = float(cohen_kappa_score(y_true, y_pred, labels=label_list))
    qwk = float(
        cohen_kappa_score(y_true, y_pred, labels=label_list, weights="quadratic")
    )
    mcc = float(matthews_corrcoef(y_true, y_pred, sample_weight=None))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    within_1_acc = _within_k_accuracy(y_true, y_pred, 1)
    within_2_acc = _within_k_accuracy(y_true, y_pred, 2)

    result: dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": cohen_kappa,
        "quadratic_weighted_kappa": qwk,
        "matthews_corrcoef": mcc,
        "mean_absolute_error": mae,
        "within_1_accuracy": within_1_acc,
        "within_2_accuracy": within_2_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_class": {
            str(int(lbl)): {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": label_list,
    }
    result.update(
        _high_severity_metrics(
            y_true, y_pred, threshold=int(high_severity_threshold)
        )
    )

    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=float)
        n_classes = len(label_list)
        if prob.ndim == 2 and prob.shape[1] >= n_classes and len(y_true) > 0:
            prob = prob[:, :n_classes]
            prob = np.clip(prob, 0.0, 1.0)
            row_sums = prob.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0.0] = 1.0
            prob = prob / row_sums
            try:
                present = sorted(set(y_true.tolist()))
                result["log_loss"] = float(
                    log_loss(y_true, prob, labels=label_list)
                )
                if len(present) >= 2:
                    result["macro_ovr_auc"] = float(
                        roc_auc_score(
                            y_true,
                            prob,
                            multi_class="ovr",
                            average="macro",
                            labels=label_list,
                        )
                    )
                    result["weighted_ovr_auc"] = float(
                        roc_auc_score(
                            y_true,
                            prob,
                            multi_class="ovr",
                            average="weighted",
                            labels=label_list,
                        )
                    )
                if prob.shape[1] >= 2:
                    result["top2_accuracy"] = float(
                        top_k_accuracy_score(
                            y_true,
                            prob,
                            k=min(2, n_classes),
                            labels=label_list,
                        )
                    )
                if prob.shape[1] >= 3:
                    result["top3_accuracy"] = float(
                        top_k_accuracy_score(
                            y_true,
                            prob,
                            k=min(3, n_classes),
                            labels=label_list,
                        )
                    )
            except ValueError as err:
                logger.warning("Probability metrics skipped: %s", err)

    return result


HEADLINE_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "quadratic_weighted_kappa",
    "matthews_corrcoef",
    "mean_absolute_error",
    "within_1_accuracy",
    "within_2_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "high_severity_f1",
    "log_loss",
    "macro_ovr_auc",
    "top2_accuracy",
    "top3_accuracy",
]


def metrics_to_row(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten headline metrics for CSV history rows."""
    row: dict[str, float] = {}
    for key in HEADLINE_METRIC_KEYS:
        if key in metrics and metrics[key] is not None:
            try:
                row[f"{prefix}{key}" if prefix else key] = float(metrics[key])
            except (TypeError, ValueError):
                pass
    return row


def metrics_summary_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Single-column summary of all headline evaluation metrics."""
    rows = []
    for key in HEADLINE_METRIC_KEYS:
        if key in metrics:
            rows.append({"metric": key, "value": metrics[key]})
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    cm: np.ndarray | list[list[int]],
    labels: Sequence[int],
    output_path: Path,
    *,
    title: str = "Confusion Matrix",
    dpi: int = 150,
    normalize: bool = False,
) -> Path:
    """Save a confusion-matrix heatmap."""
    if normalize:
        matrix = np.asarray(cm, dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums
        fmt = ".2f"
    else:
        matrix = np.asarray(cm, dtype=int)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted severity")
    ax.set_ylabel("True severity")
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", output_path)
    return output_path


def plot_history_curves(
    history_df: pd.DataFrame,
    plots_dir: Path,
    *,
    dpi: int = 150,
) -> dict[str, Path]:
    """
    Write the required training curves:

    training_loss, validation_loss, training_accuracy, validation_accuracy,
    macro_f1, micro_f1, weighted_f1, learning_rate.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    curve_specs = [
        ("train_loss", "training_loss.png", "Training Loss", "Loss"),
        ("val_loss", "validation_loss.png", "Validation Loss", "Loss"),
        ("train_accuracy", "training_accuracy.png", "Training Accuracy", "Accuracy"),
        ("val_accuracy", "validation_accuracy.png", "Validation Accuracy", "Accuracy"),
        ("val_macro_f1", "macro_f1.png", "Validation Macro F1", "Macro F1"),
        ("val_micro_f1", "micro_f1.png", "Validation Micro F1", "Micro F1"),
        ("val_weighted_f1", "weighted_f1.png", "Validation Weighted F1", "Weighted F1"),
        ("learning_rate", "learning_rate.png", "Learning Rate Schedule", "LR"),
    ]

    for column, filename, title, ylabel in curve_specs:
        if column not in history_df.columns:
            logger.warning("History missing column %s — skip plot %s", column, filename)
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history_df["epoch"], history_df[column], marker="o", color="#2c5f7c")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = plots_dir / filename
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved[column] = out
        logger.info("Saved plot → %s", out)

    return saved


def classification_report_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """Convert sklearn dict report to a tidy DataFrame."""
    rows = []
    for key, value in report.items():
        if isinstance(value, dict):
            row = {"label": key}
            row.update(value)
            rows.append(row)
        else:
            rows.append({"label": key, "score": value})
    return pd.DataFrame(rows)
