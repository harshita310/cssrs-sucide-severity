"""
Training loop for MentalBERT-CSSR.

Features: AdamW, cosine schedule with warmup, label smoothing, AMP
(``torch.cuda.amp``), gradient clipping, early stopping, checkpointing,
and full metric history export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from utils.device import amp_enabled, get_grad_scaler
from utils.logging_utils import get_logger
from utils.metrics import (
    compute_classification_metrics,
    metrics_to_row,
    plot_confusion_matrix,
    plot_history_curves,
)
from utils.model import save_checkpoint

logger = get_logger(__name__)


@dataclass
class EarlyStopping:
    """Stop when the monitored validation metric stops improving."""

    patience: int = 3
    mode: str = "max"  # "max" for F1/accuracy, "min" for loss
    min_delta: float = 0.0
    best_score: float | None = None
    bad_epochs: int = 0
    should_stop: bool = False

    def step(self, score: float) -> bool:
        """Return True if this step improved the best score."""
        if self.best_score is None:
            self.best_score = score
            self.bad_epochs = 0
            return True

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-5
    classifier_lr_mult: float = 10.0  # head LR = encoder LR * mult
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.0
    epochs: int = 20
    patience: int = 5
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    num_labels: int = 7
    seed: int = 42
    monitor_metric: str = "macro_f1"
    loss_type: str = "focal"  # "focal" | "ce"
    focal_gamma: float = 2.0
    ordinal_distance_weight: float = 0.0
    composite_metric_weights: dict[str, float] = field(default_factory=dict)
    use_class_weights: bool = True
    gradient_accumulation_steps: int = 2
    freeze_encoder_epochs: int = 0  # optional: train head-only for N epochs
    save_dir: Path = field(default_factory=lambda: Path("saved_model"))
    results_dir: Path = field(default_factory=lambda: Path("RESULTS"))
    plots_dir: Path = field(default_factory=lambda: Path("RESULTS/plots"))
    metrics_dir: Path = field(default_factory=lambda: Path("RESULTS/metrics"))
    model_path: Path = field(default_factory=lambda: Path("saved_model/best_model.pt"))
    optimizer_path: Path = field(default_factory=lambda: Path("saved_model/optimizer.pt"))
    figure_dpi: int = 150


_NO_DECAY_SUFFIXES = ("bias", "LayerNorm.weight", "layer_norm.weight")


def compute_composite_score(
    metrics: dict[str, Any],
    weights: dict[str, float],
) -> float:
    """Weighted sum of named metrics for balanced checkpoint selection."""
    return float(
        sum(float(metrics[name]) * float(weight) for name, weight in weights.items())
    )


def _is_head_param(name: str) -> bool:
    return name.startswith("classifier") or ".classifier." in name


def _should_decay(name: str) -> bool:
    return not any(name.endswith(suffix) for suffix in _NO_DECAY_SUFFIXES)


def _build_adamw_param_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    classifier_lr_mult: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """
    HuggingFace-style AdamW groups: no weight decay on bias / LayerNorm;
    discriminative LR for the classification head.
    """
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = learning_rate * (
            classifier_lr_mult if _is_head_param(name) else 1.0
        )
        wd = weight_decay if _should_decay(name) else 0.0
        key = (lr, wd)
        groups.setdefault(key, []).append(param)

    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in groups.items()
    ]


class MentalBERTTrainer:
    """
    End-to-end fine-tuning trainer for MentalBERT severity classification.

    Performance-oriented defaults for this imbalanced 7-class corpus:
    focal + class weights, discriminative LRs, gradient accumulation,
    weighted sampling (via DataLoader), and early stopping on macro-F1.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: TrainerConfig,
        label_list: list[int] | None = None,
        experiment_metadata: dict[str, Any] | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_list = label_list or list(range(config.num_labels))
        self.experiment_metadata = experiment_metadata or {}

        weights = None
        if config.use_class_weights and class_weights is not None:
            weights = class_weights.to(device)

        from utils.losses import build_criterion

        self.criterion = build_criterion(
            loss_type=config.loss_type,
            class_weights=weights,
            label_smoothing=config.label_smoothing,
            focal_gamma=config.focal_gamma,
            ordinal_distance_weight=config.ordinal_distance_weight,
            num_labels=config.num_labels,
        )
        # Move criterion buffers (class weights) to device
        self.criterion = self.criterion.to(device)

        param_groups = _build_adamw_param_groups(
            self.model,
            learning_rate=config.learning_rate,
            classifier_lr_mult=config.classifier_lr_mult,
            weight_decay=config.weight_decay,
        )
        self.optimizer = AdamW(param_groups)

        accum = max(1, int(config.gradient_accumulation_steps))
        steps_per_epoch = max(1, len(train_loader) // accum)
        total_steps = max(1, steps_per_epoch * config.epochs)
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.accum_steps = accum

        self.use_amp = amp_enabled(config.use_mixed_precision, device)
        self.scaler = get_grad_scaler(self.use_amp)
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode="max",
            min_delta=1e-4,
        )
        self.history: list[dict[str, Any]] = []

        config.save_dir.mkdir(parents=True, exist_ok=True)
        config.plots_dir.mkdir(parents=True, exist_ok=True)
        config.metrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trainer ready | device=%s amp=%s loss=%s gamma=%s "
            "encoder_lr=%.2e head_lr=%.2e accum=%s steps=%s warmup=%s monitor=%s",
            device,
            self.use_amp,
            config.loss_type,
            config.focal_gamma,
            config.learning_rate,
            config.learning_rate * config.classifier_lr_mult,
            self.accum_steps,
            total_steps,
            warmup_steps,
            config.monitor_metric,
        )

    def _set_encoder_trainable(self, trainable: bool) -> None:
        for name, param in self.model.named_parameters():
            if name.startswith("classifier") or ".classifier." in name:
                param.requires_grad = True
            else:
                param.requires_grad = trainable

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _forward_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        losses: list[float] = []
        all_true: list[int] = []
        all_pred: list[int] = []
        all_prob: list[list[float]] = []

        for batch in loader:
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss, logits, labels = self._forward_loss(batch)
            else:
                loss, logits, labels = self._forward_loss(batch)

            losses.append(float(loss.item()))
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())
            all_prob.extend(probs.detach().cpu().tolist())

        y_prob = np.asarray(all_prob, dtype=float) if all_prob else None
        metrics = compute_classification_metrics(
            all_true,
            all_pred,
            labels=self.label_list,
            target_names=[str(x) for x in self.label_list],
            y_prob=y_prob,
        )
        metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
        metrics["y_true"] = all_true
        metrics["y_pred"] = all_pred
        if y_prob is not None:
            metrics["y_prob"] = y_prob.tolist()
        return metrics

    def _train_epoch(self) -> dict[str, Any]:
        self.model.train()
        running_loss = 0.0
        all_true: list[int] = []
        all_pred: list[int] = []
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(self.train_loader, desc="train", leave=False)
        for step, batch in enumerate(progress, start=1):
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss, logits, labels = self._forward_loss(batch)
                    loss = loss / self.accum_steps
                self.scaler.scale(loss).backward()
            else:
                loss, logits, labels = self._forward_loss(batch)
                loss = loss / self.accum_steps
                loss.backward()

            if step % self.accum_steps == 0 or step == len(self.train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * self.accum_steps
            preds = torch.argmax(logits.detach(), dim=-1)
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())
            progress.set_postfix(loss=float(loss.item()) * self.accum_steps, lr=self._current_lr())

        metrics = compute_classification_metrics(
            all_true,
            all_pred,
            labels=self.label_list,
            target_names=[str(x) for x in self.label_list],
        )
        metrics["loss"] = running_loss / max(1, len(self.train_loader))
        return metrics

    def fit(self) -> pd.DataFrame:
        """
        Run the full training loop.

        Saves ``best_model.pt``, ``optimizer.pt``, ``history.csv``,
        ``training_metrics.csv``, curve plots, and confusion matrix for the
        best validation epoch predictions refreshed at the end.
        """
        best_monitor = -float("inf")
        best_epoch = -1
        best_val_metrics: dict[str, Any] | None = None

        for epoch in range(1, self.config.epochs + 1):
            logger.info("===== Epoch %s / %s =====", epoch, self.config.epochs)

            # Optional head-only warm-start
            if self.config.freeze_encoder_epochs > 0:
                if epoch <= self.config.freeze_encoder_epochs:
                    self._set_encoder_trainable(False)
                    logger.info("Encoder frozen (head-only) for epoch %s", epoch)
                elif epoch == self.config.freeze_encoder_epochs + 1:
                    self._set_encoder_trainable(True)
                    logger.info("Encoder unfrozen from epoch %s", epoch)

            train_metrics = self._train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            if self.config.composite_metric_weights:
                train_metrics["composite_score"] = compute_composite_score(
                    train_metrics,
                    self.config.composite_metric_weights,
                )
                val_metrics["composite_score"] = compute_composite_score(
                    val_metrics,
                    self.config.composite_metric_weights,
                )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "learning_rate": self._current_lr(),
            }
            row.update(metrics_to_row(train_metrics, prefix="train_"))
            row.update(metrics_to_row(val_metrics, prefix="val_"))
            # Canonical aliases expected by plot_history_curves / project brief
            row["train_accuracy"] = train_metrics["accuracy"]
            row["val_accuracy"] = val_metrics["accuracy"]
            row["train_macro_f1"] = train_metrics["macro_f1"]
            row["val_macro_f1"] = val_metrics["macro_f1"]
            row["train_micro_f1"] = train_metrics["micro_f1"]
            row["val_micro_f1"] = val_metrics["micro_f1"]
            row["train_weighted_f1"] = train_metrics["weighted_f1"]
            row["val_weighted_f1"] = val_metrics["weighted_f1"]
            if "composite_score" in train_metrics:
                row["train_composite_score"] = train_metrics["composite_score"]
                row["val_composite_score"] = val_metrics["composite_score"]

            self.history.append(row)

            monitor_value = float(val_metrics[self.config.monitor_metric])
            logger.info(
                "Epoch %s | train_loss=%.4f val_loss=%.4f val_acc=%.4f "
                "val_macro_f1=%.4f lr=%.2e",
                epoch,
                row["train_loss"],
                row["val_loss"],
                row["val_accuracy"],
                row["val_macro_f1"],
                row["learning_rate"],
            )

            improved = self.early_stopping.step(monitor_value)
            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                best_val_metrics = val_metrics
                save_checkpoint(
                    self.config.model_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_metric=best_monitor,
                    history=self.history,
                    config_snapshot=self.experiment_metadata,
                )
                torch.save(
                    {
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_monitor,
                    },
                    self.config.optimizer_path,
                )
                logger.info(
                    "New best %s=%.4f — saved %s and %s",
                    self.config.monitor_metric,
                    best_monitor,
                    self.config.model_path,
                    self.config.optimizer_path,
                )

            if self.early_stopping.should_stop:
                logger.info(
                    "Early stopping at epoch %s (patience=%s). Best epoch=%s %s=%.4f",
                    epoch,
                    self.config.patience,
                    best_epoch,
                    self.config.monitor_metric,
                    best_monitor,
                )
                break

        history_df = pd.DataFrame(self.history)
        history_path = self.config.metrics_dir / "history.csv"
        history_df.to_csv(history_path, index=False)

        # Final validation metrics table (last epoch + best snapshot summary)
        training_metrics_rows = []
        for r in self.history:
            training_metrics_rows.append(r)
        training_metrics_df = pd.DataFrame(training_metrics_rows)
        training_metrics_path = self.config.metrics_dir / "training_metrics.csv"
        training_metrics_df.to_csv(training_metrics_path, index=False)

        plot_history_curves(history_df, self.config.plots_dir, dpi=self.config.figure_dpi)

        # Reload best weights for canonical val confusion matrix
        if self.config.model_path.exists():
            checkpoint = torch.load(
                self.config.model_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            best_val_metrics = self.evaluate(self.val_loader)

        if best_val_metrics is not None:
            cm_path = self.config.plots_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                best_val_metrics["confusion_matrix"],
                labels=self.label_list,
                output_path=cm_path,
                title=f"Validation Confusion Matrix (best {self.config.monitor_metric})",
                dpi=self.config.figure_dpi,
            )
            report_df = pd.DataFrame(best_val_metrics["classification_report"]).T
            report_df.to_csv(self.config.metrics_dir / "val_classification_report.csv")
            with (self.config.metrics_dir / "val_best_metrics.json").open(
                "w", encoding="utf-8"
            ) as fh:
                payload = {
                    k: v
                    for k, v in best_val_metrics.items()
                    if k not in {"y_true", "y_pred", "y_prob"}
                }
                payload["best_epoch"] = best_epoch
                payload["monitor_metric"] = self.config.monitor_metric
                payload["best_monitor_value"] = best_monitor
                json.dump(payload, fh, indent=2)

        self._write_experiment_log(
            best_epoch=best_epoch,
            best_monitor=best_monitor,
            best_val_metrics=best_val_metrics,
            history_path=history_path,
            training_metrics_path=training_metrics_path,
        )

        logger.info("Training complete. History → %s", history_path)
        return history_df

    def _write_experiment_log(
        self,
        *,
        best_epoch: int,
        best_monitor: float,
        best_val_metrics: dict[str, Any] | None,
        history_path: Path,
        training_metrics_path: Path,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = {
            "timestamp_utc": timestamp,
            "stage": "training",
            "notebook": "04_Training.ipynb",
            **self.experiment_metadata,
            "best_epoch": best_epoch,
            "monitor_metric": self.config.monitor_metric,
            "best_monitor_value": best_monitor,
            "metrics": {
                k: best_val_metrics.get(k)
                for k in [
                    "loss",
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                    "weighted_precision",
                    "weighted_recall",
                    "weighted_f1",
                ]
            }
            if best_val_metrics
            else {},
            "artifacts": {
                "best_model": str(self.config.model_path),
                "optimizer": str(self.config.optimizer_path),
                "history_csv": str(history_path),
                "training_metrics_csv": str(training_metrics_path),
            },
        }
        log_path = self.config.metrics_dir / "experiment_log.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        with (self.config.metrics_dir / "training_experiment.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
