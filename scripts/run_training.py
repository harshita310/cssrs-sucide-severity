"""Run full MentalBERT-CSSR training (same pipeline as Notebook 04)."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import ensure_directories, get_logger, load_config, set_seed
from utils.dataset import (
    compute_balanced_class_weights,
    create_dataloader,
    stratified_split,
)
from utils.device import resolve_device
from utils.io import load_processed_dataset, training_columns
from utils.model import load_mentalbert_for_classification
from utils.runtime_settings import resolve_max_length
from utils.tokenization import load_mentalbert_tokenizer
from utils.trainer import MentalBERTTrainer, TrainerConfig

logger = get_logger("run_training")


def parse_args():
    parser = ArgumentParser(
        description="Train MentalBERT-CSSR using the configured pipeline."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path, relative to the project root.",
    )
    return parser.parse_args()


def _load_or_create_splits(
    df: pd.DataFrame,
    cfg,
    *,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reuse persisted CSV splits when available for reproducible metrics."""
    splits_dir = project_root / cfg.training.splits_dir
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    if bool(cfg.training.persist_splits) and train_path.exists() and val_path.exists():
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else None
        val_df = pd.read_csv(val_path)
        if test_df is not None:
            logger.info(
                "Loaded persisted splits | train=%s val=%s test=%s",
                len(train_df),
                len(val_df),
                len(test_df),
            )
            return train_df, val_df, test_df

    train_df, val_df, test_df = stratified_split(
        df,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        train_ratio=float(cfg.training.train_ratio),
        val_ratio=float(cfg.training.val_ratio),
        test_ratio=float(cfg.training.test_ratio),
        seed=cfg.SEED,
        stratify=bool(cfg.training.stratify),
    )

    if bool(cfg.training.persist_splits):
        splits_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.SEED)
    ensure_directories(cfg)
    device = resolve_device(cfg.DEVICE)
    max_length, max_length_source = resolve_max_length(
        cfg.MAX_LENGTH,
        recommendation_path=cfg.tokenization.recommendation_path,
        project_root=PROJECT_ROOT,
    )

    df = training_columns(
        load_processed_dataset(cfg.paths.processed_data),
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        ignore_columns=list(cfg.data.ignore_columns),
    )
    train_df, val_df, test_df = _load_or_create_splits(df, cfg, project_root=PROJECT_ROOT)

    tokenizer = load_mentalbert_tokenizer(
        model_name=cfg.MODEL_NAME,
        truncation_side=cfg.model.truncation_side,
    )
    pin_memory = bool(cfg.runtime.pin_memory) and device.type == "cuda"

    class_weights = compute_balanced_class_weights(
        train_df[cfg.data.label_column].astype(int).tolist(),
        num_labels=cfg.NUM_LABELS,
    )

    train_loader = create_dataloader(
        train_df,
        tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        max_length=max_length,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=int(cfg.runtime.num_workers),
        pin_memory=pin_memory,
        use_weighted_sampler=bool(cfg.training.use_weighted_sampler),
        num_labels=cfg.NUM_LABELS,
    )
    val_loader = create_dataloader(
        val_df,
        tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        max_length=max_length,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        pin_memory=pin_memory,
    )

    model = load_mentalbert_for_classification(
        model_name=cfg.MODEL_NAME,
        num_labels=cfg.NUM_LABELS,
        dropout=cfg.DROPOUT,
    )

    experiment_metadata = {
        "dataset_name": cfg.experiment.dataset_name,
        "experiment_name": cfg.experiment.name,
        "model_name": cfg.MODEL_NAME,
        "max_length": max_length,
        "max_length_source": max_length_source,
        "truncation_side": cfg.model.truncation_side,
        "num_labels": cfg.NUM_LABELS,
        "epochs": cfg.EPOCHS,
        "learning_rate": cfg.LEARNING_RATE,
        "batch_size": cfg.BATCH_SIZE,
        "dropout": cfg.DROPOUT,
        "weight_decay": cfg.WEIGHT_DECAY,
        "warmup_ratio": cfg.WARMUP_RATIO,
        "label_smoothing": cfg.LABEL_SMOOTHING,
        "loss_type": cfg.training.loss_type,
        "focal_gamma": cfg.training.focal_gamma,
        "use_class_weights": cfg.training.use_class_weights,
        "use_weighted_sampler": cfg.training.use_weighted_sampler,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "classifier_lr_mult": cfg.training.classifier_lr_mult,
        "freeze_encoder_epochs": cfg.training.freeze_encoder_epochs,
        "patience": cfg.PATIENCE,
        "max_grad_norm": cfg.training.max_grad_norm,
        "use_mixed_precision": cfg.USE_MIXED_PRECISION,
        "optimizer": cfg.training.optimizer,
        "scheduler": cfg.training.scheduler,
        "seed": cfg.SEED,
        "device": str(device),
        "monitor_metric": cfg.training.monitor_metric,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

    trainer_cfg = TrainerConfig(
        learning_rate=cfg.LEARNING_RATE,
        classifier_lr_mult=float(cfg.training.classifier_lr_mult),
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        label_smoothing=cfg.LABEL_SMOOTHING,
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        max_grad_norm=float(cfg.training.max_grad_norm),
        use_mixed_precision=bool(cfg.USE_MIXED_PRECISION),
        num_labels=cfg.NUM_LABELS,
        seed=cfg.SEED,
        monitor_metric=str(cfg.training.monitor_metric),
        loss_type=str(cfg.training.loss_type),
        focal_gamma=float(cfg.training.focal_gamma),
        use_class_weights=bool(cfg.training.use_class_weights),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        freeze_encoder_epochs=int(cfg.training.freeze_encoder_epochs),
        save_dir=Path(cfg.SAVE_PATH),
        results_dir=Path(cfg.RESULTS_PATH),
        plots_dir=Path(cfg.paths.plots_path),
        metrics_dir=Path(cfg.paths.metrics_path),
        model_path=Path(cfg.MODEL_PATH),
        optimizer_path=Path(cfg.SAVE_PATH) / "optimizer.pt",
        figure_dpi=int(cfg.eda.figure_dpi),
    )

    trainer = MentalBERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_cfg,
        label_list=list(range(cfg.NUM_LABELS)),
        experiment_metadata=experiment_metadata,
        class_weights=class_weights,
    )

    history = trainer.fit()
    monitor = str(cfg.training.monitor_metric)
    best_col = f"val_{monitor}" if monitor in history.columns else "val_macro_f1"
    best = history.loc[history[best_col].idxmax()]
    logger.info(
        "Done | best val_%s=%.4f val_acc=%.4f epoch=%s",
        monitor,
        float(best[best_col]),
        float(best["val_accuracy"]),
        int(best["epoch"]),
    )
    print(history[["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
