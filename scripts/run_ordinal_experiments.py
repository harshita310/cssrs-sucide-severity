"""Run isolated CUDA-only ordinal MentalBERT-CSSR experiments."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import ensure_directories, get_logger, load_config, set_seed
from utils.dataset import compute_balanced_class_weights, create_dataloader
from utils.device import resolve_device
from utils.io import load_processed_dataset, training_columns
from utils.model import load_mentalbert_for_classification
from utils.runtime_settings import resolve_max_length
from utils.tokenization import load_mentalbert_tokenizer
from utils.trainer import MentalBERTTrainer, TrainerConfig

logger = get_logger("run_ordinal_experiments")


def build_run_paths(project_root: Path, run_name: str) -> dict[str, Path]:
    """Return isolated artifact paths for one ordinal experiment run."""
    save_dir = project_root / "saved_model" / "ordinal_experiments"
    metrics_dir = (
        project_root / "RESULTS" / "metrics" / "ordinal_experiments" / run_name
    )
    plots_dir = project_root / "RESULTS" / "plots" / "ordinal_experiments" / run_name
    return {
        "save_dir": save_dir,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "model_path": save_dir / f"best_model_{run_name}.pt",
        "optimizer_path": save_dir / f"optimizer_{run_name}.pt",
    }


def require_cuda(device: torch.device) -> None:
    """Fail fast when the selected runtime cannot use CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-only experiment requested, but PyTorch cannot access CUDA."
        )


def _load_splits(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir = PROJECT_ROOT / cfg.training.splits_dir
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"
    if train_path.exists() and val_path.exists() and test_path.exists():
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    from utils.dataset import stratified_split

    df = training_columns(
        load_processed_dataset(cfg.paths.processed_data),
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        ignore_columns=list(cfg.data.ignore_columns),
    )
    return stratified_split(
        df,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        train_ratio=float(cfg.training.train_ratio),
        val_ratio=float(cfg.training.val_ratio),
        test_ratio=float(cfg.training.test_ratio),
        seed=cfg.SEED,
        stratify=bool(cfg.training.stratify),
    )


def main() -> None:
    cfg = load_config()
    set_seed(cfg.SEED)
    ensure_directories(cfg)
    device = resolve_device("cuda")
    require_cuda(device)
    logger.info("CUDA ordinal experiments on %s", torch.cuda.get_device_name(0))

    max_length, max_length_source = resolve_max_length(
        cfg.MAX_LENGTH,
        recommendation_path=cfg.tokenization.recommendation_path,
        project_root=PROJECT_ROOT,
    )
    train_df, val_df, test_df = _load_splits(cfg)
    tokenizer = load_mentalbert_tokenizer(
        model_name=cfg.MODEL_NAME,
        truncation_side=cfg.model.truncation_side,
    )
    pin_memory = bool(cfg.runtime.pin_memory)
    class_weights = compute_balanced_class_weights(
        train_df[cfg.data.label_column].astype(int).tolist(),
        num_labels=cfg.NUM_LABELS,
    )

    runs = [
        {
            "run_name": "ordinal_ce_dw_0_10_do_0_15",
            "ordinal_distance_weight": 0.10,
            "learning_rate": 1.5e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
            "label_smoothing": 0.0,
            "freeze_encoder_epochs": 0,
        },
        {
            "run_name": "ordinal_ce_dw_0_20_do_0_20",
            "ordinal_distance_weight": 0.20,
            "learning_rate": 1.2e-5,
            "dropout": 0.20,
            "weight_decay": 0.02,
            "label_smoothing": 0.03,
            "freeze_encoder_epochs": 0,
        },
        {
            "run_name": "ordinal_ce_dw_0_30_do_0_25",
            "ordinal_distance_weight": 0.30,
            "learning_rate": 1.2e-5,
            "dropout": 0.25,
            "weight_decay": 0.02,
            "label_smoothing": 0.05,
            "freeze_encoder_epochs": 1,
        },
    ]

    summary_rows = []
    for run in runs:
        run_name = run["run_name"]
        paths = build_run_paths(PROJECT_ROOT, run_name)
        paths["save_dir"].mkdir(parents=True, exist_ok=True)
        paths["metrics_dir"].mkdir(parents=True, exist_ok=True)
        paths["plots_dir"].mkdir(parents=True, exist_ok=True)

        logger.info("Starting ordinal run: %s", run_name)
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
            use_weighted_sampler=False,
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
            dropout=run["dropout"],
        )
        trainer_cfg = TrainerConfig(
            learning_rate=run["learning_rate"],
            classifier_lr_mult=float(cfg.training.classifier_lr_mult),
            weight_decay=run["weight_decay"],
            warmup_ratio=float(cfg.WARMUP_RATIO),
            label_smoothing=run["label_smoothing"],
            epochs=int(cfg.EPOCHS),
            patience=int(cfg.PATIENCE),
            max_grad_norm=float(cfg.training.max_grad_norm),
            use_mixed_precision=bool(cfg.USE_MIXED_PRECISION),
            num_labels=cfg.NUM_LABELS,
            seed=cfg.SEED,
            monitor_metric="weighted_f1",
            loss_type="ordinal_ce",
            focal_gamma=float(cfg.training.focal_gamma),
            ordinal_distance_weight=run["ordinal_distance_weight"],
            use_class_weights=False,
            gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
            freeze_encoder_epochs=int(run["freeze_encoder_epochs"]),
            save_dir=paths["save_dir"],
            results_dir=PROJECT_ROOT / "RESULTS" / "ordinal_experiments",
            plots_dir=paths["plots_dir"],
            metrics_dir=paths["metrics_dir"],
            model_path=paths["model_path"],
            optimizer_path=paths["optimizer_path"],
            figure_dpi=int(cfg.eda.figure_dpi),
        )
        metadata = {
            **run,
            "experiment_name": run_name,
            "model_name": cfg.MODEL_NAME,
            "max_length": max_length,
            "max_length_source": max_length_source,
            "num_labels": cfg.NUM_LABELS,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "device": str(device),
        }
        trainer = MentalBERTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=trainer_cfg,
            label_list=list(range(cfg.NUM_LABELS)),
            experiment_metadata=metadata,
            class_weights=class_weights,
        )
        history = trainer.fit()
        best = history.loc[history["val_weighted_f1"].idxmax()]
        row = {
            **run,
            "best_epoch": int(best["epoch"]),
            "val_accuracy": float(best["val_accuracy"]),
            "val_macro_f1": float(best["val_macro_f1"]),
            "val_weighted_f1": float(best["val_weighted_f1"]),
            "model_path": str(paths["model_path"]),
        }
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            PROJECT_ROOT
            / "RESULTS"
            / "metrics"
            / "ordinal_experiments"
            / "ordinal_experiment_summary.csv",
            index=False,
        )

        del trainer, model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows).sort_values("val_weighted_f1", ascending=False)
    summary_path = (
        PROJECT_ROOT
        / "RESULTS"
        / "metrics"
        / "ordinal_experiments"
        / "ordinal_experiment_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    with summary_path.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(orient="records"), fh, indent=2)

    print("\n=== Ordinal Experiment Summary ===\n")
    print(summary.to_string(index=False))
    print(f"\nArtifacts -> {summary_path.parent}")


if __name__ == "__main__":
    main()
