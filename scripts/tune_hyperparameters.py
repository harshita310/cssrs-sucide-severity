"""Hyperparameter tuning sweep for MentalBERT-CSSR."""

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

logger = get_logger("tune_hyperparameters")


def main() -> None:
    cfg = load_config()
    set_seed(cfg.SEED)
    ensure_directories(cfg)
    device = resolve_device(cfg.DEVICE)
    
    max_length, max_length_source = resolve_max_length(
        cfg.MAX_LENGTH,
        recommendation_path=cfg.tokenization.recommendation_path,
        project_root=PROJECT_ROOT,
    )

    # Load and split dataset
    df = training_columns(
        load_processed_dataset(PROJECT_ROOT / cfg.paths.processed_data),
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        ignore_columns=list(cfg.data.ignore_columns),
    )
    train_df, val_df, _ = stratified_split(
        df,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        train_ratio=float(cfg.training.train_ratio),
        val_ratio=float(cfg.training.val_ratio),
        test_ratio=float(cfg.training.test_ratio),
        seed=cfg.SEED,
        stratify=bool(cfg.training.stratify),
    )

    tokenizer = load_mentalbert_tokenizer(
        model_name=cfg.MODEL_NAME,
        truncation_side=cfg.model.truncation_side,
    )
    pin_memory = bool(cfg.runtime.pin_memory) and device.type == "cuda"

    # Define hyperparameter configurations to test
    runs = [
        # Run 1: Baseline (Focal, Class Weights, Weighted Sampler)
        {
            "run_name": "Run_1_Focal_Both_Weights",
            "loss_type": "focal",
            "focal_gamma": 2.0,
            "use_class_weights": True,
            "use_weighted_sampler": True,
            "learning_rate": 1.5e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
        },
        # Run 2: CE + Sampler (No class weights)
        {
            "run_name": "Run_2_CE_Sampler_Only",
            "loss_type": "ce",
            "focal_gamma": 2.0,
            "use_class_weights": False,
            "use_weighted_sampler": True,
            "learning_rate": 1.5e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
        },
        # Run 3: CE + Class Weights (No sampler)
        {
            "run_name": "Run_3_CE_Weights_Only",
            "loss_type": "ce",
            "focal_gamma": 2.0,
            "use_class_weights": True,
            "use_weighted_sampler": False,
            "learning_rate": 1.5e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
        },
        # Run 4: Vanilla CE (No sampler, no class weights)
        {
            "run_name": "Run_4_Vanilla_CE",
            "loss_type": "ce",
            "focal_gamma": 2.0,
            "use_class_weights": False,
            "use_weighted_sampler": False,
            "learning_rate": 1.5e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
        },
        # Run 5: Focal + Class Weights (No sampler, gamma=1.5)
        {
            "run_name": "Run_5_Focal_1.5_Weights_Only",
            "loss_type": "focal",
            "focal_gamma": 1.5,
            "use_class_weights": True,
            "use_weighted_sampler": False,
            "learning_rate": 1.5e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
        },
        # Run 6: CE + Both (Sampler + Class Weights)
        {
            "run_name": "Run_6_CE_Both_Weights",
            "loss_type": "ce",
            "focal_gamma": 2.0,
            "use_class_weights": True,
            "use_weighted_sampler": True,
            "learning_rate": 1.5e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
        },
        # Run 7: CE + Regularization (Sampler + Class Weights, more dropout & weight decay)
        {
            "run_name": "Run_7_CE_Regularized",
            "loss_type": "ce",
            "focal_gamma": 2.0,
            "use_class_weights": True,
            "use_weighted_sampler": True,
            "learning_rate": 1.5e-5,
            "dropout": 0.20,
            "weight_decay": 0.02,
        },
        # Run 8: CE + High LR (Sampler + Class Weights, LR=2e-5)
        {
            "run_name": "Run_8_CE_High_LR",
            "loss_type": "ce",
            "focal_gamma": 2.0,
            "use_class_weights": True,
            "use_weighted_sampler": True,
            "learning_rate": 2e-5,
            "dropout": 0.15,
            "weight_decay": 0.01,
        },
    ]

    results = []
    tuning_output_csv = PROJECT_ROOT / "RESULTS/metrics/hyperparameter_tuning_results.csv"
    tuning_output_csv.parent.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(runs, start=1):
        logger.info("==================================================")
        logger.info("STARTING CONFIG SWEEP %d/%d: %s", i, len(runs), r["run_name"])
        logger.info("==================================================")

        # Set paths for this run to avoid overwriting main artifacts
        tune_save_dir = PROJECT_ROOT / "saved_model/tune"
        tune_save_dir.mkdir(parents=True, exist_ok=True)
        
        trainer_cfg = TrainerConfig(
            learning_rate=r["learning_rate"],
            classifier_lr_mult=float(cfg.training.classifier_lr_mult),
            weight_decay=r["weight_decay"],
            warmup_ratio=cfg.WARMUP_RATIO,
            label_smoothing=cfg.LABEL_SMOOTHING,
            epochs=cfg.EPOCHS,
            patience=cfg.PATIENCE,
            max_grad_norm=float(cfg.training.max_grad_norm),
            use_mixed_precision=bool(cfg.USE_MIXED_PRECISION),
            num_labels=cfg.NUM_LABELS,
            seed=cfg.SEED,
            monitor_metric=str(cfg.training.monitor_metric),
            loss_type=r["loss_type"],
            focal_gamma=r["focal_gamma"],
            use_class_weights=r["use_class_weights"],
            gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
            freeze_encoder_epochs=int(cfg.training.freeze_encoder_epochs),
            save_dir=tune_save_dir,
            results_dir=PROJECT_ROOT / "RESULTS/tune",
            plots_dir=PROJECT_ROOT / f"RESULTS/plots/tune/{r['run_name']}",
            metrics_dir=PROJECT_ROOT / f"RESULTS/metrics/tune/{r['run_name']}",
            model_path=tune_save_dir / f"best_model_{r['run_name']}.pt",
            optimizer_path=tune_save_dir / f"optimizer_{r['run_name']}.pt",
            figure_dpi=int(cfg.eda.figure_dpi),
        )

        # Create dataloaders for this specific run (weighted sampler option)
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
            use_weighted_sampler=r["use_weighted_sampler"],
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

        class_weights = compute_balanced_class_weights(
            train_df[cfg.data.label_column].astype(int).tolist(),
            num_labels=cfg.NUM_LABELS,
        )

        # Load fresh model
        model = load_mentalbert_for_classification(
            model_name=cfg.MODEL_NAME,
            num_labels=cfg.NUM_LABELS,
            dropout=r["dropout"],
        )

        trainer = MentalBERTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=trainer_cfg,
            label_list=list(range(cfg.NUM_LABELS)),
            experiment_metadata=r,
            class_weights=class_weights,
        )

        history = trainer.fit()
        
        # Load best metrics from output JSON
        best_metrics_json = trainer_cfg.metrics_dir / "val_best_metrics.json"
        if best_metrics_json.exists():
            with open(best_metrics_json, "r", encoding="utf-8") as f:
                best_val = json.load(f)
            
            run_result = {
                "run_name": r["run_name"],
                "loss_type": r["loss_type"],
                "focal_gamma": r["focal_gamma"],
                "use_class_weights": r["use_class_weights"],
                "use_weighted_sampler": r["use_weighted_sampler"],
                "learning_rate": r["learning_rate"],
                "dropout": r["dropout"],
                "weight_decay": r["weight_decay"],
                "best_epoch": best_val["best_epoch"],
                "val_loss": best_val["loss"],
                "val_accuracy": best_val["accuracy"],
                "val_macro_f1": best_val["macro_f1"],
                "val_weighted_f1": best_val["weighted_f1"],
                "val_macro_precision": best_val["macro_precision"],
                "val_macro_recall": best_val["macro_recall"],
            }
            results.append(run_result)
            
            # Print metrics summary
            logger.info(
                "Run %s Finished | Best Epoch: %d | Val Acc: %.4f | Val Macro F1: %.4f | Val Weighted F1: %.4f",
                r["run_name"],
                best_val["best_epoch"],
                best_val["accuracy"],
                best_val["macro_f1"],
                best_val["weighted_f1"]
            )
            
            # Save progress incrementally to CSV
            pd.DataFrame(results).to_csv(tuning_output_csv, index=False)
            
        else:
            logger.error("Best metrics file not found for run %s", r["run_name"])

        # Free memory aggressively to avoid OOM
        del model, trainer, train_loader, val_loader, trainer_cfg
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Hyperparameter sweep complete. Results saved to %s", tuning_output_csv)
    print("\nHyperparameter Sweep Summary Table:\n")
    summary_df = pd.DataFrame(results)
    print(summary_df[["run_name", "best_epoch", "val_accuracy", "val_macro_f1", "val_weighted_f1", "val_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
