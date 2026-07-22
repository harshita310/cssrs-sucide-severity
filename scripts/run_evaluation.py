"""Full evaluation on validation and held-out test splits (Notebook 05 pipeline)."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import ensure_directories, get_logger, load_config, set_seed
from utils.dataset import create_dataloader
from utils.device import resolve_device
from utils.metrics import (
    compute_classification_metrics,
    metrics_summary_table,
    plot_confusion_matrix,
)
from utils.model import load_mentalbert_for_classification, load_checkpoint
from utils.runtime_settings import resolve_max_length
from utils.tokenization import load_mentalbert_tokenizer

logger = get_logger("run_evaluation")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Evaluate a MentalBERT-CSSR checkpoint on validation/test splits."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path, relative to the project root.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Checkpoint path to evaluate. Defaults to paths.model_path in config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation artifacts. Defaults to evaluation.output_dir.",
    )
    return parser.parse_args()


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[list[int], list[int], list[list[float]], float]:
    """Return y_true, y_pred, y_prob, mean loss for a split."""
    import torch.nn.functional as F

    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[list[float]] = []
    losses: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        logits = outputs.logits
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        losses.append(float(loss.item()))
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_prob.extend(probs.cpu().tolist())

    mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")
    return all_true, all_pred, all_prob, mean_loss


def evaluate_split(
    *,
    split_name: str,
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer,
    cfg,
    device: torch.device,
    max_length: int,
    label_list: list[int],
    output_dir: Path,
) -> dict:
    pin_memory = bool(cfg.runtime.pin_memory) and device.type == "cuda"
    loader = create_dataloader(
        df,
        tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        max_length=max_length,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        pin_memory=pin_memory,
    )

    use_amp = bool(cfg.USE_MIXED_PRECISION) and device.type == "cuda"
    y_true, y_pred, y_prob, loss = predict_split(
        model, loader, device, use_amp=use_amp
    )

    import numpy as np

    prob_array = np.asarray(y_prob, dtype=float)
    threshold = int(getattr(cfg.evaluation, "high_severity_threshold", 4))
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=label_list,
        target_names=[str(x) for x in label_list],
        y_prob=prob_array,
        high_severity_threshold=threshold,
    )
    metrics["loss"] = loss
    metrics["split"] = split_name
    metrics["n_samples"] = len(y_true)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{split_name}_metrics.json"
    summary_path = output_dir / f"{split_name}_metrics_summary.csv"
    report_path = output_dir / f"{split_name}_classification_report.csv"
    preds_path = output_dir / f"{split_name}_predictions.csv"

    payload = {
        k: v for k, v in metrics.items() if k not in {"y_true", "y_pred", "y_prob"}
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    metrics_summary_table(metrics).to_csv(summary_path, index=False)
    pd.DataFrame(metrics["classification_report"]).T.to_csv(report_path)

    pred_df = df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["correct"] = [int(t == p) for t, p in zip(y_true, y_pred)]
    pred_df["abs_error"] = [abs(t - p) for t, p in zip(y_true, y_pred)]
    pred_df.to_csv(preds_path, index=False)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        labels=label_list,
        output_path=output_dir / f"{split_name}_confusion_matrix.png",
        title=f"{split_name.title()} Confusion Matrix",
        dpi=int(cfg.eda.figure_dpi),
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        labels=label_list,
        output_path=output_dir / f"{split_name}_confusion_matrix_normalized.png",
        title=f"{split_name.title()} Confusion Matrix (row-normalized)",
        dpi=int(cfg.eda.figure_dpi),
        normalize=True,
    )

    logger.info(
        "%s | acc=%.4f bal_acc=%.4f macro_f1=%.4f qwk=%.4f mae=%.4f within1=%.4f high_sev_f1=%.4f",
        split_name,
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["quadratic_weighted_kappa"],
        metrics["mean_absolute_error"],
        metrics["within_1_accuracy"],
        metrics["high_severity_f1"],
    )
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.SEED)
    ensure_directories(cfg)
    device = resolve_device(cfg.DEVICE)

    max_length, _ = resolve_max_length(
        cfg.MAX_LENGTH,
        recommendation_path=cfg.tokenization.recommendation_path,
        project_root=PROJECT_ROOT,
    )

    splits_dir = PROJECT_ROOT / cfg.training.splits_dir
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    model_path = Path(args.model_path) if args.model_path else Path(cfg.MODEL_PATH)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    tokenizer = load_mentalbert_tokenizer(
        model_name=cfg.MODEL_NAME,
        truncation_side=cfg.model.truncation_side,
    )
    model = load_mentalbert_for_classification(
        model_name=cfg.MODEL_NAME,
        num_labels=cfg.NUM_LABELS,
        dropout=cfg.DROPOUT,
    )
    checkpoint = load_checkpoint(model_path, model, map_location=device)
    model.to(device)

    label_list = list(range(cfg.NUM_LABELS))
    eval_dir = Path(args.output_dir) if args.output_dir else Path(cfg.evaluation.output_dir)
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir

    results: dict[str, dict] = {}
    for split_name, split_df in [("validation", val_df), ("test", test_df)]:
        results[split_name] = evaluate_split(
            split_name=split_name,
            df=split_df,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            max_length=max_length,
            label_list=label_list,
            output_dir=eval_dir,
        )

    # Combined comparison table
    compare_rows = []
    for split_name, m in results.items():
        row = {"split": split_name, "checkpoint_epoch": checkpoint.get("epoch")}
        for key in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "cohen_kappa",
            "quadratic_weighted_kappa",
            "matthews_corrcoef",
            "mean_absolute_error",
            "within_1_accuracy",
            "within_2_accuracy",
            "high_severity_f1",
            "macro_ovr_auc",
            "top2_accuracy",
            "top3_accuracy",
            "log_loss",
            "loss",
        ]:
            if key in m:
                row[key] = m[key]
        compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)
    compare_path = eval_dir / "evaluation_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    with (eval_dir / "evaluation_experiment.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_path": str(model_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "best_metric": checkpoint.get("best_metric"),
                "splits": compare_rows,
            },
            fh,
            indent=2,
        )

    print("\n=== Full Evaluation Summary ===\n")
    print(compare_df.to_string(index=False))
    print(f"\nArtifacts -> {eval_dir}")


if __name__ == "__main__":
    main()
