"""Compare saved MentalBERT-CSSR checkpoints on validation and test splits."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_evaluation import evaluate_split
from utils import ensure_directories, get_logger, load_config, set_seed
from utils.device import resolve_device
from utils.model import load_checkpoint, load_mentalbert_for_classification
from utils.runtime_settings import resolve_max_length
from utils.tokenization import load_mentalbert_tokenizer

logger = get_logger("compare_checkpoints")


def parse_args():
    parser = ArgumentParser(
        description="Evaluate multiple checkpoints and rank them by a metric."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path, relative to the project root.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="saved_model/tune/best_model_*.pt",
        help="Glob of checkpoints to compare, relative to project root.",
    )
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="Also compare the checkpoint configured as paths.model_path.",
    )
    parser.add_argument(
        "--metric",
        default="weighted_f1",
        help="Metric used to rank checkpoints on the test split.",
    )
    parser.add_argument(
        "--output-dir",
        default="RESULTS/metrics/checkpoint_comparison",
        help="Directory for per-checkpoint artifacts and summary CSV.",
    )
    return parser.parse_args()


def _safe_name(path: Path) -> str:
    return path.stem.replace("best_model_", "").replace(" ", "_")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.SEED)
    ensure_directories(cfg)
    device = resolve_device(cfg.DEVICE)

    pattern = Path(args.checkpoint_glob)
    if pattern.is_absolute():
        checkpoints = sorted(pattern.parent.glob(pattern.name))
    else:
        checkpoints = sorted(PROJECT_ROOT.glob(str(pattern)))
    if args.include_current:
        current = Path(cfg.MODEL_PATH)
        if not current.is_absolute():
            current = PROJECT_ROOT / current
        checkpoints.insert(0, current)

    checkpoints = [p for p in dict.fromkeys(checkpoints) if p.exists()]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob!r}")

    max_length, _ = resolve_max_length(
        cfg.MAX_LENGTH,
        recommendation_path=cfg.tokenization.recommendation_path,
        project_root=PROJECT_ROOT,
    )
    splits_dir = PROJECT_ROOT / cfg.training.splits_dir
    split_frames = {
        "validation": pd.read_csv(splits_dir / "val.csv"),
        "test": pd.read_csv(splits_dir / "test.csv"),
    }
    tokenizer = load_mentalbert_tokenizer(
        model_name=cfg.MODEL_NAME,
        truncation_side=cfg.model.truncation_side,
    )
    label_list = list(range(cfg.NUM_LABELS))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for checkpoint_path in checkpoints:
        run_name = _safe_name(checkpoint_path)
        run_dir = output_dir / run_name
        logger.info("Evaluating %s", checkpoint_path)

        model = load_mentalbert_for_classification(
            model_name=cfg.MODEL_NAME,
            num_labels=cfg.NUM_LABELS,
            dropout=cfg.DROPOUT,
        )
        checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
        model.to(device)

        split_metrics = {}
        for split_name, split_df in split_frames.items():
            split_metrics[split_name] = evaluate_split(
                split_name=split_name,
                df=split_df,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
                max_length=max_length,
                label_list=label_list,
                output_dir=run_dir,
            )

        row = {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_metric": checkpoint.get("best_metric"),
        }
        for split_name, metrics in split_metrics.items():
            for key in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "quadratic_weighted_kappa",
                "within_1_accuracy",
                "high_severity_f1",
                "loss",
            ]:
                if key in metrics:
                    row[f"{split_name}_{key}"] = metrics[key]
        rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    rank_col = f"test_{args.metric}"
    ascending = args.metric in {"loss", "mean_absolute_error", "log_loss"}
    if rank_col in summary.columns:
        summary = summary.sort_values(rank_col, ascending=ascending)
    summary_path = output_dir / "checkpoint_comparison.csv"
    summary.to_csv(summary_path, index=False)
    with (output_dir / "checkpoint_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(orient="records"), fh, indent=2)

    print("\n=== Checkpoint Comparison ===\n")
    print(summary.to_string(index=False))
    print(f"\nBest by {rank_col}: {summary.iloc[0]['checkpoint']}")
    print(f"Artifacts -> {output_dir}")


if __name__ == "__main__":
    main()
