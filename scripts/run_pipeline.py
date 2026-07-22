"""Run the clean MentalBERT-CSSR train -> evaluate pipeline."""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(
        description="Train MentalBERT-CSSR and evaluate the best checkpoint."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path, relative to the project root.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only run evaluation for the configured or provided checkpoint.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Checkpoint to evaluate. Defaults to paths.model_path in config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation artifacts.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()

    config_args = ["--config", args.config] if args.config else []
    if not args.skip_training:
        _run([sys.executable, "scripts/run_training.py", *config_args])

    eval_cmd = [sys.executable, "scripts/run_evaluation.py", *config_args]
    if args.model_path:
        eval_cmd.extend(["--model-path", args.model_path])
    if args.output_dir:
        eval_cmd.extend(["--output-dir", args.output_dir])
    _run(eval_cmd)


if __name__ == "__main__":
    main()
