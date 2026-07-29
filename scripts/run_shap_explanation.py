"""Run SHAP explanation for one text using the final focused CE model."""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config, set_seed
from xai.shap_explainer import explain_text, load_final_predictor, write_shap_artifacts


def parse_args():
    parser = ArgumentParser(description="Explain one MentalBERT-CSSR prediction.")
    parser.add_argument("--text", required=True, help="Input text to explain.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional artifact prefix. Defaults to a normalized text preview.",
    )
    return parser.parse_args()


def _default_run_id(text: str) -> str:
    preview = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (preview[:40] or "sample_text")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    set_seed(cfg.SEED)
    predictor = load_final_predictor(cfg, project_root=PROJECT_ROOT)
    explanation = explain_text(
        predictor,
        args.text,
        top_k=int(cfg.xai.top_k_tokens),
    )
    output_dir = PROJECT_ROOT / cfg.xai.shap_output_dir
    run_id = args.run_id or _default_run_id(args.text)
    paths = write_shap_artifacts(explanation, output_dir, run_id)

    print(
        "SHAP explanation complete | "
        f"label={explanation.prediction.label} "
        f"confidence={explanation.prediction.confidence:.4f}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
