"""Generate a Neo4j-backed XAI clinical decision support report."""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.clinical_report import build_report, write_report
from knowledge_graph.graph_loader import Neo4jSettings, open_driver
from knowledge_graph.recommendation_engine import rank_recommendations
from knowledge_graph.retriever import retrieve_intervention_rows
from knowledge_graph.shap_mapper import map_shap_tokens
from utils import load_config, set_seed
from xai.shap_explainer import explain_text, load_final_predictor, write_shap_artifacts


def parse_args():
    parser = ArgumentParser(
        description="Generate a deterministic Neo4j-backed XAI clinical report."
    )
    parser.add_argument("--text", required=True, help="Input text to analyze.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional artifact prefix. Defaults to a normalized text preview.",
    )
    return parser.parse_args()


def _default_run_id(text: str) -> str:
    preview = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return preview[:40] or "sample_text"


def main() -> None:
    args = parse_args()
    cfg = load_config()
    set_seed(cfg.SEED)
    run_id = args.run_id or _default_run_id(args.text)

    predictor = load_final_predictor(cfg, project_root=PROJECT_ROOT)
    explanation = explain_text(
        predictor,
        args.text,
        top_k=int(cfg.xai.top_k_tokens),
    )
    write_shap_artifacts(
        explanation,
        PROJECT_ROOT / cfg.xai.shap_output_dir,
        run_id,
    )

    settings = Neo4jSettings.from_config(cfg)
    driver = open_driver(settings)
    try:
        with driver.session(database=settings.database) as session:
            concepts = map_shap_tokens(session, explanation.positive)
            rows = retrieve_intervention_rows(
                session,
                concepts=concepts,
                severity=explanation.prediction.label,
            )
    finally:
        driver.close()

    recommendations = rank_recommendations(
        rows,
        top_k=int(cfg.xai.top_k_recommendations),
    )
    report = build_report(
        input_preview=args.text[:240],
        prediction=explanation.prediction,
        positive_factors=explanation.positive,
        negative_factors=explanation.negative,
        concepts=concepts,
        recommendations=recommendations,
    )
    paths = write_report(report, PROJECT_ROOT / cfg.xai.reports_output_dir, run_id)

    print(
        "Clinical report complete | "
        f"label={explanation.prediction.label} "
        f"confidence={explanation.prediction.confidence:.4f} "
        f"recommendations={len(recommendations)}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
