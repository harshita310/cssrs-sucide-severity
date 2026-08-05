"""FastAPI entrypoint for the V2 clinical dashboard."""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.dashboard_service import build_dashboard_payload, write_dashboard_report
from knowledge_graph.graph_loader import Neo4jSettings, open_driver
from knowledge_graph.recommendation_engine import rank_recommendations
from knowledge_graph.retriever import retrieve_intervention_rows
from knowledge_graph.shap_mapper import map_shap_tokens
from scripts.generate_clinical_report import report_mapping_factors
from utils import load_config, set_seed
from xai.shap_explainer import explain_text, load_final_predictor


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    run_id: str | None = Field(default=None, max_length=80)


class AnalyzeResponse(BaseModel):
    dashboard: dict[str, Any]


def _run_id(text: str, explicit: str | None) -> str:
    if explicit:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", explicit).strip("_")
        return safe[:80] or "dashboard_run"
    preview = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return f"v2_{preview[:40] or 'sample_text'}"


@lru_cache(maxsize=1)
def _runtime():
    cfg = load_config()
    set_seed(cfg.SEED)
    predictor = load_final_predictor(cfg, project_root=PROJECT_ROOT)
    settings = Neo4jSettings.from_config(cfg)
    return cfg, predictor, settings


app = FastAPI(title="CSSRS V2 Explainable Clinical Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "postPredictionLLM": False}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    cfg, predictor, settings = _runtime()
    run_id = _run_id(request.text, request.run_id)
    explanation = explain_text(
        predictor,
        request.text,
        top_k=int(cfg.xai.top_k_tokens),
    )
    mapping_factors = report_mapping_factors(explanation.positive, explanation.negative)

    driver = open_driver(settings)
    try:
        with driver.session(database=settings.database) as session:
            concepts = map_shap_tokens(session, mapping_factors)
            rows = retrieve_intervention_rows(
                session,
                concepts=concepts,
                severity=explanation.prediction.label,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j retrieval failed: {exc}",
        ) from exc
    finally:
        driver.close()

    recommendations = rank_recommendations(
        rows,
        top_k=int(cfg.xai.top_k_recommendations),
    )
    export_paths = write_dashboard_report(
        text=request.text,
        explanation=explanation,
        concepts=concepts,
        recommendations=recommendations,
        output_dir=PROJECT_ROOT / cfg.xai.reports_output_dir,
        run_id=run_id,
    )
    dashboard = build_dashboard_payload(
        text=request.text,
        explanation=explanation,
        concepts=concepts,
        recommendations=recommendations,
        report_html_path=export_paths.get("html"),
    )
    return AnalyzeResponse(dashboard=dashboard)


@app.get("/api/report/html")
def html_report(path: str):
    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path = report_path.resolve()
    reports_root = (PROJECT_ROOT / "RESULTS" / "xai" / "reports").resolve()
    if reports_root not in report_path.parents or not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="text/html")

