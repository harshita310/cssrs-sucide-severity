import json
from pathlib import Path

from knowledge_graph.clinical_report import build_report, write_report
from knowledge_graph.schema import MappedConcept, Recommendation, ShapTokenFactor
from xai.shap_explainer import PredictionResult


def test_build_report_contains_traceability_and_disclaimer():
    report = build_report(
        input_preview="I feel hopeless",
        prediction=PredictionResult(
            label=4,
            confidence=0.88,
            probabilities=[0, 0, 0, 0, 0.88, 0.1, 0.02],
        ),
        positive_factors=[ShapTokenFactor("hopeless", 0.7, 1, "positive")],
        negative_factors=[],
        concepts=[MappedConcept("Hopelessness", "Symptom", "hopeless", 0.7)],
        recommendations=[
            Recommendation(
                "Safety Planning",
                1.2,
                ["Hopelessness"],
                [{"name": "WHO"}],
                [{"name": "Helpline"}],
            )
        ],
    )
    assert report["prediction"]["label"] == 4
    assert report["recommendations"][0]["evidence"][0]["name"] == "WHO"
    assert "research" in report["disclaimer"].lower()


def test_write_report_creates_json_and_markdown(tmp_path: Path):
    report = {
        "prediction": {"label": 2},
        "recommendations": [],
        "disclaimer": "research only",
    }
    paths = write_report(report, tmp_path, "sample")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["prediction"][
        "label"
    ] == 2
    assert paths["markdown"].exists()
