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
        "shap": {
            "positive_factors": [{"token": "alone", "value": 0.4}],
            "negative_factors": [{"token": "support", "value": -0.2}],
        },
        "mapped_concepts": [
            {
                "name": "Isolation",
                "label": "Symptom",
                "matched_alias": "alone",
                "shap_value": 0.4,
            }
        ],
        "recommendations": [
            {
                "name": "Peer Support",
                "score": 1.2,
                "concepts": ["Isolation"],
                "evidence": [{"name": "WHO Suicide Prevention Guidance"}],
                "resources": [{"name": "Support Group"}],
            }
        ],
        "disclaimer": "research only",
    }
    paths = write_report(report, tmp_path, "sample")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["prediction"][
        "label"
    ] == 2
    assert paths["markdown"].exists()
    assert paths["shap_chart"].exists()
    assert paths["graph_path"].exists()
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "sample_shap_chart.png" in markdown
    assert "sample_graph_path.png" in markdown
