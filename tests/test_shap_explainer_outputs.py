import json
from pathlib import Path

from knowledge_graph.schema import ShapTokenFactor
from xai.shap_explainer import PredictionResult, ShapExplanation, write_shap_artifacts


def test_write_shap_artifacts_creates_json_and_csv(tmp_path: Path):
    explanation = ShapExplanation(
        prediction=PredictionResult(
            label=3,
            confidence=0.91,
            probabilities=[0.01, 0.02, 0.03, 0.91, 0.01, 0.01, 0.01],
        ),
        positive=[ShapTokenFactor("hopeless", 0.6, 1, "positive")],
        negative=[ShapTokenFactor("support", -0.2, 1, "negative")],
        values=[{"token": "hopeless", "value": 0.6}],
    )
    paths = write_shap_artifacts(explanation, tmp_path, "sample")
    assert paths["json"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["prediction"]["label"] == 3
    assert paths["tokens_csv"].exists()
