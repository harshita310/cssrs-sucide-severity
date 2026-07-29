import json
import builtins
from pathlib import Path

from knowledge_graph.schema import ShapTokenFactor
from xai.shap_explainer import (
    PredictionResult,
    ShapExplanation,
    explain_text,
    write_shap_artifacts,
)


class FakePredictor:
    tokenizer = None

    def predict_one(self, text: str):
        return PredictionResult(label=1, confidence=0.8, probabilities=[0.2, 0.8])

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            score = 0.8
            if "hopeless" not in text:
                score -= 0.4
            if "alone" not in text:
                score -= 0.2
            rows.append([1.0 - score, score])
        return rows


class SleepFakePredictor:
    tokenizer = None

    def predict_one(self, text: str):
        return PredictionResult(label=1, confidence=0.8, probabilities=[0.2, 0.8])

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            score = 0.8
            if "cannot sleep" not in text:
                score -= 0.3
            rows.append([1.0 - score, score])
        return rows


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


def test_explain_text_falls_back_when_shap_import_is_blocked(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("blocked by application control")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    explanation = explain_text(FakePredictor(), "hopeless and alone", top_k=3)
    assert explanation.prediction.label == 1
    assert explanation.positive
    assert explanation.positive[0].token == "hopeless"


def test_write_shap_artifacts_accepts_fallback_metadata(tmp_path: Path):
    explanation = ShapExplanation(
        prediction=PredictionResult(label=1, confidence=0.8, probabilities=[0.2, 0.8]),
        positive=[ShapTokenFactor("hopeless", 0.4, 1, "positive")],
        negative=[],
        values=[
            {
                "token": "hopeless",
                "value": 0.4,
                "method": "occlusion_fallback",
                "shap_import_error": "blocked",
            }
        ],
    )
    paths = write_shap_artifacts(explanation, tmp_path, "fallback")
    csv_text = paths["tokens_csv"].read_text(encoding="utf-8")
    assert "shap_import_error" in csv_text


def test_fallback_includes_meaningful_bigrams_when_shap_is_blocked(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("blocked by application control")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    explanation = explain_text(SleepFakePredictor(), "I cannot sleep", top_k=5)
    tokens = {row["token"] for row in explanation.values}
    assert "cannot sleep" in tokens
