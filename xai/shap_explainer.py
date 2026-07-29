"""SHAP explanation utilities for the final focused CE MentalBERT model."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knowledge_graph.schema import ShapTokenFactor


@dataclass(frozen=True)
class PredictionResult:
    label: int
    confidence: float
    probabilities: list[float]


@dataclass(frozen=True)
class ShapExplanation:
    prediction: PredictionResult
    positive: list[ShapTokenFactor]
    negative: list[ShapTokenFactor]
    values: list[dict[str, Any]]


class MentalBertPredictor:
    """Callable probability predictor used by SHAP and CLI scripts."""

    def __init__(self, *, model, tokenizer, device: torch.device, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict_one(self, text: str) -> PredictionResult:
        probs = self.predict_proba([text])[0]
        label = int(np.argmax(probs))
        return PredictionResult(
            label=label,
            confidence=float(probs[label]),
            probabilities=[float(value) for value in probs],
        )


def load_final_predictor(cfg, *, project_root: Path):
    """Load the final focused CE model as a probability predictor."""
    from utils.device import resolve_device
    from utils.model import load_checkpoint, load_mentalbert_for_classification
    from utils.runtime_settings import resolve_max_length
    from utils.tokenization import load_mentalbert_tokenizer

    device = resolve_device(cfg.DEVICE)
    max_length, _ = resolve_max_length(
        cfg.MAX_LENGTH,
        recommendation_path=cfg.tokenization.recommendation_path,
        project_root=project_root,
    )
    tokenizer = load_mentalbert_tokenizer(
        model_name=cfg.MODEL_NAME,
        truncation_side=cfg.model.truncation_side,
    )
    model = load_mentalbert_for_classification(
        model_name=cfg.MODEL_NAME,
        num_labels=cfg.NUM_LABELS,
        dropout=cfg.DROPOUT,
    )
    load_checkpoint(Path(cfg.MODEL_PATH), model, map_location=device)
    model.to(device)
    return MentalBertPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )


def explain_text(
    predictor: MentalBertPredictor,
    text: str,
    *,
    top_k: int,
) -> ShapExplanation:
    """Compute SHAP token factors for one input text."""
    try:
        import shap
    except ImportError as exc:
        return _explain_text_occlusion_fallback(
            predictor,
            text,
            top_k=top_k,
            import_error=str(exc),
        )

    prediction = predictor.predict_one(text)
    masker = shap.maskers.Text(predictor.tokenizer)
    explainer = shap.Explainer(predictor.predict_proba, masker)
    shap_values = explainer([text])

    class_values = shap_values.values[0, :, prediction.label]
    tokens = list(shap_values.data[0])
    value_rows = [
        {"token": str(token), "value": float(value)}
        for token, value in zip(tokens, class_values)
        if str(token).strip()
    ]
    positive_rows = sorted(
        [row for row in value_rows if row["value"] > 0],
        key=lambda row: row["value"],
        reverse=True,
    )[:top_k]
    negative_rows = sorted(
        [row for row in value_rows if row["value"] < 0],
        key=lambda row: row["value"],
    )[:top_k]

    positive = [
        ShapTokenFactor(
            token=row["token"],
            value=float(row["value"]),
            rank=idx + 1,
            direction="positive",
        )
        for idx, row in enumerate(positive_rows)
    ]
    negative = [
        ShapTokenFactor(
            token=row["token"],
            value=float(row["value"]),
            rank=idx + 1,
            direction="negative",
        )
        for idx, row in enumerate(negative_rows)
    ]
    return ShapExplanation(
        prediction=prediction,
        positive=positive,
        negative=negative,
        values=value_rows,
    )


def _explain_text_occlusion_fallback(
    predictor,
    text: str,
    *,
    top_k: int,
    import_error: str,
) -> ShapExplanation:
    """Produce deterministic token attributions when SHAP cannot import."""
    prediction = predictor.predict_one(text)
    base_prob = float(predictor.predict_proba([text])[0][prediction.label])
    tokens = text.split()
    spans: list[tuple[str, list[int]]] = [
        (token, [index]) for index, token in enumerate(tokens)
    ]
    spans.extend(
        (
            f"{tokens[index]} {tokens[index + 1]}",
            [index, index + 1],
        )
        for index in range(len(tokens) - 1)
    )
    value_rows: list[dict[str, Any]] = []
    for token, indices in spans:
        masked_tokens = [
            current for index, current in enumerate(tokens) if index not in indices
        ]
        masked_text = " ".join(masked_tokens)
        masked_prob = float(
            predictor.predict_proba([masked_text or " "])[0][prediction.label]
        )
        value_rows.append(
            {
                "token": token,
                "value": base_prob - masked_prob,
                "method": "occlusion_fallback",
                "shap_import_error": import_error,
            }
        )

    positive_rows = sorted(
        [row for row in value_rows if row["value"] > 0],
        key=lambda row: row["value"],
        reverse=True,
    )[:top_k]
    negative_rows = sorted(
        [row for row in value_rows if row["value"] < 0],
        key=lambda row: row["value"],
    )[:top_k]
    positive = [
        ShapTokenFactor(
            token=str(row["token"]),
            value=float(row["value"]),
            rank=idx + 1,
            direction="positive",
        )
        for idx, row in enumerate(positive_rows)
    ]
    negative = [
        ShapTokenFactor(
            token=str(row["token"]),
            value=float(row["value"]),
            rank=idx + 1,
            direction="negative",
        )
        for idx, row in enumerate(negative_rows)
    ]
    return ShapExplanation(
        prediction=prediction,
        positive=positive,
        negative=negative,
        values=value_rows,
    )


def _explanation_payload(explanation: ShapExplanation) -> dict[str, Any]:
    return {
        "prediction": asdict(explanation.prediction),
        "positive": [asdict(factor) for factor in explanation.positive],
        "negative": [asdict(factor) for factor in explanation.negative],
        "values": explanation.values,
    }


def write_shap_artifacts(
    explanation: ShapExplanation,
    output_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Write deterministic SHAP JSON and token CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_shap.json"
    csv_path = output_dir / f"{run_id}_shap_tokens.csv"

    payload = _explanation_payload(explanation)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    extra_fields = sorted(
        {
            key
            for row in explanation.values
            for key in row
            if key not in {"token", "value"}
        }
    )
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["token", "value", *extra_fields])
        writer.writeheader()
        writer.writerows(explanation.values)

    return {"json": json_path, "tokens_csv": csv_path}
