from __future__ import annotations

from knowledge_graph.schema import MappedConcept, Recommendation, ShapTokenFactor
from xai.shap_explainer import PredictionResult, ShapExplanation


def test_build_dashboard_payload_contains_traceable_evidence_cards():
    from api.dashboard_service import build_dashboard_payload

    explanation = ShapExplanation(
        prediction=PredictionResult(
            label=1,
            confidence=0.9044,
            probabilities=[0.02, 0.9044, 0.03, 0.02, 0.01, 0.01, 0.0056],
        ),
        positive=[
            ShapTokenFactor(
                token="cannot sleep",
                value=0.0716,
                rank=1,
                direction="positive",
            )
        ],
        negative=[
            ShapTokenFactor(
                token="alone",
                value=-0.0104,
                rank=1,
                direction="negative",
            )
        ],
        values=[],
    )
    concepts = [
        MappedConcept(
            name="Insomnia",
            label="Symptom",
            matched_alias="cannot sleep",
            shap_value=0.0716,
        ),
        MappedConcept(
            name="Isolation",
            label="Symptom",
            matched_alias="alone",
            shap_value=-0.0104,
        ),
    ]
    recommendations = [
        Recommendation(
            name="Sleep Hygiene",
            score=0.6716,
            concepts=["Insomnia"],
            evidence=[{"name": "APA Suicide Risk Practice Guidance"}],
            resources=[{"name": "Mental Health Professional"}],
            description="Structured behavioral habits that support improved sleep.",
            action_steps=["Set a fixed wake-up time for tomorrow."],
            support_options=["Use an audio relaxation exercise."],
            evidence_chunks=[
                {
                    "chunk_id": "apa-guidance-001",
                    "document_name": "APA Suicide Risk Practice Guidance",
                    "section_title": "Structured support and follow-up",
                    "text": "APA-oriented clinical guidance emphasizes structured assessment.",
                    "url": "https://psychiatry.org",
                    "source_type": "APA Guideline",
                    "citation": "American Psychiatric Association practice guidance.",
                }
            ],
        )
    ]

    payload = build_dashboard_payload(
        text="I cannot sleep and I feel alone",
        explanation=explanation,
        concepts=concepts,
        recommendations=recommendations,
        report_html_path="RESULTS/xai/reports/demo.html",
    )

    assert payload["prediction"]["severityLabel"] == 1
    assert payload["prediction"]["riskLevel"] == "Low monitored"
    assert payload["explainability"]["positiveTokens"][0]["token"] == "cannot sleep"
    assert payload["concepts"][0]["name"] == "Insomnia"
    assert payload["evidence"][0]["title"] == "APA Suicide Risk Practice Guidance"
    assert payload["evidence"][0]["similarityScore"] > 0.0
    assert payload["graph"]["trace"][0]["token"] == "cannot sleep"
    assert payload["pathways"][0]["detectedText"] == "cannot sleep"
    assert payload["pathways"][0]["mappedConcept"] == "Insomnia"
    assert payload["pathways"][0]["guidance"] == "Sleep Hygiene"
    assert payload["pathways"][0]["evidenceSource"] == "APA Suicide Risk Practice Guidance"
    assert payload["recommendations"][0]["name"] == "Sleep Hygiene"
    assert payload["exports"]["htmlReport"] == "RESULTS/xai/reports/demo.html"
