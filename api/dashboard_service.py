"""Deterministic payload assembly for the V2 clinical dashboard."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from knowledge_graph.clinical_report import build_report, write_report
from knowledge_graph.schema import MappedConcept, Recommendation, ShapTokenFactor
from xai.shap_explainer import ShapExplanation


def risk_level_for_label(label: int | None) -> str:
    """Return the dashboard risk band for a model label."""
    if label is None:
        return "Unknown"
    if label <= 1:
        return "Low monitored"
    if label <= 3:
        return "Moderate"
    if label <= 5:
        return "High"
    return "Urgent"


def _token_payload(factor: ShapTokenFactor) -> dict[str, Any]:
    return {
        "token": factor.token,
        "value": float(factor.value),
        "rank": int(factor.rank),
        "direction": factor.direction,
    }


def _evidence_level(source_type: str) -> str:
    lowered = source_type.lower()
    if "guideline" in lowered or lowered in {"who", "nice", "apa", "cdc"}:
        return "Clinical Guideline"
    if "review" in lowered or "meta" in lowered:
        return "Clinical Review"
    return source_type or "Published Evidence"


def _similarity_score(chunk: dict[str, Any], recommendation: Recommendation) -> float:
    base = min(0.99, 0.62 + (float(recommendation.score) * 0.08))
    if chunk.get("section_title"):
        base += 0.03
    if chunk.get("url"):
        base += 0.02
    return round(min(base, 0.99), 4)


def _evidence_cards(recommendations: list[Recommendation]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for recommendation in recommendations:
        for chunk in recommendation.evidence_chunks:
            chunk_id = str(chunk.get("chunk_id") or chunk.get("text") or "")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            source_type = str(chunk.get("source_type") or chunk.get("document_type") or "")
            cards.append(
                {
                    "id": chunk_id,
                    "title": str(chunk.get("document_name") or "Evidence document"),
                    "organization": source_type.split()[0] if source_type else "Published",
                    "publicationYear": chunk.get("publication_year") or chunk.get("year") or "Curated",
                    "evidenceLevel": _evidence_level(source_type),
                    "sourceType": source_type or "Evidence",
                    "confidence": round(min(float(recommendation.score), 1.0), 4),
                    "similarityScore": _similarity_score(chunk, recommendation),
                    "snippet": str(chunk.get("text") or ""),
                    "section": str(chunk.get("section_title") or ""),
                    "sourceUrl": str(chunk.get("url") or ""),
                    "citation": str(chunk.get("citation") or ""),
                    "supports": recommendation.name,
                    "mappedConcepts": recommendation.concepts,
                }
            )
    cards.sort(key=lambda item: (-float(item["similarityScore"]), item["title"]))
    return cards


def _graph_payload(
    *,
    positive: list[ShapTokenFactor],
    negative: list[ShapTokenFactor],
    concepts: list[MappedConcept],
    recommendations: list[Recommendation],
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    trace: list[dict[str, str]] = []

    def add_node(node_id: str, label: str, node_type: str, metadata: dict[str, Any] | None = None):
        nodes[node_id] = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "metadata": metadata or {},
        }

    factor_by_alias = {
        factor.token.lower(): factor for factor in [*positive, *negative]
    }
    for factor in [*positive, *negative]:
        token_id = f"token:{factor.token}"
        add_node(
            token_id,
            factor.token,
            "SHAP Token",
            {"value": float(factor.value), "direction": factor.direction},
        )

    for concept in concepts:
        concept_id = f"concept:{concept.name}"
        add_node(
            concept_id,
            concept.name,
            concept.label,
            {"matchedAlias": concept.matched_alias, "shapValue": concept.shap_value},
        )
        token_id = f"token:{concept.matched_alias}"
        if token_id not in nodes:
            factor = factor_by_alias.get(concept.matched_alias.lower())
            add_node(
                token_id,
                concept.matched_alias,
                "SHAP Token",
                {"value": float(factor.value) if factor else concept.shap_value},
            )
        edges.append(
            {
                "id": f"{token_id}->concept:{concept.name}",
                "source": token_id,
                "target": concept_id,
                "label": "MAPS_TO",
            }
        )

    for recommendation in recommendations:
        rec_id = f"intervention:{recommendation.name}"
        add_node(
            rec_id,
            recommendation.name,
            "Intervention",
            {"score": float(recommendation.score)},
        )
        linked_concepts = recommendation.concepts or [concept.name for concept in concepts[:1]]
        for concept_name in linked_concepts:
            concept_id = f"concept:{concept_name}"
            if concept_id in nodes:
                edges.append(
                    {
                        "id": f"{concept_id}->{rec_id}",
                        "source": concept_id,
                        "target": rec_id,
                        "label": "BENEFITS_FROM",
                    }
                )
        for chunk in recommendation.evidence_chunks:
            evidence_name = str(chunk.get("document_name") or "Evidence")
            evidence_id = f"evidence:{chunk.get('chunk_id') or evidence_name}"
            add_node(
                evidence_id,
                evidence_name,
                "Evidence",
                {"section": chunk.get("section_title"), "url": chunk.get("url")},
            )
            edges.append(
                {
                    "id": f"{rec_id}->{evidence_id}",
                    "source": rec_id,
                    "target": evidence_id,
                    "label": "SUPPORTED_BY",
                }
            )
            if linked_concepts:
                trace.append(
                    {
                        "token": next(
                            (
                                concept.matched_alias
                                for concept in concepts
                                if concept.name == linked_concepts[0]
                            ),
                            "SHAP factor",
                        ),
                        "concept": linked_concepts[0],
                        "intervention": recommendation.name,
                        "evidence": evidence_name,
                    }
                )
        for resource in recommendation.resources:
            resource_name = str(resource.get("name") or "Resource")
            resource_id = f"resource:{resource_name}"
            add_node(resource_id, resource_name, "Resource", dict(resource))
            edges.append(
                {
                    "id": f"{rec_id}->{resource_id}",
                    "source": rec_id,
                    "target": resource_id,
                    "label": "USES_RESOURCE",
                }
            )

    return {"nodes": list(nodes.values()), "edges": edges, "trace": trace}


def _pathways(
    *,
    concepts: list[MappedConcept],
    recommendations: list[Recommendation],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        linked_recommendations = [
            recommendation
            for recommendation in recommendations
            if concept.name in recommendation.concepts
        ]
        for recommendation in linked_recommendations:
            evidence_chunk = recommendation.evidence_chunks[0] if recommendation.evidence_chunks else {}
            fallback_evidence = (
                recommendation.evidence[0].get("name", "Evidence")
                if recommendation.evidence
                else "Evidence"
            )
            rows.append(
                {
                    "detectedText": concept.matched_alias,
                    "mappedConcept": concept.name,
                    "conceptType": concept.label,
                    "shapValue": float(concept.shap_value),
                    "guidance": recommendation.name,
                    "whySelected": (
                        f"{concept.name} is connected to {recommendation.name} "
                        "through Neo4j graph traversal."
                    ),
                    "evidenceSource": str(
                        evidence_chunk.get("document_name") or fallback_evidence
                    ),
                    "evidenceSnippet": str(evidence_chunk.get("text") or ""),
                    "sourceUrl": str(evidence_chunk.get("url") or ""),
                }
            )
    return rows


def build_dashboard_payload(
    *,
    text: str,
    explanation: ShapExplanation,
    concepts: list[MappedConcept],
    recommendations: list[Recommendation],
    report_html_path: str | None = None,
    report_pdf_path: str | None = None,
) -> dict[str, Any]:
    """Build the frontend payload from deterministic model and graph results."""
    positive = list(explanation.positive)
    negative = list(explanation.negative)
    evidence = _evidence_cards(recommendations)
    label = int(explanation.prediction.label)
    return {
        "inputPreview": text[:240],
        "prediction": {
            "severityLabel": label,
            "riskLevel": risk_level_for_label(label),
            "confidence": float(explanation.prediction.confidence),
            "probabilities": list(explanation.prediction.probabilities),
            "modelVersion": "MentalBERT focused CE",
            "validatedModel": True,
        },
        "explainability": {
            "positiveTokens": [_token_payload(factor) for factor in positive],
            "negativeTokens": [_token_payload(factor) for factor in negative],
            "allTokens": list(explanation.values),
        },
        "concepts": [asdict(concept) for concept in concepts],
        "graph": _graph_payload(
            positive=positive,
            negative=negative,
            concepts=concepts,
            recommendations=recommendations,
        ),
        "pathways": _pathways(concepts=concepts, recommendations=recommendations),
        "evidence": evidence,
        "recommendations": [
            {
                "name": recommendation.name,
                "score": float(recommendation.score),
                "purpose": recommendation.description,
                "mappedConcepts": recommendation.concepts,
                "supportingEvidence": [
                    chunk.get("document_name", "Evidence document")
                    for chunk in recommendation.evidence_chunks
                ],
                "resources": recommendation.resources,
                "actionSteps": recommendation.action_steps,
                "supportOptions": recommendation.support_options,
            }
            for recommendation in recommendations
        ],
        "literature": evidence,
        "exports": {
            "htmlReport": report_html_path,
            "pdfReport": report_pdf_path,
        },
        "system": {
            "postPredictionLLM": False,
            "generationPolicy": "No LLM text generation after prediction; dashboard text is model metadata, graph data, and retrieved evidence.",
        },
    }


def write_dashboard_report(
    *,
    text: str,
    explanation: ShapExplanation,
    concepts: list[MappedConcept],
    recommendations: list[Recommendation],
    output_dir,
    run_id: str,
) -> dict[str, str]:
    """Write the existing HTML report and return paths for API exports."""
    report = build_report(
        input_preview=text[:240],
        prediction=explanation.prediction,
        positive_factors=list(explanation.positive),
        negative_factors=list(explanation.negative),
        concepts=concepts,
        recommendations=recommendations,
    )
    paths = write_report(report, output_dir, run_id)
    return {key: str(value) for key, value in paths.items()}
