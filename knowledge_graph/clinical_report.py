"""Deterministic clinical decision support report generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import MappedConcept, Recommendation, ShapTokenFactor

DISCLAIMER = (
    "Research decision-support output only; not a diagnosis, treatment plan, "
    "or emergency response tool."
)


def build_report(
    *,
    input_preview: str,
    prediction,
    positive_factors: list[ShapTokenFactor],
    negative_factors: list[ShapTokenFactor],
    concepts: list[MappedConcept],
    recommendations: list[Recommendation],
) -> dict[str, Any]:
    """Build a traceable structured report from model, SHAP, and graph outputs."""
    return {
        "input_preview": input_preview,
        "prediction": asdict(prediction),
        "shap": {
            "positive_factors": [asdict(factor) for factor in positive_factors],
            "negative_factors": [asdict(factor) for factor in negative_factors],
        },
        "mapped_concepts": [asdict(concept) for concept in concepts],
        "recommendations": [asdict(rec) for rec in recommendations],
        "disclaimer": DISCLAIMER,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Clinical Decision Support Report",
        "",
        report.get("disclaimer", DISCLAIMER),
        "",
        "## Prediction",
        "",
        f"- Severity label: {report.get('prediction', {}).get('label')}",
        f"- Confidence: {float(report.get('prediction', {}).get('confidence', 0.0)):.4f}",
        "",
        "## SHAP Important Factors",
        "",
        "### Positive",
    ]
    for factor in report.get("shap", {}).get("positive_factors", []):
        lines.append(f"- {factor['token']}: {factor['value']:.4f}")

    lines.extend(["", "### Negative"])
    for factor in report.get("shap", {}).get("negative_factors", []):
        lines.append(f"- {factor['token']}: {factor['value']:.4f}")

    lines.extend(["", "## Mapped Graph Concepts"])
    for concept in report.get("mapped_concepts", []):
        lines.append(
            f"- {concept['name']} ({concept['label']}) via {concept['matched_alias']}"
        )

    lines.extend(["", "## Evidence-Backed Recommendations"])
    for rec in report.get("recommendations", []):
        lines.append(f"- {rec['name']} | score={rec['score']:.4f}")
        if rec.get("description"):
            lines.append(f"  Why this helps: {rec['description']}")
        if rec.get("concepts"):
            lines.append(f"  Concepts: {', '.join(rec['concepts'])}")
        if rec.get("action_steps"):
            lines.append("  Suggested steps:")
            for step in rec["action_steps"]:
                lines.append(f"  - {step}")
        if rec.get("support_options"):
            lines.append("  Support options:")
            for option in rec["support_options"]:
                lines.append(f"  - {option}")
        if rec.get("evidence"):
            evidence_names = [item.get("name", "") for item in rec["evidence"]]
            lines.append(f"  Evidence: {', '.join(evidence_names)}")
        if rec.get("evidence_chunks"):
            lines.append("  Evidence chunks:")
            for chunk in rec["evidence_chunks"]:
                source = chunk.get("document_name", "Evidence document")
                section = chunk.get("section_title", "Section")
                lines.append(
                    f"  - {source} / {section} [{chunk.get('chunk_id', '')}]"
                )
                lines.append(f"    {chunk.get('text', '')}")
                if chunk.get("url"):
                    lines.append(f"    Source: {chunk['url']}")
        if rec.get("resources"):
            resource_names = [item.get("name", "") for item in rec["resources"]]
            lines.append(f"  Resources: {', '.join(resource_names)}")

    return "\n".join(lines) + "\n"


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Write report JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_clinical_report.json"
    markdown_path = output_dir / f"{run_id}_clinical_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
