"""Deterministic clinical decision support report generation."""

from __future__ import annotations

import json
import textwrap
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


def _plot_shap_chart(report: dict[str, Any], output_path: Path) -> None:
    """Create a simple SHAP token contribution bar chart."""
    import matplotlib.pyplot as plt

    factors = []
    factors.extend(report.get("shap", {}).get("positive_factors", []))
    factors.extend(report.get("shap", {}).get("negative_factors", []))
    factors = sorted(factors, key=lambda item: abs(float(item.get("value", 0.0))))[-12:]

    if not factors:
        factors = [{"token": "no mapped token", "value": 0.0}]

    labels = [str(item["token"]) for item in factors]
    values = [float(item["value"]) for item in factors]
    colors = ["#3b82f6" if value >= 0 else "#ef4444" for value in values]

    fig_height = max(3.2, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_title("Token Contribution Summary")
    ax.set_xlabel("Attribution value")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_graph_path(report: dict[str, Any], output_path: Path) -> None:
    """Create a visual graph-path diagram from mapped concepts to evidence."""
    import matplotlib.pyplot as plt

    concept = "Mapped concept"
    if report.get("mapped_concepts"):
        concept = report["mapped_concepts"][0].get("name", concept)

    recommendation = "Recommendation"
    evidence = "Evidence"
    resource = "Resource"
    if report.get("recommendations"):
        rec = report["recommendations"][0]
        recommendation = rec.get("name", recommendation)
        if rec.get("evidence"):
            evidence = rec["evidence"][0].get("name", evidence)
        elif rec.get("evidence_chunks"):
            evidence = rec["evidence_chunks"][0].get("document_name", evidence)
        if rec.get("resources"):
            resource = rec["resources"][0].get("name", resource)

    nodes = [
        ("SHAP token", "#dbeafe"),
        (concept, "#dcfce7"),
        (recommendation, "#fef3c7"),
        (evidence, "#ede9fe"),
        (resource, "#fee2e2"),
    ]

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_axis_off()
    x_positions = [0.07, 0.29, 0.50, 0.72, 0.93]
    for idx, ((label, color), x_pos) in enumerate(zip(nodes, x_positions)):
        wrapped_label = "\n".join(textwrap.wrap(label, width=18)) or label
        ax.text(
            x_pos,
            0.55,
            wrapped_label,
            ha="center",
            va="center",
            fontsize=9,
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": color,
                "edgecolor": "#374151",
                "linewidth": 1.0,
            },
        )
        if idx < len(nodes) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.095, 0.55),
                xytext=(x_pos + 0.095, 0.55),
                arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.4},
            )
    ax.set_title("Explanation Graph Path", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _markdown_report(
    report: dict[str, Any],
    *,
    shap_chart_name: str | None = None,
    graph_path_name: str | None = None,
) -> str:
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
        "## Visual Summary",
        "",
    ]
    if shap_chart_name:
        lines.append(f"![SHAP contribution chart]({shap_chart_name})")
    if graph_path_name:
        lines.append(f"![Explanation graph path]({graph_path_name})")

    lines.extend(
        [
            "",
        "## SHAP Important Factors",
        "",
        "### Positive",
        ]
    )
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
    shap_chart_path = output_dir / f"{run_id}_shap_chart.png"
    graph_path_path = output_dir / f"{run_id}_graph_path.png"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_shap_chart(report, shap_chart_path)
    _plot_graph_path(report, graph_path_path)
    markdown_path.write_text(
        _markdown_report(
            report,
            shap_chart_name=shap_chart_path.name,
            graph_path_name=graph_path_path.name,
        ),
        encoding="utf-8",
    )
    return {
        "json": json_path,
        "markdown": markdown_path,
        "shap_chart": shap_chart_path,
        "graph_path": graph_path_path,
    }
