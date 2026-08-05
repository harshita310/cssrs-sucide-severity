"""Deterministic clinical decision support report generation."""

from __future__ import annotations

import json
import textwrap
from base64 import b64decode
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

from .schema import MappedConcept, Recommendation, ShapTokenFactor

DISCLAIMER = (
    "Research decision-support output only; not a diagnosis, treatment plan, "
    "or emergency response tool."
)

_FALLBACK_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _write_fallback_png(output_path: Path) -> None:
    output_path.write_bytes(b64decode(_FALLBACK_PNG))


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
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_fallback_png(output_path)
        return

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
    try:
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_title("Token Contribution Summary")
        ax.set_xlabel("Attribution value")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    except Exception:
        _write_fallback_png(output_path)


def _plot_graph_path(report: dict[str, Any], output_path: Path) -> None:
    """Create a visual graph-path diagram from mapped concepts to evidence."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_fallback_png(output_path)
        return

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

    try:
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
    except Exception:
        _write_fallback_png(output_path)


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


def _risk_band(label: int | None) -> tuple[str, str]:
    if label is None:
        return "Unknown", "neutral"
    if label <= 1:
        return "Low monitored", "low"
    if label <= 3:
        return "Moderate", "moderate"
    if label <= 5:
        return "High", "high"
    return "Urgent", "urgent"


def _severity_meaning(label: int | None) -> str:
    meanings = {
        0: "Label 0: no or minimal detected suicide-severity signal in this model output.",
        1: "Label 1: low monitored severity; early or indirect distress signal.",
        2: "Label 2: mild-to-moderate indicator requiring contextual review.",
        3: "Label 3: moderate severity signal; review support needs and protective factors.",
        4: "Label 4: high severity band; safety planning and professional support should be considered.",
        5: "Label 5: very high severity band; urgent support pathways may be relevant.",
        6: "Label 6: urgent/highest severity band; immediate crisis or emergency support may be relevant.",
    }
    return meanings.get(label, "Label unavailable: severity meaning could not be resolved.")


def _decision_trace_html(report: dict[str, Any]) -> str:
    concepts = report.get("mapped_concepts", [])
    recommendations = report.get("recommendations", [])
    if not concepts and not recommendations:
        return "<p class=\"muted\">No traceable graph path was produced.</p>"

    rows = []
    for rec in recommendations:
        rec_concepts = set(rec.get("concepts", []))
        linked_concepts = [
            concept for concept in concepts if concept.get("name") in rec_concepts
        ]
        if not linked_concepts:
            linked_concepts = concepts[:1]
        evidence_name = "Graph evidence"
        if rec.get("evidence_chunks"):
            evidence_name = rec["evidence_chunks"][0].get("document_name", evidence_name)
        elif rec.get("evidence"):
            evidence_name = rec["evidence"][0].get("name", evidence_name)
        for concept in linked_concepts:
            token = concept.get("matched_alias", "SHAP factor")
            rows.append(
                "<div class=\"trace-row\">"
                f"<span>{escape(str(token))}</span>"
                "<b>maps to</b>"
                f"<span>{escape(str(concept.get('name', 'Concept')))}</span>"
                "<b>recommends</b>"
                f"<span>{escape(str(rec.get('name', 'Intervention')))}</span>"
                "<b>supported by</b>"
                f"<span>{escape(str(evidence_name))}</span>"
                "</div>"
            )
    return "".join(rows)


def _html_list(items: list[str]) -> str:
    if not items:
        return "<p class=\"muted\">No linked items available.</p>"
    return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"


def _html_report(
    report: dict[str, Any],
    *,
    shap_chart_name: str,
    graph_path_name: str,
) -> str:
    prediction = report.get("prediction", {})
    label = prediction.get("label")
    confidence = float(prediction.get("confidence", 0.0))
    band, band_class = _risk_band(label if isinstance(label, int) else None)
    severity_meaning = _severity_meaning(label if isinstance(label, int) else None)
    positive = report.get("shap", {}).get("positive_factors", [])
    negative = report.get("shap", {}).get("negative_factors", [])
    concepts = report.get("mapped_concepts", [])
    recommendations = report.get("recommendations", [])

    concept_cards = "".join(
        "<div class=\"chip\">"
        f"<strong>{escape(concept.get('name', ''))}</strong>"
        f"<span>{escape(concept.get('label', ''))} via {escape(concept.get('matched_alias', ''))}</span>"
        "</div>"
        for concept in concepts
    ) or "<p class=\"muted\">No graph concepts were mapped.</p>"

    def factor_rows(factors: list[dict]) -> str:
        if not factors:
            return "<p class=\"muted\">No factors in this direction.</p>"
        return "".join(
            "<div class=\"factor-row\">"
            f"<span>{escape(str(factor.get('token', '')))}</span>"
            f"<strong>{float(factor.get('value', 0.0)):.4f}</strong>"
            "</div>"
            for factor in factors
        )

    recommendation_cards = []
    for rec in recommendations:
        chunks = []
        for chunk in rec.get("evidence_chunks", []):
            url = escape(str(chunk.get("url", "")))
            source = escape(str(chunk.get("document_name", "Evidence document")))
            section = escape(str(chunk.get("section_title", "Section")))
            text = escape(str(chunk.get("text", "")))
            link = f"<a href=\"{url}\" target=\"_blank\" rel=\"noreferrer\">Open source</a>" if url else ""
            chunks.append(
                "<article class=\"evidence-chunk\">"
                f"<h4>{source}</h4>"
                f"<p class=\"muted\">{section} | {escape(str(chunk.get('chunk_id', '')))}</p>"
                f"<p>{text}</p>"
                f"{link}"
                "</article>"
            )
        evidence_names = [
            str(item.get("name", "")) for item in rec.get("evidence", []) if item
        ]
        resource_names = [
            str(item.get("name", "")) for item in rec.get("resources", []) if item
        ]
        recommendation_cards.append(
            "<section class=\"recommendation-card\">"
            "<div class=\"rec-head\">"
            f"<h3>{escape(str(rec.get('name', 'Recommendation')))}</h3>"
            f"<span class=\"score\">Score {float(rec.get('score', 0.0)):.2f}</span>"
            "</div>"
            f"<p>{escape(str(rec.get('description', '')))}</p>"
            "<div class=\"rec-grid\">"
            "<div><h4>Suggested steps</h4>"
            f"{_html_list([str(item) for item in rec.get('action_steps', [])])}</div>"
            "<div><h4>Support options</h4>"
            f"{_html_list([str(item) for item in rec.get('support_options', [])])}</div>"
            "</div>"
            "<div class=\"meta-row\">"
            f"<span>Concepts: {escape(', '.join(rec.get('concepts', [])) or 'Severity band')}</span>"
            f"<span>Evidence: {escape(', '.join(evidence_names) or 'Graph evidence')}</span>"
            f"<span>Resources: {escape(', '.join(resource_names) or 'None linked')}</span>"
            "</div>"
            "<div class=\"evidence-grid\">"
            f"{''.join(chunks)}"
            "</div>"
            "</section>"
        )

    rec_html = "".join(recommendation_cards) or (
        "<p class=\"muted\">No evidence-backed recommendations were returned.</p>"
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Clinical Decision Support Report</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #667085;
      --line: #d9e2ec;
      --surface: #ffffff;
      --wash: #f5f7fb;
      --blue: #2563eb;
      --green: #047857;
      --amber: #b45309;
      --red: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--wash);
      line-height: 1.5;
    }}
    .clinical-report-shell {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: start;
      margin-bottom: 22px;
    }}
    h1, h2, h3, h4 {{ margin: 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; margin-bottom: 14px; }}
    h3 {{ font-size: 17px; }}
    h4 {{ font-size: 14px; margin-bottom: 8px; }}
    .disclaimer {{ color: var(--muted); max-width: 760px; }}
    .risk-badge {{
      min-width: 190px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      text-align: right;
    }}
    .risk-badge strong {{ display: block; font-size: 22px; }}
    .risk-badge.low strong {{ color: var(--green); }}
    .risk-badge.moderate strong {{ color: var(--amber); }}
    .risk-badge.high strong, .risk-badge.urgent strong {{ color: var(--red); }}
    .section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .visual-grid, .factor-grid, .rec-grid, .evidence-grid {{
      display: grid;
      gap: 14px;
    }}
    .visual-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .factor-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .rec-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .visual-grid img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .factor-row {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding: 8px 0;
      gap: 12px;
    }}
    .chip {{
      display: inline-flex;
      flex-direction: column;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      margin: 0 8px 8px 0;
      background: #f8fafc;
    }}
    .chip span, .muted {{ color: var(--muted); font-size: 13px; }}
    .method-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .method-strip span {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      padding: 10px;
      font-size: 13px;
      text-align: center;
      font-weight: 700;
    }}
    .severity-note {{
      border-left: 4px solid var(--blue);
      padding: 10px 12px;
      background: #f8fafc;
      border-radius: 6px;
      margin-top: 10px;
    }}
    .trace-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
    }}
    .trace-row span {{
      background: #eef2ff;
      border: 1px solid #c7d2fe;
      border-radius: 999px;
      padding: 5px 9px;
    }}
    .trace-row b {{ color: var(--muted); font-size: 13px; }}
    .recommendation-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 14px;
      background: #fcfdff;
    }}
    .rec-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .score {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      color: var(--blue);
      background: #eff6ff;
      white-space: nowrap;
      font-size: 13px;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }}
    .meta-row span {{
      background: #eef2f7;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
    }}
    .evidence-chunk {{
      border-left: 4px solid var(--blue);
      padding: 10px 12px;
      background: #f8fafc;
      border-radius: 6px;
    }}
    a {{ color: var(--blue); font-weight: 600; }}
    @media (max-width: 800px) {{
      header, .visual-grid, .factor-grid, .rec-grid {{ grid-template-columns: 1fr; }}
      .risk-badge {{ text-align: left; }}
      .clinical-report-shell {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <main class=\"clinical-report-shell\">
    <header>
      <div>
        <h1>Clinical Decision Support Report</h1>
        <p class=\"disclaimer\">{escape(str(report.get("disclaimer", DISCLAIMER)))}</p>
        <div class=\"method-strip\" aria-label=\"Pipeline method\">
          <span>MentalBERT</span>
          <span>XAI factors</span>
          <span>Neo4j graph</span>
          <span>Evidence retrieval</span>
        </div>
      </div>
      <aside class=\"risk-badge {band_class}\">
        <span>Severity label {escape(str(label))}</span>
        <strong>{escape(band)}</strong>
        <span>{confidence:.2%} confidence</span>
      </aside>
    </header>

    <section class=\"section\">
      <h2>MentalBERT + XAI + Neo4j Summary</h2>
      <p class=\"severity-note\">{escape(severity_meaning)}</p>
      <p class=\"muted\">The model prediction is interpreted through token-level attribution, mapped to graph concepts, then connected to evidence-backed interventions through Neo4j traversal.</p>
    </section>

    <section class=\"section\">
      <h2>Visual Summary</h2>
      <div class=\"visual-grid\">
        <img src=\"{escape(shap_chart_name)}\" alt=\"SHAP contribution chart\">
        <img src=\"{escape(graph_path_name)}\" alt=\"Explanation graph path\">
      </div>
    </section>

    <section class=\"section\">
      <h2>Important Factors</h2>
      <div class=\"factor-grid\">
        <div><h3>Positive</h3>{factor_rows(positive)}</div>
        <div><h3>Negative</h3>{factor_rows(negative)}</div>
      </div>
    </section>

    <section class=\"section\">
      <h2>Mapped Graph Concepts</h2>
      {concept_cards}
    </section>

    <section class=\"section\">
      <h2>Decision Trace</h2>
      {_decision_trace_html(report)}
    </section>

    <section class=\"section\">
      <h2>Evidence-Backed Recommendations</h2>
      {rec_html}
    </section>
  </main>
</body>
</html>
"""


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Write report JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_clinical_report.json"
    markdown_path = output_dir / f"{run_id}_clinical_report.md"
    html_path = output_dir / f"{run_id}_clinical_report.html"
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
    html_path.write_text(
        _html_report(
            report,
            shap_chart_name=shap_chart_path.name,
            graph_path_name=graph_path_path.name,
        ),
        encoding="utf-8",
    )
    return {
        "json": json_path,
        "markdown": markdown_path,
        "html": html_path,
        "shap_chart": shap_chart_path,
        "graph_path": graph_path_path,
    }
