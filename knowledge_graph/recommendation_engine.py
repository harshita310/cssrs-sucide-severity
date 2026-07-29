"""Deterministic recommendation ranking for graph traversal outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import Recommendation


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        if not item:
            continue
        key = str(item.get("name", item))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def rank_recommendations(rows: list[dict], top_k: int) -> list[Recommendation]:
    """Rank intervention rows using SHAP strength and traceability."""
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "concepts": set(),
            "shap_total": 0.0,
            "evidence": [],
            "chunks": [],
            "resources": [],
            "severity_matched": False,
            "props": {},
        }
    )

    for row in rows:
        intervention = row.get("intervention")
        if not intervention:
            continue
        item = grouped[str(intervention)]
        if row.get("intervention_props"):
            item["props"].update(dict(row["intervention_props"]))
        concept = row.get("concept")
        if concept:
            item["concepts"].add(str(concept))
            item["shap_total"] += abs(float(row.get("shap_value") or 0.0))
        if row.get("severity_matched"):
            item["severity_matched"] = True
        if row.get("evidence"):
            item["evidence"].append(dict(row["evidence"]))
        if row.get("chunk"):
            item["chunks"].append(dict(row["chunk"]))
        if row.get("resource"):
            item["resources"].append(dict(row["resource"]))

    recommendations: list[Recommendation] = []
    for name, item in grouped.items():
        evidence = _dedupe_dicts(item["evidence"])
        chunks = _dedupe_dicts(item["chunks"])
        resources = _dedupe_dicts(item["resources"])
        if not evidence and not chunks:
            continue
        severity_bonus = 0.5 if item["severity_matched"] else 0.0
        score = (
            float(item["shap_total"])
            + (0.25 * len(evidence))
            + (0.20 * len(chunks))
            + (0.15 * len(resources))
            + severity_bonus
        )
        props = item["props"]
        recommendations.append(
            Recommendation(
                name=name,
                score=score,
                concepts=sorted(item["concepts"]),
                evidence=evidence,
                resources=resources,
                description=str(props.get("description", "")),
                action_steps=list(props.get("action_steps", []) or []),
                support_options=list(props.get("support_options", []) or []),
                evidence_chunks=chunks,
            )
        )

    recommendations.sort(key=lambda rec: (-rec.score, rec.name))
    return recommendations[:top_k]
