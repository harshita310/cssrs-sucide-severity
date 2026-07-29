"""Evidence and resource retrieval from Neo4j traversal rows."""

from __future__ import annotations

from .graph_query import fetch_interventions, fetch_severity_resources
from .schema import MappedConcept


def retrieve_intervention_rows(
    session,
    *,
    concepts: list[MappedConcept],
    severity: int,
) -> list[dict]:
    """Retrieve intervention/evidence/resource rows for mapped concepts."""
    concept_names = sorted({concept.name for concept in concepts})
    concept_values: dict[str, float] = {}
    for concept in concepts:
        concept_values[concept.name] = max(
            abs(float(concept.shap_value)),
            concept_values.get(concept.name, 0.0),
        )

    return fetch_interventions(
        session,
        concept_names=concept_names,
        concept_values=concept_values,
        severity=severity,
    )


def retrieve_severity_resources(session, *, severity: int) -> list[dict]:
    """Retrieve resources directly linked to a predicted severity level."""
    return fetch_severity_resources(session, severity)
