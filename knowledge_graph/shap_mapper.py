"""Map SHAP token factors to Neo4j clinical concepts."""

from __future__ import annotations

from .graph_query import fetch_concepts
from .schema import MappedConcept, ShapTokenFactor
from .text import normalize_key


def map_shap_tokens(session, factors: list[ShapTokenFactor]) -> list[MappedConcept]:
    """Map SHAP tokens to graph concepts through Neo4j Alias nodes."""
    key_to_factor: dict[str, ShapTokenFactor] = {}
    for factor in factors:
        key = normalize_key(factor.token)
        if key and key not in key_to_factor:
            key_to_factor[key] = factor

    if not key_to_factor:
        return []

    rows = fetch_concepts(session, list(key_to_factor))
    concepts: list[MappedConcept] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        alias_key = str(row["alias_key"])
        factor = key_to_factor.get(alias_key)
        if factor is None:
            continue
        name = str(row["concept_name"])
        label = str(row["concept_label"])
        dedupe_key = (name, alias_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        concepts.append(
            MappedConcept(
                name=name,
                label=label,
                matched_alias=alias_key,
                shap_value=float(factor.value),
            )
        )

    return concepts
