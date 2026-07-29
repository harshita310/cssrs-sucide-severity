"""Reusable Neo4j Cypher queries for XAI graph traversal."""

from __future__ import annotations


def concept_query() -> str:
    """Return Cypher that maps normalized alias keys to graph concepts."""
    return """
    MATCH (a:Alias)-[:ALIAS_OF]->(c)
    WHERE a.key IN $keys
    RETURN
        a.key AS alias_key,
        c.name AS concept_name,
        labels(c)[0] AS concept_label
    """


def intervention_query() -> str:
    """Return Cypher for concept and severity linked interventions."""
    return """
    MATCH (c)
    WHERE c.name IN $concept_names
    MATCH (c)-[:TREATED_BY|BENEFITS_FROM]->(i:Intervention)
    OPTIONAL MATCH (s:SeverityBand {level: $severity})-[:RECOMMENDS]->(i)
    OPTIONAL MATCH (i)-[:SUPPORTED_BY]->(e:EvidenceSource)
    OPTIONAL MATCH (chunk:EvidenceChunk)-[:SUPPORTS]->(i)
    OPTIONAL MATCH (i)-[:USES_RESOURCE]->(r:Resource)
    RETURN
        i.name AS intervention,
        properties(i) AS intervention_props,
        c.name AS concept,
        $concept_values[c.name] AS shap_value,
        CASE WHEN s IS NULL THEN false ELSE true END AS severity_matched,
        CASE WHEN e IS NULL THEN NULL ELSE properties(e) END AS evidence,
        CASE WHEN chunk IS NULL THEN NULL ELSE properties(chunk) END AS chunk,
        CASE WHEN r IS NULL THEN NULL ELSE properties(r) END AS resource
    UNION
    MATCH (s:SeverityBand {level: $severity})-[:RECOMMENDS]->(i:Intervention)
    OPTIONAL MATCH (i)-[:SUPPORTED_BY]->(e:EvidenceSource)
    OPTIONAL MATCH (chunk:EvidenceChunk)-[:SUPPORTS]->(i)
    OPTIONAL MATCH (i)-[:USES_RESOURCE]->(r:Resource)
    RETURN
        i.name AS intervention,
        properties(i) AS intervention_props,
        NULL AS concept,
        0.0 AS shap_value,
        true AS severity_matched,
        CASE WHEN e IS NULL THEN NULL ELSE properties(e) END AS evidence,
        CASE WHEN chunk IS NULL THEN NULL ELSE properties(chunk) END AS chunk,
        CASE WHEN r IS NULL THEN NULL ELSE properties(r) END AS resource
    """


def resource_query() -> str:
    """Return Cypher for resources directly recommended by severity."""
    return """
    MATCH (s:SeverityBand {level: $severity})-[:RECOMMENDS]->(r:Resource)
    RETURN properties(r) AS resource
    """


def fetch_concepts(session, keys: list[str]) -> list[dict]:
    result = session.run(concept_query(), {"keys": keys})
    return [dict(record) for record in result]


def fetch_interventions(
    session,
    *,
    concept_names: list[str],
    concept_values: dict[str, float],
    severity: int,
) -> list[dict]:
    result = session.run(
        intervention_query(),
        {
            "concept_names": concept_names,
            "concept_values": concept_values,
            "severity": int(severity),
        },
    )
    return [dict(record) for record in result]


def fetch_severity_resources(session, severity: int) -> list[dict]:
    result = session.run(resource_query(), {"severity": int(severity)})
    return [dict(record) for record in result]
