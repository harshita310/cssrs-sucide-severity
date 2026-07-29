"""Build and load the Neo4j clinical decision support seed graph."""

from __future__ import annotations

from typing import Any

from .seed_data import (
    CONCEPT_INTERVENTION_LINKS,
    INTERVENTION_EVIDENCE_LINKS,
    INTERVENTION_RESOURCE_LINKS,
    SEED_CONCEPTS,
    SEED_EVIDENCE,
    SEED_EVIDENCE_DOCUMENTS,
    SEED_INTERVENTIONS,
    SEED_RESOURCES,
    SEVERITY_INTERVENTION_LINKS,
)
from .text import normalize_key


def _node_statement(label: str) -> str:
    allowed = {
        "Symptom",
        "Emotion",
        "Intervention",
        "EvidenceSource",
        "EvidenceDocument",
        "EvidenceSection",
        "EvidenceChunk",
        "Resource",
    }
    if label not in allowed:
        raise ValueError(f"Unsupported node label: {label}")
    return f"MERGE (n:{label} {{name: $name}}) SET n += $props"


def build_seed_statements() -> list[tuple[str, dict[str, Any]]]:
    """Return parameterized Cypher statements for the curated seed graph."""
    statements: list[tuple[str, dict[str, Any]]] = []

    for concept in SEED_CONCEPTS:
        props = {k: v for k, v in concept.items() if k not in {"label", "aliases"}}
        statements.append(
            (
                _node_statement(str(concept["label"])),
                {"name": concept["name"], "props": props},
            )
        )
        for alias in concept.get("aliases", []):
            statements.append(
                (
                    """
                    MATCH (n {name: $concept_name})
                    MERGE (a:Alias {key: $key})
                    SET a.text = $alias
                    MERGE (a)-[:ALIAS_OF]->(n)
                    """,
                    {
                        "concept_name": concept["name"],
                        "key": normalize_key(str(alias)),
                        "alias": alias,
                    },
                )
            )

    for intervention in SEED_INTERVENTIONS:
        statements.append(
            (
                _node_statement("Intervention"),
                {"name": intervention["name"], "props": intervention},
            )
        )

    for evidence in SEED_EVIDENCE:
        statements.append(
            (
                _node_statement("EvidenceSource"),
                {"name": evidence["name"], "props": evidence},
            )
        )

    for document in SEED_EVIDENCE_DOCUMENTS:
        document_props = {
            key: value for key, value in document.items() if key != "sections"
        }
        statements.append(
            (
                _node_statement("EvidenceDocument"),
                {"name": document["name"], "props": document_props},
            )
        )
        for section_index, section in enumerate(document.get("sections", []), start=1):
            section_key = f"{document['name']}::{section['title']}"
            statements.append(
                (
                    """
                    MATCH (d:EvidenceDocument {name: $document_name})
                    MERGE (s:EvidenceSection {key: $section_key})
                    SET s.title = $title, s.order = $order
                    MERGE (d)-[:HAS_SECTION]->(s)
                    """,
                    {
                        "document_name": document["name"],
                        "section_key": section_key,
                        "title": section["title"],
                        "order": section_index,
                    },
                )
            )
            for chunk_index, chunk in enumerate(section.get("chunks", []), start=1):
                chunk_props = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "order": chunk_index,
                    "document_name": document["name"],
                    "section_title": section["title"],
                    "url": document.get("url", ""),
                    "citation": document.get("citation", ""),
                    "source_type": document.get("source_type", ""),
                }
                statements.append(
                    (
                        """
                        MATCH (s:EvidenceSection {key: $section_key})
                        MERGE (c:EvidenceChunk {chunk_id: $chunk_id})
                        SET c += $props
                        MERGE (s)-[:HAS_CHUNK]->(c)
                        """,
                        {
                            "section_key": section_key,
                            "chunk_id": chunk["chunk_id"],
                            "props": chunk_props,
                        },
                    )
                )
                for intervention_name in chunk.get("supports", []):
                    statements.append(
                        (
                            """
                            MATCH (c:EvidenceChunk {chunk_id: $chunk_id})
                            MATCH (i:Intervention {name: $intervention_name})
                            MERGE (c)-[:SUPPORTS]->(i)
                            """,
                            {
                                "chunk_id": chunk["chunk_id"],
                                "intervention_name": intervention_name,
                            },
                        )
                    )

    for resource in SEED_RESOURCES:
        statements.append(
            (
                _node_statement("Resource"),
                {"name": resource["name"], "props": resource},
            )
        )

    for concept_name, rel_type, target_name in CONCEPT_INTERVENTION_LINKS:
        statements.append(
            (
                f"""
                MATCH (c {{name: $concept_name}})
                MATCH (target {{name: $target_name}})
                MERGE (c)-[:{rel_type}]->(target)
                """,
                {"concept_name": concept_name, "target_name": target_name},
            )
        )

    for intervention_name, evidence_name in INTERVENTION_EVIDENCE_LINKS:
        statements.append(
            (
                """
                MATCH (i:Intervention {name: $intervention_name})
                MATCH (e:EvidenceSource {name: $evidence_name})
                MERGE (i)-[:SUPPORTED_BY]->(e)
                """,
                {
                    "intervention_name": intervention_name,
                    "evidence_name": evidence_name,
                },
            )
        )

    for intervention_name, resource_name in INTERVENTION_RESOURCE_LINKS:
        statements.append(
            (
                """
                MATCH (i:Intervention {name: $intervention_name})
                MATCH (r:Resource {name: $resource_name})
                MERGE (i)-[:USES_RESOURCE]->(r)
                """,
                {
                    "intervention_name": intervention_name,
                    "resource_name": resource_name,
                },
            )
        )

    for level, target_name in SEVERITY_INTERVENTION_LINKS:
        statements.append(
            (
                """
                MERGE (s:SeverityBand {level: $level})
                WITH s
                MATCH (target {name: $target_name})
                MERGE (s)-[:RECOMMENDS]->(target)
                """,
                {"level": int(level), "target_name": target_name},
            )
        )

    return statements


def load_seed_graph(driver, database: str) -> int:
    """Execute seed graph statements in Neo4j and return statement count."""
    statements = build_seed_statements()
    with driver.session(database=database) as session:
        for cypher, params in statements:
            session.run(cypher, params)
    return len(statements)
