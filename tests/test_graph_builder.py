from knowledge_graph.graph_builder import build_seed_statements
from knowledge_graph.seed_data import SEED_CONCEPTS, SEED_EVIDENCE_DOCUMENTS


def test_seed_data_contains_required_clinical_concepts():
    names = {item["name"] for item in SEED_CONCEPTS}
    assert {"Hopelessness", "Worthlessness", "Isolation", "Insomnia", "Self Harm"} <= names


def test_build_seed_statements_returns_parameterized_cypher():
    statements = build_seed_statements()
    assert statements
    assert all("$" in cypher for cypher, _ in statements)
    assert any("MERGE (n:Intervention" in cypher for cypher, _ in statements)


def test_seed_documents_include_multiple_source_families():
    source_types = {item["source_type"] for item in SEED_EVIDENCE_DOCUMENTS}
    assert {"WHO", "NICE Guideline", "CDC", "SAMHSA"} <= source_types
