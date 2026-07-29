from knowledge_graph.graph_query import concept_query, intervention_query
from knowledge_graph.recommendation_engine import rank_recommendations


def test_concept_query_uses_alias_lookup():
    cypher = concept_query()
    assert "Alias" in cypher
    assert "ALIAS_OF" in cypher
    assert "$keys" in cypher


def test_intervention_query_uses_concepts_and_severity():
    cypher = intervention_query()
    assert "SUPPORTED_BY" in cypher
    assert "SeverityBand" in cypher
    assert "$concept_names" in cypher
    assert "$severity" in cypher


def test_rank_recommendations_prefers_more_evidence_and_higher_shap():
    rows = [
        {
            "intervention": "Peer Support",
            "concept": "Isolation",
            "shap_value": 0.4,
            "evidence": {"name": "Evidence A"},
            "resource": {"name": "Group"},
        },
        {
            "intervention": "Safety Planning",
            "concept": "Hopelessness",
            "shap_value": 0.9,
            "evidence": {"name": "Evidence B"},
            "resource": {"name": "Professional"},
        },
        {
            "intervention": "Safety Planning",
            "concept": "Self Harm",
            "shap_value": 0.7,
            "evidence": {"name": "Evidence C"},
            "resource": {"name": "Helpline"},
        },
    ]
    ranked = rank_recommendations(rows, top_k=2)
    assert ranked[0].name == "Safety Planning"
    assert ranked[0].score > ranked[1].score
