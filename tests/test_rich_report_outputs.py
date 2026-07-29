from knowledge_graph.clinical_report import build_report
from knowledge_graph.graph_builder import build_seed_statements
from knowledge_graph.recommendation_engine import rank_recommendations
from knowledge_graph.seed_data import SEED_EVIDENCE_DOCUMENTS
from knowledge_graph.schema import Recommendation
from xai.shap_explainer import PredictionResult


def test_seed_data_contains_document_chunks():
    who = next(doc for doc in SEED_EVIDENCE_DOCUMENTS if doc["name"].startswith("WHO"))
    assert who["sections"]
    assert who["sections"][0]["chunks"]
    assert who["sections"][0]["chunks"][0]["chunk_id"]


def test_seed_statements_create_document_section_chunk_nodes():
    statements = build_seed_statements()
    cypher = "\n".join(statement for statement, _ in statements)
    assert "EvidenceDocument" in cypher
    assert "EvidenceSection" in cypher
    assert "EvidenceChunk" in cypher
    assert "HAS_CHUNK" in cypher


def test_rank_recommendations_carries_steps_support_options_and_chunks():
    rows = [
        {
            "intervention": "Peer Support",
            "intervention_props": {
                "name": "Peer Support",
                "description": "Connection with supportive peers.",
                "action_steps": ["Send one short check-in message."],
                "support_options": ["Text message", "Audio call"],
            },
            "concept": "Isolation",
            "shap_value": 0.5,
            "severity_matched": True,
            "evidence": {"name": "WHO Suicide Prevention Guidance"},
            "chunk": {
                "chunk_id": "who-1",
                "text": "Support groups and health workers can help.",
                "section_title": "What can help",
                "document_name": "WHO Suicide Q&A",
                "url": "https://www.who.int/news-room/questions-and-answers/item/suicide",
            },
            "resource": {"name": "Support Group"},
        }
    ]
    rec = rank_recommendations(rows, top_k=1)[0]
    assert rec.action_steps == ["Send one short check-in message."]
    assert "Audio call" in rec.support_options
    assert rec.evidence_chunks[0]["chunk_id"] == "who-1"


def test_markdown_report_includes_detailed_support_sections(tmp_path):
    report = build_report(
        input_preview="I feel alone",
        prediction=PredictionResult(label=1, confidence=0.7, probabilities=[0.3, 0.7]),
        positive_factors=[],
        negative_factors=[],
        concepts=[],
        recommendations=[
            Recommendation(
                name="Peer Support",
                score=1.0,
                concepts=["Isolation"],
                evidence=[{"name": "WHO Suicide Prevention Guidance"}],
                resources=[{"name": "Support Group"}],
                description="Connection with supportive peers.",
                action_steps=["Send one short check-in message."],
                support_options=["Text message", "Audio call", "Video call"],
                evidence_chunks=[
                    {
                        "chunk_id": "who-1",
                        "text": "Support groups and health workers can help.",
                        "document_name": "WHO Suicide Q&A",
                        "section_title": "What can help",
                        "url": "https://www.who.int/news-room/questions-and-answers/item/suicide",
                    }
                ],
            )
        ],
    )
    assert report["recommendations"][0]["action_steps"]
    assert report["recommendations"][0]["evidence_chunks"]
