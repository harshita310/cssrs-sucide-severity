from knowledge_graph.schema import MappedConcept, Recommendation, ShapTokenFactor
from knowledge_graph.text import normalize_key


def test_normalize_key_lowercases_and_removes_noise():
    assert normalize_key(" Can't sleep!! ") == "cant sleep"


def test_shap_token_factor_requires_direction():
    factor = ShapTokenFactor(
        token="hopeless",
        value=0.42,
        rank=1,
        direction="positive",
    )
    assert factor.direction == "positive"


def test_recommendation_schema_holds_traceable_evidence():
    rec = Recommendation(
        name="Safety Planning",
        score=2.5,
        concepts=["Hopelessness"],
        evidence=[
            {
                "name": "WHO guideline",
                "passage": "Safety planning is evidence linked.",
            }
        ],
        resources=[{"name": "Emergency services"}],
    )
    assert rec.evidence[0]["name"] == "WHO guideline"
