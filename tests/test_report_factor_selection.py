from knowledge_graph.schema import ShapTokenFactor
from scripts.generate_clinical_report import report_mapping_factors


def test_report_mapping_factors_uses_negative_concept_tokens_and_filters_stopwords():
    factors = report_mapping_factors(
        positive=[
            ShapTokenFactor("I", 0.45, 1, "positive"),
            ShapTokenFactor("and I", 0.30, 2, "positive"),
            ShapTokenFactor("feel alone", 0.20, 3, "positive"),
            ShapTokenFactor("here", 0.01, 2, "positive"),
        ],
        negative=[
            ShapTokenFactor("alone", -0.20, 1, "negative"),
            ShapTokenFactor("and", -0.10, 2, "negative"),
            ShapTokenFactor("cannot sleep", -0.08, 3, "negative"),
            ShapTokenFactor("sleep", -0.07, 4, "negative"),
            ShapTokenFactor("cannot", -0.06, 5, "negative"),
        ],
    )
    tokens = [factor.token for factor in factors]
    assert "alone" in tokens
    assert "cannot sleep" in tokens
    assert "I" not in tokens
    assert "and I" not in tokens
    assert "feel alone" not in tokens
    assert "and" not in tokens
    assert "sleep" not in tokens
    assert "cannot" not in tokens
