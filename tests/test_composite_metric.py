from utils.trainer import compute_composite_score


def test_compute_composite_score_uses_named_metric_weights():
    metrics = {
        "weighted_f1": 0.72,
        "macro_f1": 0.64,
        "quadratic_weighted_kappa": 0.88,
    }
    score = compute_composite_score(
        metrics,
        {
            "weighted_f1": 0.4,
            "macro_f1": 0.3,
            "quadratic_weighted_kappa": 0.3,
        },
    )
    assert score == (0.72 * 0.4) + (0.64 * 0.3) + (0.88 * 0.3)
