from pathlib import Path

from scripts.run_ordinal_experiments import build_run_paths


def test_build_run_paths_keeps_ordinal_artifacts_isolated():
    paths = build_run_paths(Path("ROOT"), "ordinal_ce_dw_0_20")
    assert paths["model_path"] == Path(
        "ROOT/saved_model/ordinal_experiments/best_model_ordinal_ce_dw_0_20.pt"
    )
    assert paths["optimizer_path"] == Path(
        "ROOT/saved_model/ordinal_experiments/optimizer_ordinal_ce_dw_0_20.pt"
    )
    assert paths["metrics_dir"] == Path(
        "ROOT/RESULTS/metrics/ordinal_experiments/ordinal_ce_dw_0_20"
    )
    assert paths["plots_dir"] == Path(
        "ROOT/RESULTS/plots/ordinal_experiments/ordinal_ce_dw_0_20"
    )
