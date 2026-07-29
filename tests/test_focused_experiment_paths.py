from pathlib import Path

from scripts.run_focused_experiments import build_run_paths


def test_build_run_paths_keeps_focused_artifacts_isolated():
    paths = build_run_paths(Path("ROOT"), "focused_test")
    assert paths["model_path"] == Path(
        "ROOT/saved_model/focused_experiments/best_model_focused_test.pt"
    )
    assert paths["optimizer_path"] == Path(
        "ROOT/saved_model/focused_experiments/optimizer_focused_test.pt"
    )
    assert paths["metrics_dir"] == Path(
        "ROOT/RESULTS/metrics/focused_experiments/focused_test"
    )
    assert paths["plots_dir"] == Path(
        "ROOT/RESULTS/plots/focused_experiments/focused_test"
    )
