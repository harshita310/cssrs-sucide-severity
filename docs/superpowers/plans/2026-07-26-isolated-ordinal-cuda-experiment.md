# Isolated Ordinal CUDA Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an ordinal-aware MentalBERT experiment on CUDA without overwriting the current best metrics, plots, or checkpoints.

**Architecture:** Add only the missing wiring and a separate experiment runner. The default pipeline remains intact, while ordinal runs write to `saved_model/ordinal_experiments/`, `RESULTS/metrics/ordinal_experiments/`, and `RESULTS/plots/ordinal_experiments/`.

**Tech Stack:** Python, PyTorch, Transformers, sklearn, pandas, existing project utilities.

## Global Constraints

- Keep all severity labels as seven classes: `0` through `6`.
- Do not overwrite current best checkpoint `saved_model/tune/best_model_Run_4_Vanilla_CE.pt`.
- Do not overwrite current canonical evaluation folder `RESULTS/metrics/evaluation`.
- Require CUDA before training; fail fast if CUDA is unavailable.
- Compare new results against current best test accuracy `0.7045`, macro F1 `0.6177`, and weighted F1 `0.7074`.

---

### Task 1: Wire Ordinal Loss Through TrainerConfig

**Files:**
- Test: `tests/test_losses.py`
- Modify: `utils/trainer.py`

**Interfaces:**
- Consumes: `utils.losses.build_criterion(loss_type="ordinal_ce", ordinal_distance_weight=float, num_labels=int)`
- Produces: `TrainerConfig.ordinal_distance_weight: float`

- [ ] **Step 1: Write failing test**

```python
from utils.losses import OrdinalCrossEntropyLoss, build_criterion
from utils.trainer import TrainerConfig


def test_trainer_config_exposes_ordinal_distance_weight():
    cfg = TrainerConfig(ordinal_distance_weight=0.25)
    assert cfg.ordinal_distance_weight == 0.25


def test_build_criterion_returns_ordinal_ce_with_distance_weight():
    criterion = build_criterion(
        loss_type="ordinal_ce",
        ordinal_distance_weight=0.25,
        num_labels=7,
    )
    assert isinstance(criterion, OrdinalCrossEntropyLoss)
    assert criterion.distance_weight == 0.25
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_losses.py -q`

- [ ] **Step 3: Add minimal trainer wiring**

Add `ordinal_distance_weight` to `TrainerConfig` and pass it into `build_criterion(...)`.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_losses.py -q`

### Task 2: Add Isolated Ordinal Experiment Runner

**Files:**
- Test: `tests/test_ordinal_experiment_paths.py`
- Create: `scripts/run_ordinal_experiments.py`

**Interfaces:**
- Produces: `build_run_paths(project_root: Path, run_name: str) -> dict[str, Path]`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path

from scripts.run_ordinal_experiments import build_run_paths


def test_build_run_paths_keeps_ordinal_artifacts_isolated():
    paths = build_run_paths(Path("ROOT"), "ordinal_ce_dw_0_20")
    assert paths["model_path"] == Path("ROOT/saved_model/ordinal_experiments/best_model_ordinal_ce_dw_0_20.pt")
    assert paths["optimizer_path"] == Path("ROOT/saved_model/ordinal_experiments/optimizer_ordinal_ce_dw_0_20.pt")
    assert paths["metrics_dir"] == Path("ROOT/RESULTS/metrics/ordinal_experiments/ordinal_ce_dw_0_20")
    assert paths["plots_dir"] == Path("ROOT/RESULTS/plots/ordinal_experiments/ordinal_ce_dw_0_20")
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ordinal_experiment_paths.py -q`

- [ ] **Step 3: Implement runner**

Create `run_ordinal_experiments.py` by reusing the existing training utilities with hard-coded focused configs and CUDA-only validation.

- [ ] **Step 4: Run tests and syntax checks**

Run:
- `.venv\Scripts\python.exe -m pytest tests -q`
- `.venv\Scripts\python.exe -m py_compile scripts/run_ordinal_experiments.py`

### Task 3: Run And Compare

**Files:**
- Generated only under ignored ordinal experiment folders.

- [ ] **Step 1: Verify CUDA**

Run: `.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"`

- [ ] **Step 2: Run isolated ordinal sweep**

Run: `.venv\Scripts\python.exe scripts/run_ordinal_experiments.py`

- [ ] **Step 3: Evaluate comparison**

Compare generated summary against current best and do not update README unless new metrics are better.
