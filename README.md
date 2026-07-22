# MentalBERT-CSSR

Fine-tuning MentalBERT for suicide severity classification with C-SSRS labels.

This project trains `mental/mental-bert-base-uncased` on Reddit post text from
the `content` column and predicts human severity labels in `severity` (`0`-`6`).
LLM label columns are kept only for later benchmarking and are never used as
training targets.

> Ethical note: this is a research classification pipeline, not a clinical
> diagnostic or crisis-intervention system.

## Project Structure

```text
.
|-- configs/
|   `-- default.yaml
|-- DATA/
|   |-- RAW/
|   |   `-- human_n_llm_labeled_rSuicidewatch_posts.csv
|   `-- processed/
|       |-- cssrs_processed.csv
|       `-- splits/
|           |-- train.csv
|           |-- val.csv
|           `-- test.csv
|-- NOTEBOOKS/
|-- RESULTS/
|   |-- metrics/
|   `-- plots/
|-- docs/
|   `-- assets/
|-- saved_model/
|-- scripts/
|   |-- run_pipeline.py
|   |-- run_training.py
|   |-- run_evaluation.py
|   `-- compare_checkpoints.py
|-- utils/
|-- README.md
`-- requirements.txt
```

## Setup

Use Python 3.11+ and install the project requirements:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

The pipeline uses CUDA automatically when available. The MentalBERT model is
gated on Hugging Face, so accept the model terms and log in if the weights are
not already cached locally.

## Main Pipeline

Train and then evaluate the best checkpoint:

```bash
python scripts/run_pipeline.py
```

Evaluate the configured checkpoint without retraining:

```bash
python scripts/run_pipeline.py --skip-training
```

Evaluate a specific checkpoint:

```bash
python scripts/run_evaluation.py --model-path saved_model/tune/best_model_Run_4_Vanilla_CE.pt
```

Compare all tuned checkpoints against validation and test splits:

```bash
python scripts/compare_checkpoints.py --include-current
```

## Current Best Training Recipe

The strongest saved sweep run is `Run_4_Vanilla_CE`:

- Loss: cross entropy
- Learning rate: `1.5e-5`
- Dropout: `0.15`
- Weight decay: `0.01`
- Label smoothing: `0.0`
- Class weights: disabled
- Weighted sampler: disabled
- Best epoch: `16`

Earlier tuning showed this recipe produced the best validation accuracy,
macro F1, and weighted F1 among the saved sweep runs.

## Best Results

Best checkpoint: `saved_model/tune/best_model_Run_4_Vanilla_CE.pt`

| Split | Accuracy | Macro F1 | Weighted F1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Train | 99.88% | 99.70% | 99.88% | 0.0112 |
| Validation | 72.73% | 64.17% | 72.08% | 1.4932 |
| Test | 70.45% | 61.77% | 70.74% | 1.5398 |

### Training Curves

![Validation accuracy](docs/assets/validation_accuracy.png)

![Macro F1](docs/assets/macro_f1.png)

![Weighted F1](docs/assets/weighted_f1.png)

### Confusion Matrices

![Validation confusion matrix](docs/assets/validation_confusion_matrix.png)

![Test confusion matrix](docs/assets/test_confusion_matrix.png)

## Dataset Schema

| Column | Role |
| --- | --- |
| `content` | Model input text |
| `severity` | Human ground-truth label (`0`-`6`) |
| `gpt_label`, `claude_label`, `gemini_label`, `llama_label`, `mistral_label` | Benchmark only |
| `url`, `author`, `created` | Metadata |

## Outputs

Training writes checkpoints to `saved_model/` and metrics/plots to `RESULTS/`.
Evaluation writes split summaries, reports, confusion matrices, and predictions
under `RESULTS/metrics/evaluation/` by default.

Dataset CSVs, checkpoints, and generated full result folders are intentionally
ignored by Git to avoid publishing sensitive data or large artifacts.
