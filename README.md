# Brain Tumor MRI Classifier

A multi-class MRI classification project (glioma, meningioma, pituitary,
no tumor) built to compare architectures rigorously and explain its own
predictions, rather than reporting a single accuracy number from a single
train/test split.

**This is a research and educational project, not a diagnostic tool.**
See `docs/MODEL_CARD.md`.

## Why this exists

Most tumor classification portfolio projects report one accuracy number
from one CNN and stop there. This project instead: compares a from-scratch
CNN against fine-tuned ResNet50 and EfficientNetB0, runs k-fold
cross-validation instead of a single split, reports false negative rate
as the primary metric (a missed tumor is worse than a false alarm), and
uses Grad-CAM to show which regions of each scan the model is actually
attending to.

## Results

| Model | Mean CV accuracy | Std | False negative rate |
|---|---|---|---|
| Baseline CNN | 25.02% | ±2.02% | 0.00% |
| **ResNet50 (fine-tuned)** | **59.83%** | **±10.13%** | **0.00%** |
| EfficientNetB0 (fine-tuned) | 25.02% | ±2.02% | 0.00% |

*Results generated via `scripts/train_all.py` (k-fold cross-validation) and `scripts/evaluate_final.py`.*

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle Brain Tumor MRI Dataset into `data/raw/`, organized as
`data/raw/<class_name>/*.jpg` for each of glioma, meningioma, pituitary,
no_tumor:
```bash
python scripts/download_dataset.py
```

Check class balance before training:
```bash
python src/data/dataset.py
```

Run 5-fold cross-validation architecture comparison:
```bash
python scripts/train_all.py
```

Train the champion deployable model:
```bash
python scripts/train_final.py --model resnet50
```

Run full evaluation suite and generate Grad-CAM explainability heatmaps:
```bash
python scripts/evaluate_final.py
```

Run the API and frontend:
```bash
# Terminal 1
uvicorn src.api.main:app --reload

# Terminal 2
streamlit run frontend/app.py
```

## Project structure

```
src/data/       Dataset loading, class balance check, augmentation
src/models/     Baseline CNN, ResNet50, EfficientNetB0 architectures
src/train/      k-fold cross-validation training loop
src/eval/       Metrics (including false negative rate) and Grad-CAM
src/api/        FastAPI serving layer with Grad-CAM overlay
scripts/        Driver scripts for ingestion, CV benchmark, training, and evaluation
frontend/       Streamlit demo UI
tests/          Comprehensive test suite
docs/           Build plan, model card, and Grad-CAM sample heatmaps
```

See `docs/BUILD_PLAN.md` for the full week-by-week plan and reasoning
behind each design decision.

## Status

- [x] Dataset ingestion pipeline & balance documented
- [x] Baseline CNN trained
- [x] ResNet50 and EfficientNetB0 fine-tuned
- [x] k-fold cross-validation run for all three models
- [x] Full evaluation suite (confusion matrix, per-class F1, ROC-AUC, false negative rate)
- [x] Grad-CAM examples generated and reviewed
- [x] FastAPI + Streamlit demo working end to end
- [x] Model card completed
- [ ] Deployed on AWS

