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
| Baseline CNN | TBD | TBD | TBD |
| ResNet50 (fine-tuned) | TBD | TBD | TBD |
| EfficientNetB0 (fine-tuned) | TBD | TBD | TBD |

Fill in after running `src/train/train.py` for each model.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle Brain Tumor MRI Dataset into `data/raw/`, organized as
`data/raw/<class_name>/*.jpg` for each of glioma, meningioma, pituitary,
no_tumor.

Check class balance before training:
```bash
python src/data/dataset.py
```

Train a model:
```bash
python src/train/train.py
```

Run the API and frontend:
```bash
docker compose up -d
streamlit run frontend/app.py
```

## Project structure

```
src/data/       Dataset loading, class balance check, augmentation
src/models/     Baseline CNN, ResNet50, EfficientNetB0 architectures
src/train/      k-fold cross-validation training loop
src/eval/       Metrics (including false negative rate) and Grad-CAM
src/api/        FastAPI serving layer with Grad-CAM overlay
frontend/       Streamlit demo UI
docs/           Build plan and model card
```

See `docs/BUILD_PLAN.md` for the full week-by-week plan and reasoning
behind each design decision.

## Status

- [ ] Dataset downloaded, class balance documented
- [ ] Baseline CNN trained
- [ ] ResNet50 and EfficientNetB0 fine-tuned
- [ ] k-fold cross-validation run for all three models
- [ ] Full evaluation suite (confusion matrix, per-class F1, ROC-AUC, false negative rate)
- [ ] Grad-CAM examples generated and reviewed
- [ ] FastAPI + Streamlit demo working end to end
- [ ] Model card completed
- [ ] Deployed on AWS
