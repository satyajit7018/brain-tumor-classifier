# Brain Tumor MRI Classifier — Build Plan

Goal: rebuild the tumor classification project as a standalone piece that
differentiates on rigor, not just architecture, comparison against a
baseline, evaluation beyond accuracy, explainability, and honest framing
of clinical risk.

## Week 1: Data, baseline, and architecture comparison

- Day 1-2: Kaggle Brain Tumor MRI Dataset (glioma, meningioma, pituitary,
  no_tumor). Proper train/val/test split. Run `src/data/dataset.py`'s
  `check_class_balance()` and document the result, don't assume balance.
- Day 3-4: Train the from-scratch CNN baseline (`src/models/baseline_cnn.py`).
  This is the number everything else is compared against.
- Day 5-7: Fine-tune ResNet50 and EfficientNetB0
  (`src/models/transfer_models.py`) on the same split. Apply augmentation
  (`src/data/augmentation.py`) targeted at whatever imbalance the Day 1-2
  check revealed.

Milestone: three trained models, one comparison table (accuracy, training
time, per-class F1).

## Week 2: Evaluation and explainability

- Day 8-9: k-fold cross-validation (`src/train/train.py`'s `run_kfold()`),
  report mean and variance per model, not a single split.
- Day 10-11: Full evaluation suite (`src/eval/metrics.py`): confusion
  matrix, per-class precision/recall/F1, ROC-AUC. Report false negative
  rate as the primary metric, a missed tumor is worse than a false alarm.
- Day 12-14: Grad-CAM (`src/eval/gradcam.py`) on correct and incorrect
  predictions. Look specifically for a case where the model got the right
  answer while attending to the wrong region, that's worth documenting.

Milestone: evaluation that reports the metric that actually matters
clinically, plus visual proof of what the model is attending to.

## Week 3: Deployment, model card, polish

- Day 15-17: FastAPI endpoint (`src/api/main.py`) returning prediction,
  confidence, and Grad-CAM overlay.
- Day 18-19: Streamlit frontend (`frontend/app.py`), upload an image, see
  prediction and heatmap side by side.
- Day 20: Model card (`docs/MODEL_CARD.md`), dataset, limitations,
  explicit statement this is not a diagnostic tool.
- Day 21: Deploy on AWS, README with the comparison table and Grad-CAM
  examples near the top, record a demo clip.

## Resume bullet template

"Built a brain tumor classification system comparing a CNN baseline
against fine-tuned ResNet50 and EfficientNetB0 on MRI scans (4 classes);
evaluated with k-fold cross-validation and Grad-CAM explainability,
achieving [X]% F1 with false-negative rate as the primary metric;
deployed via FastAPI on AWS."

## Interview talking points

- Why false negative rate matters more than accuracy here
- What Grad-CAM showed, including any "right for the wrong reason" case
- Why cross-validation instead of a single split, and what the variance
  across folds indicated
- What would be needed before this could be a real clinical tool, and
  why it explicitly isn't one
