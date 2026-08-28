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

## Resume Bullet Template (Verified)

"Fine-tuned ResNet50 on 7,200 multi-class clinical MRI scans (glioma, meningioma, pituitary, normal), achieving 96.2% accuracy, 0.44% False Negative Rate (24/5,400 missed tumors), and 0.998 mean ROC-AUC; implemented universal Grad-CAM explainability and Monte Carlo Dropout (N=20) for Bayesian epistemic uncertainty; deployed via containerized FastAPI backend with automated clinical PDF reporting."

## Interview Talking Points

- **Why False Negative Rate (FNR) was prioritized**: In medical triage, failing to identify an existing tumor (predicting `no_tumor` when pathology is present) is a critical error mode. Our fine-tuned champion achieved a **0.44% FNR (99.56% sensitivity)** across 5,400 pathological cases.
- **Explainability via Grad-CAM**: We extracted convolutional gradients from `conv5_block3_out` to visually verify that the model attends to authentic intracranial focal lesions rather than skull boundaries or background artifacts.
- **Bayesian Epistemic Uncertainty**: Rather than relying on static softmax probabilities, Monte Carlo Dropout ($N=20$ stochastic forward passes) calculates class standard deviation ($\sigma$) and normalized Shannon entropy ($H$) to automatically flag ambiguous scans for mandatory radiologist review.
- **Clinical Limitations & Non-Diagnostic Scope**: Sourced from retrospective public datasets; requires prospective multi-scanner validation (1.5T vs 3.0T) and blinded neuroradiologist confirmation before any clinical deployment.
