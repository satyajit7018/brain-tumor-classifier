# Model Card — Brain Tumor MRI Classifier

## Intended use

This is a research and educational project demonstrating a multi-class
image classification pipeline with explainability. **It is not a
diagnostic tool and must not be used to make or support real clinical
decisions.** Any real medical use would require regulatory clearance,
much larger and more diverse validation data, and clinical oversight
that this project does not have.

## Dataset

- Source: Kaggle Brain Tumor MRI Dataset
- Classes: glioma, meningioma, pituitary, no_tumor
- Size: 80 test samples across 4 classes (~7,023 full dataset)
- Class balance: Balanced (20 samples per class)
- Train/val split: 80% Training (64 samples), 20% Validation (16 samples)

## Models compared

| Model | Mean CV accuracy | Std | False negative rate | Notes |
|---|---|---|---|---|
| Baseline CNN (from scratch) | 25.02% | ±2.02% | 0.00% | 4-layer conv baseline |
| **ResNet50 (fine-tuned)** | **59.83%** | **±10.13%** | **0.00%** | **Top performing architecture** |
| EfficientNetB0 (fine-tuned) | 25.02% | ±2.02% | 0.00% | Compound scaling baseline |

## Primary metric

False negative rate (tumor cases predicted as no_tumor) is reported as
the primary metric, not accuracy. In this context a missed tumor is a
categorically worse error than a false alarm, so the model that
minimizes false negatives, even at some accuracy cost, is the one worth
preferring.

## Explainability

Grad-CAM overlays are generated for both correct and incorrect
predictions. See `docs/gradcam_examples/` for sample outputs, including
analysis of anatomical attention boundaries around hyperintense lesions
and dural-based focal mass regions.

## Known limitations

- Single public dataset, not validated against scans from different
  scanners, institutions, or populations
- No radiologist review of predictions
- Class definitions and labels inherited as-is from the source dataset
- Not evaluated for fairness across demographic subgroups (data not available)

