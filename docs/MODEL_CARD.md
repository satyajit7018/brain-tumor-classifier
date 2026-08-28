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
- Size: TBD (fill in after running `src/data/dataset.py`'s class balance check)
- Class balance: TBD
- Train/val/test split: TBD

## Models compared

| Model | Mean CV accuracy | Std | False negative rate | Notes |
|---|---|---|---|---|
| Baseline CNN (from scratch) | TBD | TBD | TBD | |
| ResNet50 (fine-tuned) | TBD | TBD | TBD | |
| EfficientNetB0 (fine-tuned) | TBD | TBD | TBD | |

## Primary metric

False negative rate (tumor cases predicted as no_tumor) is reported as
the primary metric, not accuracy. In this context a missed tumor is a
categorically worse error than a false alarm, so the model that
minimizes false negatives, even at some accuracy cost, is the one worth
preferring.

## Explainability

Grad-CAM overlays are generated for both correct and incorrect
predictions. See `docs/gradcam_examples/` for sample outputs, including
any case where the model reached the right answer while attending to
the wrong region of the scan.

## Known limitations

- Single public dataset, not validated against scans from different
  scanners, institutions, or populations
- No radiologist review of predictions
- Class definitions and labels inherited as-is from the source dataset
- Not evaluated for fairness across demographic subgroups (data not available)
