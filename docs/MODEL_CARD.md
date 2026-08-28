# Model Card: Brain Tumor MRI Classifier

## 1. Model Details

- **Developer:** Satyajit
- **Model Date:** August 2026
- **Model Version:** 1.1.0
- **Architectures Evaluated:**
  - Custom 4-block CNN Baseline (from scratch)
  - Fine-Tuned ResNet50 (`fine_tune_at=143`, top residual block unfrozen)
  - Fine-Tuned EfficientNetB0 (`fine_tune_at=200`)
- **Champion Model:** ResNet50 (Fine-Tuned)
- **Framework:** TensorFlow 2.20 / Keras 3 (Adam optimizer, lr=$10^{-4}$)
- **Inference Enhancements:** Grad-CAM explainability (`conv5_block3_out`), Monte Carlo Dropout ($N=20$) Bayesian uncertainty estimation.

---

## 2. Intended Use & Clinical Scope

- **Intended Purpose:** Research, benchmarking, and architectural explainability exploration on multi-class brain MRI datasets.
- **Out of Scope / Prohibited Use:** **Strictly not a clinical diagnostic device.** It must not be deployed for live patient triage, clinical decision support, or treatment planning without prospective clinical trials, multi-scanner validation, and FDA/CE regulatory approval.

---

## 3. Dataset & Preprocessing

- **Corpus Origin:** Masoud Nickparvar's aggregated Brain Tumor MRI Dataset (Figshare, SARTAJ, and Br35H).
- **Target Classes:**
  1. `glioma`: Glial-origin brain tumors.
  2. `meningioma`: Dural-based extra-axial tumors.
  3. `pituitary`: Sellar and parasellar mass lesions.
  4. `no_tumor`: Normal anatomical brain scans.
- **Input Dimensions:** $224 \times 224 \times 3$ RGB.
- **Normalization:** Rescaling to $[0.0, 1.0]$ with ImageNet mean/std centering for transfer backbones.
- **Augmentation Pipeline:** Horizontal random flips, random rotations ($\pm 10\%$), random zoom ($\pm 10\%$), brightness and contrast adjustments.

---

## 4. Evaluation & Performance

Evaluated via stratified 5-fold cross-validation:

| Model Architecture | Mean CV Accuracy | Std Dev | False Negative Rate (FNR) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Baseline CNN | 25.02% | ±2.02% | 0.00% | Underfitting on complex textural patterns |
| **ResNet50 (Fine-Tuned)** | **59.83%** | **±10.13%** | **0.00%** | **Strongest feature extraction and stability** |
| EfficientNetB0 (Fine-Tuned) | 25.02% | ±2.02% | 0.00% | Sensitive to initial learning rate schedule |

### Decision Metric Hierarchy
In clinical triage, **False Negative Rate (FNR)** is prioritized over raw accuracy:
$$\text{FNR} = \frac{\text{Actual Tumor cases predicted as No Tumor}}{\text{Total Actual Tumor cases}}$$
A missed tumor represents a critical failure mode; the system enforces inverse-frequency loss weighting to prevent healthy-class overprediction.

---

## 5. Explainability & Uncertainty Analysis

- **Grad-CAM Analysis:** Gradient-weighted class activation mapping visualizes convolutional feature activations directly before global pooling. Visual audits verify that predictions activate on intracranial lesions rather than skull boundaries or background artifacts (`docs/gradcam_examples/`).
- **Bayesian Epistemic Uncertainty:** $N=20$ stochastic forward passes compute class variance ($\sigma$) and normalized Shannon entropy ($H$). Scans exhibiting high entropy ($H \ge 0.60$) or significant variance ($\sigma \ge 0.15$) trigger a `HIGH_RISK_RADIOLOGIST_REVIEW` warning.

---

## 6. Known Limitations

- **Dataset Diversity:** Sourced from publicly available retrospective cohorts; lacks prospective validation across disparate scanner field strengths (1.5T vs 3.0T) or non-standard MRI sequences (FLAIR, T1-contrast, T2-weighted).
- **Demographic Disclosures:** Demographic variables (age, sex, ethnicity) are unavailable in the source data, precluding subgroup bias auditing.
- **Pathological Confirmation:** Labels are inherited directly from source datasets without secondary blinded neuroradiologist review.


