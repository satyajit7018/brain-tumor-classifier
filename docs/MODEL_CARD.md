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

The champion ResNet50 was trained on the full 7,200-scan dataset with `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`, then evaluated on all scans:

| Model Architecture | Accuracy | Macro F1-Score | False Negative Rate (FNR) | Mean ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Baseline CNN | 88.40% | 87.90% | 3.80% | 0.954 |
| **ResNet50 (Fine-Tuned)** | **96.19%** | **96.18%** | **0.44%** | **0.998** |
| EfficientNetB0 (Fine-Tuned) | 91.75% | 91.50% | 1.85% | 0.976 |

### Per-Class Detailed Breakdown (Champion ResNet50)
- **Glioma**: Precision 97.9%, Recall 96.9%, F1 97.4% (ROC-AUC: 0.9987)
- **Meningioma**: Precision 98.2%, Recall 89.2%, F1 93.5% (ROC-AUC: 0.9955)
- **Pituitary**: Precision 90.7%, Recall 99.2%, F1 94.8% (ROC-AUC: 0.9992)
- **No Tumor (Healthy Control)**: Precision 98.7%, Recall 99.4%, F1 99.0% (ROC-AUC: 0.9999)
- **Primary Clinical Metric (False Negative Rate)**: **0.44%** (Only 24 missed tumor cases out of 5,400 pathological scans).


### Decision Metric Hierarchy
In clinical triage, **False Negative Rate (FNR)** is prioritized over raw accuracy:
$$\text{FNR} = \frac{\text{Actual Tumor cases predicted as No Tumor}}{\text{Total Actual Tumor cases}}$$
A missed tumor represents a critical failure mode; the system enforces inverse-frequency loss weighting to prevent healthy-class overprediction.

---

## 5. Explainability & Uncertainty Analysis

- **Grad-CAM Analysis:** Gradient-weighted class activation mapping visualizes convolutional feature activations directly before global pooling. Visual audits verify that predictions activate on intracranial lesions rather than skull boundaries or background artifacts (`docs/gradcam_examples/`).
- **Bayesian Epistemic Uncertainty:** $N=20$ stochastic forward passes compute class variance ($\sigma$) and normalized Shannon entropy ($H$). Dropout sampling is restricted to the classification head; the convolutional feature extractor and BatchNorm layers remain in deterministic inference mode to avoid single-sample batch normalization instability. Scans exhibiting high entropy ($H \ge 0.50$), low confidence ($< 65\%$), or significant variance ($\sigma_{\max} \ge 0.35$) trigger a `HIGH_RISK_RADIOLOGIST_REVIEW` warning.

---

## 6. Known Limitations

- **Dataset Diversity:** Sourced from publicly available retrospective cohorts; lacks prospective validation across disparate scanner field strengths (1.5T vs 3.0T) or non-standard MRI sequences (FLAIR, T1-contrast, T2-weighted).
- **Demographic Disclosures:** Demographic variables (age, sex, ethnicity) are unavailable in the source data, precluding subgroup bias auditing.
- **Pathological Confirmation:** Labels are inherited directly from source datasets without secondary blinded neuroradiologist review.


