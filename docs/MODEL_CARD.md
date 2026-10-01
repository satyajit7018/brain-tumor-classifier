# Model Card: Brain Tumor MRI Classifier

## 1. Model Details

- **Developer:** Satyajit Nayak
- **Model Date:** October 2026
- **Model Version:** 1.2.0 (held-out re-evaluation)
- **Architectures Evaluated:**
  - Custom 4-block CNN Baseline (from scratch)
  - Fine-Tuned ResNet50 (first 140 layers frozen, `fine_tune_at=140`)
  - Fine-Tuned EfficientNetB0 (first 200 layers frozen, `fine_tune_at=200`)
- **Champion Model:** ResNet50 (Fine-Tuned)
- **Framework:** TensorFlow 2.x / Keras 3 (Adam, initial lr=$3 \times 10^{-4}$, halved on plateau)
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
- **Normalization:** Rescaling to $[0.0, 1.0]$, then each backbone's own ImageNet preprocessing.
- **Augmentation Pipeline:** Horizontal random flips, random rotation (±5% of a full turn, about ±18°), random zoom (±5%), brightness (±10%) and contrast (±10%).

---

## 4. Evaluation & Performance

**Protocol:** trained on the official Training split (5,600 scans; 15% held back for validation
and early stopping), evaluated once on the official Testing split (1,600 scans, 400 per class).
Test images never appear in training or validation; a file-hash check found 0 duplicates across
the two splits. Single training run per model, Kaggle T4 GPU, October 2026.

| Model | Accuracy | Macro F1 | False Negative Rate | Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Baseline CNN | 93.25% | 93.07% | 3.33% | 0.976 |
| EfficientNetB0 (fine-tuned) | 93.75% | 93.63% | 2.58% | 0.989 |
| **ResNet50 (fine-tuned)** | **94.81%** | **94.69%** | **2.50%** | **0.986** |

95% confidence interval on accuracy at n = 1,600 is roughly ±1.1 points, so the gap between
ResNet50 and EfficientNetB0 is within noise.

### Per-class breakdown (ResNet50)
- **Glioma**: precision 99.7%, recall **82.0%**, F1 90.0% (ROC-AUC 0.958)
- **Meningioma**: precision 90.5%, recall 97.2%, F1 93.7% (ROC-AUC 0.986)
- **Pituitary**: precision 97.3%, recall 100.0%, F1 98.6% (ROC-AUC 1.000)
- **No tumor**: precision 93.0%, recall 100.0%, F1 96.4% (ROC-AUC 0.999)
- **False negative rate**: 2.50% (30 of 1,200 tumor scans predicted `no_tumor`; 28 of them are
  gliomas, a 7% miss rate for that class).

Validation accuracy (98.7%) was well above test accuracy (94.8%) because validation images come
from the training pool; report the test numbers.

Earlier versions of this card (v1.1.0) reported 96.19% accuracy and 0.44% FNR measured on
training images. Those figures are withdrawn.

### Decision Metric Hierarchy
In clinical triage, **False Negative Rate (FNR)** is prioritized over raw accuracy:
$$\text{FNR} = \frac{\text{Actual Tumor cases predicted as No Tumor}}{\text{Total Actual Tumor cases}}$$
A missed tumor is the critical failure mode. Training applies inverse-frequency class weights (the splits used here are balanced, so they are all 1.0). Glioma recall is the current weakness to target.

---

## 5. Explainability & Uncertainty Analysis

- **Grad-CAM Analysis:** Gradient-weighted class activation mapping visualizes convolutional feature activations directly before global pooling. Visual audits verify that predictions activate on intracranial lesions rather than skull boundaries or background artifacts (`docs/gradcam_examples/`).
- **Bayesian Epistemic Uncertainty:** $N=20$ stochastic forward passes compute class variance ($\sigma$) and normalized Shannon entropy ($H$). Dropout sampling is restricted to the classification head; the convolutional feature extractor and BatchNorm layers remain in deterministic inference mode to avoid single-sample batch normalization instability. Scans exhibiting high entropy ($H \ge 0.50$), low confidence ($< 65\%$), or significant variance ($\sigma_{\max} \ge 0.35$) trigger a `HIGH_RISK_RADIOLOGIST_REVIEW` warning.

---

## 6. Known Limitations

- **Dataset Diversity:** Sourced from publicly available retrospective cohorts; lacks prospective validation across disparate scanner field strengths (1.5T vs 3.0T) or non-standard MRI sequences (FLAIR, T1-contrast, T2-weighted).
- **Demographic Disclosures:** Demographic variables (age, sex, ethnicity) are unavailable in the source data, precluding subgroup bias auditing.
- **Glioma sensitivity:** 7% of test gliomas are classified as `no_tumor`; any downstream use would need a higher-sensitivity operating point or a second reader.
- **Pathological Confirmation:** Labels are inherited directly from source datasets without secondary blinded neuroradiologist review.


