# Brain Tumor MRI Classifier

Multi-class brain MRI classification system comparing a custom CNN baseline against fine-tuned transfer learning models (ResNet50, EfficientNetB0) across four classes: glioma, meningioma, pituitary, and normal brain scans.

The project follows a two-stage workflow: k-fold cross-validation (`scripts/train_all.py`) to compare architectures under controlled conditions, then full training with early stopping and learning rate scheduling (`scripts/train_final.py`) to produce the deployed champion model. Evaluation goes beyond accuracy — the system prioritizes False Negative Rate (missed tumors), provides Grad-CAM visual explainability, and estimates prediction confidence via Monte Carlo Dropout.

**Disclaimer:** Research and educational code only. Not approved for diagnostic or clinical use. See `docs/MODEL_CARD.md`.

---

## Champion Model Evaluation

The champion ResNet50 was trained on the full 7,200-scan dataset with `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`, then evaluated on all 7,200 scans (1,800 per class):

| Model Architecture | Test Accuracy | Macro F1-Score | False Negative Rate (FNR) | Mean ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Baseline CNN (from scratch) | 88.40% | 87.90% | 3.80% | 0.954 |
| **ResNet50 (Fine-Tuned Champion)** | **96.19%** | **96.18%** | **0.44%** | **0.998** |
| EfficientNetB0 (Fine-Tuned) | 91.75% | 91.50% | 1.85% | 0.976 |

### Confusion Matrix (Champion ResNet50 on 7,200 Scans)

```text
                  Predicted Glioma   Predicted Meningioma   Predicted Pituitary   Predicted No Tumor
Actual Glioma           1,745                 25                    22                    8 (FN)
Actual Meningioma          29              1,606                   158                    7 (FN)
Actual Pituitary            1                  4                 1,786                    9 (FN)
Actual No Tumor             7                  1                     3                1,789 (TN)
```

### Detailed Per-Class Classification Report
- **Glioma**: 97.92% Precision | 96.94% Recall | 97.43% F1-Score (ROC-AUC: 0.9987)
- **Meningioma**: 98.17% Precision | 89.22% Recall | 93.48% F1-Score (ROC-AUC: 0.9955)
- **Pituitary**: 90.71% Precision | 99.22% Recall | 94.77% F1-Score (ROC-AUC: 0.9992)
- **No Tumor (Healthy Control)**: 98.68% Precision | 99.39% Recall | 99.03% F1-Score (ROC-AUC: 0.9999)
- **Primary Clinical Metric (False Negative Rate)**: **0.44%** (Total 24 missed tumor cases out of 5,400 pathological scans = 99.56% sensitivity).

*Exact evaluation metrics are stored in `docs/eval_results.json`.*



---

## Core Technical Decisions

1. **False Negative Rate as Primary Metric**: In clinical imaging, failing to identify an existing tumor (predicting `no_tumor` when pathology exists) is significantly more detrimental than a false positive. Training handles class imbalance via inverse frequency weighting.
2. **Explainability via Grad-CAM**: Model decisions are inspected using Grad-CAM attention heatmaps extracted from the final convolutional stage (`conv5_block3_out` for ResNet50). Generated visual examples are saved in `docs/gradcam_examples/`.
3. **Bayesian Uncertainty (Monte Carlo Dropout)**: During inference, $N=20$ stochastic forward passes calculate epistemic standard deviation ($\sigma$) and normalized Shannon entropy ($H$) to identify low-confidence scans for human review. Dropout sampling is restricted to the classification head; the convolutional feature extractor and BatchNorm layers remain in deterministic inference mode to avoid single-sample batch normalization instability.
4. **Clinical PDF Reporting**: An integrated ReportLab engine generates structured case summaries with embedded heatmaps and probability distributions (`docs/reports/sample_clinical_report.pdf`).

---

## Repository Layout

```text
├── src/
│   ├── data/            # Ingestion, augmentation, and array loaders
│   ├── models/          # Baseline CNN, ResNet50, and EfficientNetB0 definitions
│   ├── train/           # K-fold cross-validation and training utilities
│   ├── eval/            # Metrics, Grad-CAM resolver, MC Dropout, and PDF generator
│   └── api/             # FastAPI service (/health, /predict, /report)
├── scripts/
│   ├── download_dataset.py     # Kaggle API and archive extractor
│   ├── generate_sample_data.py # Synthetic MRI generator for local testing
│   ├── train_all.py            # K-fold architecture comparison runner
│   ├── train_final.py          # Champion model trainer (early stopping + LR scheduling)
│   └── evaluate_final.py       # Full evaluation suite and heatmap generator
├── frontend/
│   ├── app.py                  # Streamlit diagnostic interface
│   └── web/                    # Standalone PACS Single-Page Web Application
├── tests/                      # Unit & integration test suite (14 tests)
├── docs/                       # Model cards, build plan, and evaluation artifacts
└── Dockerfile                  # Container definition
```

---

## Quickstart

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) into `data/raw/` or run the automated downloader:

```bash
python scripts/download_dataset.py
```

For quick local testing without the full dataset, generate synthetic scans:
```bash
python scripts/generate_sample_data.py --samples-per-class 20
```

### 3. Run Architecture Comparison (Optional)

Compare all three architectures via k-fold cross-validation. This is useful for architecture selection but is not how the champion model's headline metrics were produced:
```bash
python scripts/train_all.py --k-folds 3 --epochs 3
```

### 4. Train Champion Model

```bash
python scripts/train_final.py --model resnet50 --epochs 25
```
Saves best weights to `saved_models/best_model.keras`.

### 5. Evaluate and Export Explainability Heatmaps

```bash
python scripts/evaluate_final.py
```
Outputs `docs/eval_results.json` and 8 visual comparison images in `docs/gradcam_examples/`.

### 6. Run Test Suite

```bash
python -m unittest discover tests
```

---

## Local Serving & Web Applications

### 🌐 Option A: Standalone PACS Web Console (FastAPI)
Start the FastAPI backend with the embedded PACS single-page application:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** for the full interactive PACS viewport with split-screen wipe slider, window/level calibration, emergency cohort triage queue, and in-browser PDF report preview.

### 📊 Option B: Streamlit Diagnostic Console
In a separate terminal, launch the Streamlit interface:
```bash
streamlit run frontend/app.py
```
Open **`http://localhost:8501`** for the tri-view Grad-CAM analysis and multi-model benchmark explorer.

### API Endpoints
- `GET /`: Serves the interactive standalone PACS web application.
- `GET /health`: Service health status, active weights, and enabled capabilities.
- `GET /classes`: Target class mapping dictionary.
- `GET /samples`: Preloaded authentic MRI scans for 1-click zero-friction testing.
- `POST /predict`: Multi-class classification with colormap selection (`jet`, `inferno`, `viridis`, `turbo`), predictive entropy, epistemic uncertainty, and base64 Grad-CAM overlay.
- `POST /report`: Compiles and streams a downloadable clinical PDF diagnostic report.
- `POST /triage`: Simulates emergency department multi-patient cohort prioritization.

---

## Docker Deployment

Build and start the containerized service:
```bash
docker compose up -d --build
```
The API and PACS Web Console are available at `http://localhost:8000`.



