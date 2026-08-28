# Brain Tumor MRI Classifier

Multi-class brain MRI classification system comparing a custom CNN baseline against fine-tuned transfer learning models (ResNet50, EfficientNetB0) across four classes: glioma, meningioma, pituitary, and normal brain scans.

Rather than reporting a single accuracy metric from an arbitrary train/test split, this repository uses stratified 5-fold cross-validation, evaluates model explainability via Grad-CAM, estimates prediction confidence using Monte Carlo Dropout, and prioritizes the False Negative Rate (FNR) to reflect clinical triage priorities.

**Disclaimer:** Research and educational code only. Not approved for diagnostic or clinical use. See `docs/MODEL_CARD.md`.

---

## Benchmark Results

Evaluated across 7,200 clinical MRI scans (1,800 per class):

| Model Architecture | Test Accuracy | Macro F1-Score | False Negative Rate (FNR) | Mean ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Baseline CNN (from scratch) | 88.40% | 87.90% | 3.80% | 0.954 |
| **ResNet50 (Fine-Tuned Champion)** | **96.19%** | **96.18%** | **0.44%** | **0.998** |
| EfficientNetB0 (Fine-Tuned) | 91.75% | 91.50% | 1.85% | 0.976 |

### Per-Class Performance (Champion ResNet50)
- **Glioma**: 97.4% F1-Score | 97.9% Precision | 96.9% Recall (ROC-AUC: 0.9987)
- **Meningioma**: 93.5% F1-Score | 98.2% Precision | 89.2% Recall (ROC-AUC: 0.9955)
- **Pituitary**: 94.8% F1-Score | 90.7% Precision | 99.2% Recall (ROC-AUC: 0.9992)
- **No Tumor (Healthy Control)**: 99.0% F1-Score | 98.7% Precision | 99.4% Recall (ROC-AUC: 0.9999)
- **Primary Clinical Metric (False Negative Rate)**: **0.44%** (99.56% sensitivity on tumor identification)

*Detailed metrics and fold logs are stored in `docs/eval_results.json`.*


---

## Core Technical Decisions

1. **False Negative Rate as Primary Metric**: In clinical imaging, failing to identify an existing tumor (predicting `no_tumor` when pathology exists) is significantly more detrimental than a false positive. Training handles class imbalance via inverse frequency weighting.
2. **Explainability via Grad-CAM**: Model decisions are inspected using Grad-CAM attention heatmaps extracted from the final convolutional stage (`conv5_block3_out` for ResNet50). Generated visual examples are saved in `docs/gradcam_examples/`.
3. **Bayesian Uncertainty (Monte Carlo Dropout)**: During inference, $N=20$ stochastic forward passes with dropout active calculate epistemic standard deviation ($\sigma$) and normalized Shannon entropy ($H$) to identify low-confidence scans for human review.
4. **Clinical PDF Reporting**: An integrated ReportLab engine generates structured case summaries with embedded heatmaps and probability distributions (`docs/reports/sample_clinical_report.pdf`).

---

## Repository Layout

```text
├── src/
│   ├── data/            # Ingestion, augmentation, and array loaders
│   ├── models/          # Baseline CNN, ResNet50, and EfficientNetB0 definitions
│   ├── train/           # K-fold cross-validation engine
│   ├── eval/            # Metrics, Grad-CAM resolver, MC Dropout, and PDF generator
│   └── api/             # FastAPI service (/health, /predict, /report)
├── scripts/
│   ├── download_dataset.py     # Kaggle API and archive extractor
│   ├── generate_sample_data.py # Synthetic MRI generator for local testing
│   ├── train_all.py            # 5-fold cross-validation benchmarking runner
│   ├── train_final.py          # Production checkpoint trainer
│   └── evaluate_final.py       # Full evaluation suite and heatmap generator
├── frontend/
│   └── app.py                  # Streamlit diagnostic interface
├── tests/                      # Unit test suite (12 tests)
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

### 3. Run Cross-Validation Benchmark

```bash
python scripts/train_all.py --k-folds 5 --epochs 15
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

## Local Serving

Start the FastAPI inference backend:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

In a separate terminal, launch the Streamlit frontend:
```bash
streamlit run frontend/app.py
```

### Endpoints
- `GET /health`: Service health, model status, and enabled features.
- `GET /classes`: Target class mapping.
- `POST /predict`: Upload scan image; returns classification, probability breakdown, epistemic uncertainty, and base64 Grad-CAM overlay.
- `POST /report`: Upload scan image; returns binary downloadable clinical PDF summary.

---

## Docker Deployment

Build and start the containerized service:
```bash
docker compose up -d --build
```
The API is available at `http://localhost:8000`.


