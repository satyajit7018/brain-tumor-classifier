# Brain Tumor MRI Classifier

Multi-class brain MRI classification system comparing a custom CNN baseline against fine-tuned transfer learning models (ResNet50, EfficientNetB0) across four classes: glioma, meningioma, pituitary, and normal brain scans.

Each model is trained with early stopping and learning-rate scheduling on the training split (`scripts/train_final.py`) and scored once on a held-out test split (`scripts/evaluate_final.py`). Optional k-fold cross-validation on the training split is available in `scripts/train_all.py`. Evaluation goes beyond accuracy — the system prioritizes False Negative Rate (missed tumors), provides Grad-CAM visual explainability, and estimates prediction confidence via Monte Carlo Dropout.

**Disclaimer:** Research and educational code only. Not approved for diagnostic or clinical use. See `docs/MODEL_CARD.md`.

---

## Results on a held-out test set

All three models were trained on the dataset's official **Training** split (5,600 scans, 15% of
it held back for validation and early stopping) and evaluated once on the official **Testing**
split: 1,600 scans, 400 per class, never used for training, validation or model selection. Test
images were checked against the training set by file hash: 0 exact duplicates.

| Model | Test accuracy | Macro F1 | False negative rate | Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Baseline CNN (from scratch) | 93.25% | 93.07% | 3.33% | 0.976 |
| EfficientNetB0 (fine-tuned) | 93.75% | 93.63% | 2.58% | 0.989 |
| **ResNet50 (fine-tuned)** | **94.81%** | **94.69%** | **2.50%** | **0.986** |

False negative rate = share of tumor scans predicted as `no_tumor` (missed tumors). Full metrics:
[`docs/eval_results_resnet50.json`](docs/eval_results_resnet50.json),
[`efficientnet_b0`](docs/eval_results_efficientnet_b0.json),
[`baseline_cnn`](docs/eval_results_baseline_cnn.json). Trained on a Kaggle T4 GPU with
[`brain_tumor_honest_eval.ipynb`](notebooks/brain_tumor_honest_eval.ipynb).

### ResNet50 confusion matrix (1,600 test scans)

```text
                    Pred glioma   Pred meningioma   Pred pituitary   Pred no_tumor
Actual glioma           328             41                 3              28  ← missed
Actual meningioma         1            389                 8               2  ← missed
Actual pituitary          0              0               400               0
Actual no_tumor           0              0                 0             400
```

| Class | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Glioma | 99.7% | **82.0%** | 90.0% | 0.958 |
| Meningioma | 90.5% | 97.2% | 93.7% | 0.986 |
| Pituitary | 97.3% | 100.0% | 98.6% | 1.000 |
| No tumor | 93.0% | 100.0% | 96.4% | 0.999 |

### What these numbers do and don't show

- **Glioma is the weak spot.** 28 of 400 gliomas (7%) were called `no_tumor`, and those account
  for 28 of the 30 missed tumors. The overall 2.5% false negative rate hides this, so glioma
  recall is the number to improve next.
- **The models are close.** With 1,600 test scans, the 95% confidence interval on accuracy is
  about ±1.1 points, so ResNet50's lead over EfficientNetB0 (+1.1 pts) is not clearly
  significant. Each number comes from a single training run.
- **Validation was optimistic.** ResNet50 reached 98.7% validation accuracy but 94.8% on test.
  The validation images come from the same pool as the training images, so the official test
  split is the more honest estimate.
- **The CNN and EfficientNetB0 hit the 25-epoch cap** without early stopping, so they may improve
  with longer training.

> An earlier version of this README reported 96.2% accuracy and a 0.44% false negative rate.
> Those numbers came from evaluating on images the model had been trained on, and have been
> replaced by the held-out results above.

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
├── tests/                      # Unit & integration test suite
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

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and organize it into `data/raw/` with the downloader. It prefixes every file with its original split (`Training_` / `Testing_`), which is how training and evaluation keep the test images separate:

```bash
python scripts/download_dataset.py
```

For quick local testing without the full dataset, generate synthetic scans:
```bash
python scripts/generate_sample_data.py --samples-per-class 20
```

### 3. Run Architecture Comparison (Optional)

Compare architectures with k-fold cross-validation on the training split (the test split is never used). The headline results above come from steps 4–5, not from this:
```bash
python scripts/train_all.py --k-folds 3 --epochs 3
```

### 4. Train Champion Model

```bash
python scripts/train_final.py --model resnet50 --epochs 25
```
Trains on the `Training_*` images only and saves the best weights to
`saved_models/best_model.keras`. No local GPU? Run
[`notebooks/brain_tumor_honest_eval.ipynb`](notebooks/brain_tumor_honest_eval.ipynb) on Kaggle
to train and evaluate all three models in about 30 minutes.

### 5. Evaluate and Export Explainability Heatmaps

```bash
python scripts/evaluate_final.py
```
Evaluates on the held-out `Testing_*` images and writes `docs/eval_results.json` plus 8
Grad-CAM comparison images in `docs/gradcam_examples/`.

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
- `GET /metrics`: Prometheus-compatible real-time performance telemetry.
- `GET /classes`: Target class mapping dictionary.
- `GET /samples`: Preloaded authentic MRI scans for 1-click zero-friction testing.
- `POST /predict`: Multi-class classification with in-memory caching, 5-fold TTA option (`use_tta=true`), colormap selection (`jet`, `inferno`, `viridis`, `turbo`), predictive entropy, epistemic uncertainty, and base64 Grad-CAM overlay.
- `POST /predict/batch`: Parallel batch inference endpoint for multi-scan processing.
- `POST /report`: Compiles and streams a downloadable clinical PDF diagnostic report.
- `POST /triage`: Simulates emergency department multi-patient cohort prioritization.

---

## Docker Deployment

Build and start the containerized service:
```bash
docker compose up -d --build
```
The API and PACS Web Console are available at `http://localhost:8000`.
