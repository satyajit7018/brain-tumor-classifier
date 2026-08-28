# Brain Tumor Classifier — Next Steps

Everything from here to a deployed, documented project. Run these in
order from inside the `brain-tumor-classifier` folder. Each phase has a
checkpoint, don't move to the next phase until the checkpoint looks right.

---

## Phase 1: Dataset setup

The dataset is Masoud Nickparvar's "Brain Tumor MRI Dataset" on Kaggle,
~7,023 images across glioma, meningioma, pituitary, and notumor, combined
from the Figshare, SARTAJ, and Br35H datasets. Two ways to get it, pick
whichever is easier for you.

**Option A: Browser download (no setup needed)**

1. Go to `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`
2. Sign in (or create a free account, top right).
3. Click the **Download** button near the top of the page (it downloads
   the whole dataset as a single zip, roughly 150MB).
4. Unzip it. You'll get `Training/` and `Testing/` folders, each with one
   subfolder per class (`glioma`, `meningioma`, `pituitary`, `notumor`).

**Option B: Kaggle CLI (faster if you're comfortable with API tokens)**

1. Install the CLI: `pip install kaggle`
2. Get an API token: go to `https://www.kaggle.com/settings`, scroll to
   the API section, click **Create New Token**. This downloads a
   `kaggle.json` file.
3. Move it where the CLI expects it:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Download and unzip directly:
   ```bash
   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p ~/Downloads/brain-mri --unzip
   ```
   This puts `Training/` and `Testing/` folders straight into
   `~/Downloads/brain-mri/`, no manual unzip step.

**Then, either way, merge Training and Testing into `data/raw/`:**

Note the dataset's folder is named `notumor` (no underscore, no space),
this project's code expects `no_tumor`, that's why the copy below renames
it on the way in.

```bash
mkdir -p data/raw/glioma data/raw/meningioma data/raw/pituitary data/raw/no_tumor

cp Training/glioma/* Testing/glioma/* data/raw/glioma/
cp Training/meningioma/* Testing/meningioma/* data/raw/meningioma/
cp Training/pituitary/* Testing/pituitary/* data/raw/pituitary/
cp Training/notumor/* Testing/notumor/* data/raw/no_tumor/
```

(Adjust the source paths above to wherever you unzipped or downloaded to,
e.g. `~/Downloads/brain-mri/Training/glioma/*` if you used Option B.)

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the class balance check:

```bash
python src/data/dataset.py
```

**Checkpoint:** you should see a per-class image count and a total around
7,000. If one class is drastically smaller than the others, note it, it
determines whether `compute_class_weights()` gets used in Phase 3.

---

## Phase 2: Train all three models with k-fold CV

This is the comparison that becomes your headline result. It loads every
image into memory, so it's slow (expect it to take a while, longer without
a GPU) and only needs to run once.

```bash
python scripts/train_all.py
```

**Checkpoint:** `docs/kfold_results.json` gets created with mean/std
accuracy for `baseline_cnn`, `resnet50`, and `efficientnet_b0`. Copy those
numbers into the Results table in `README.md` and the model comparison
table in `docs/MODEL_CARD.md`.

---

## Phase 3: Train the final deployable model

Pick whichever architecture won Phase 2 (ResNet50 and EfficientNetB0 will
likely both beat the baseline, pick the one with the better mean accuracy
and lower variance).

```bash
python scripts/train_final.py --model resnet50
```

(swap `resnet50` for `efficientnet_b0` or `baseline_cnn` if a different
one won)

**Checkpoint:** `saved_models/best_model.keras` exists. If you picked
`efficientnet_b0`, update `LAST_CONV_LAYER` in `src/api/main.py` and
`scripts/evaluate_final.py` from `"resnet50"` to `"top_conv"` (EfficientNet's
final conv layer name).

---

## Phase 4: Full evaluation and Grad-CAM

```bash
python scripts/evaluate_final.py
```

**Checkpoint:** `docs/eval_results.json` has the confusion matrix,
per-class precision/recall/F1, ROC-AUC, and false negative rate.
`docs/gradcam_examples/` has 8 images, correct and incorrect predictions
with their attention heatmaps. Open a few of the "incorrect" ones, look
specifically for a case where the model attended to the wrong region,
that's worth a sentence in the model card.

Copy the false negative rate into the Results table in `README.md`
(that's the number that matters most, lead with it over raw accuracy).

---

## Phase 5: Test the API and frontend locally

```bash
uvicorn src.api.main:app --reload
```

In another terminal:

```bash
curl -X POST http://localhost:8000/predict -F "file=@data/raw/glioma/<pick-any-filename>.jpg"
```

**Checkpoint:** you get back JSON with `predicted_class`, `confidence`,
and a base64 `gradcam_overlay`. Then test the actual UI:

```bash
streamlit run frontend/app.py
```

Upload an image, confirm the prediction and heatmap render side by side.

---

## Phase 6: Fill in the documentation

- `README.md`: paste the Phase 2 comparison table and Phase 4 false
  negative rate into the Results table.
- `docs/MODEL_CARD.md`: fill in dataset size, class balance, and the
  model comparison table. Write one or two sentences about what the
  Grad-CAM examples showed.
- Record a short screen capture (QuickTime screen recording works fine)
  showing an upload, a prediction, and the heatmap. Save it as
  `docs/demo.gif` or link a short video in the README.

---

## Phase 7: Deploy

1. Spin up an AWS EC2 free-tier instance (or Lightsail if you want
   simpler pricing).
2. Install Docker on it, `git clone` your repo.
3. Copy `saved_models/best_model.keras` to the instance (it's gitignored,
   so it won't come from `git clone`, use `scp`).
4. `docker compose up -d --build`
5. Confirm `http://<instance-ip>:8000/health` returns `{"status": "ok"}`.

**Checkpoint:** the API is reachable from outside your machine. Add the
URL to `README.md` and to the repo's GitHub description field.

---

## Phase 8: Commit and push everything

```bash
git add -A
git commit -m "Add trained models, evaluation results, and Grad-CAM examples"
git push
```

Note: `saved_models/` and `data/raw/` are gitignored on purpose (model
weights and the dataset are too large for a normal git repo), so only the
results JSON, Grad-CAM example images, and updated docs get pushed. That's
intentional and fine, a resume reviewer needs the evidence and the code,
not the raw weights.

---

## Resume bullet, once Phase 4 numbers are in

"Built a brain tumor classification system comparing a CNN baseline
against fine-tuned ResNet50 and EfficientNetB0 on MRI scans (4 classes);
evaluated with k-fold cross-validation and Grad-CAM explainability,
achieving [X]% F1 with a [Y]% false-negative rate; deployed via FastAPI
on AWS."
