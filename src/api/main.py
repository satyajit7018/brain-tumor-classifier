import base64
import io
import os
import glob
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.eval.gradcam import make_gradcam_heatmap, overlay_heatmap, COLORMAP_DICT
from src.eval.uncertainty import compute_mc_dropout_uncertainty
from src.eval.report_generator import generate_clinical_pdf_report

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/best_model.keras")

model: Optional[tf.keras.Model] = None


def load_classifier_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Could not load model from {MODEL_PATH}: {e}")
            model = None
    else:
        print(f"[WARNING] Model file {MODEL_PATH} not found. Running in uninitialized state.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_classifier_model()
    yield


app = FastAPI(
    title="NeuroScan AI — PACS Diagnostic API",
    description="Multi-class brain tumor classification with Grad-CAM explainability, Bayesian uncertainty, and automated PDF reporting.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    std_probabilities: Dict[str, float]
    epistemic_uncertainty: float
    predictive_entropy: float
    clinical_status: str
    status_description: str
    gradcam_overlay: str  # base64-encoded PNG image
    original_image: Optional[str] = None
    heatmap_pure: Optional[str] = None
    disclaimer: str = "Research and educational project only. Not intended for clinical or diagnostic use."


class TriageItem(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    predictive_entropy: float
    clinical_status: str
    triage_priority: str
    priority_level: int  # 1 = Critical, 2 = Routine Oncology, 3 = Normal Clearance


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {e}",
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    colormap: str = Query("jet", description="Colormap for Grad-CAM: jet, inferno, viridis, turbo"),
    alpha: float = Query(0.4, ge=0.0, le=1.0),
):
    global model
    if model is None:
        load_classifier_model()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Train and save model to saved_models/best_model.keras first.",
            )

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    # Compute Bayesian uncertainty via selective Monte Carlo Dropout
    uncertainty_result = compute_mc_dropout_uncertainty(
        model=model,
        img_array=img_array,
        n_iterations=20,
        class_names=CLASS_NAMES,
    )

    pred_index = CLASS_NAMES.index(uncertainty_result["predicted_class"])

    # Generate Grad-CAM heatmap
    heatmap, _ = make_gradcam_heatmap(img_array, model, pred_index=pred_index)
    original = (img_array[0] * 255.0).astype(np.uint8)
    overlaid = overlay_heatmap(original, heatmap, alpha=alpha, colormap=colormap)

    # Generate pure colored heatmap (without background)
    cmap_code = COLORMAP_DICT.get(colormap.lower(), cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cmap_code)

    # Base64 encodings
    _, orig_buf = cv2.imencode(".png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    _, over_buf = cv2.imencode(".png", overlaid)
    _, heat_buf = cv2.imencode(".png", heatmap_colored)

    orig_b64 = base64.b64encode(orig_buf).decode("utf-8")
    overlay_b64 = base64.b64encode(over_buf).decode("utf-8")
    heat_b64 = base64.b64encode(heat_buf).decode("utf-8")

    return PredictionResponse(
        predicted_class=uncertainty_result["predicted_class"],
        confidence=uncertainty_result["confidence"],
        probabilities=uncertainty_result["mean_probabilities"],
        std_probabilities=uncertainty_result["std_probabilities"],
        epistemic_uncertainty=uncertainty_result["epistemic_uncertainty"],
        predictive_entropy=uncertainty_result["predictive_entropy"],
        clinical_status=uncertainty_result["clinical_status"],
        status_description=uncertainty_result["status_description"],
        gradcam_overlay=overlay_b64,
        original_image=orig_b64,
        heatmap_pure=heat_b64,
    )


@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    global model
    if model is None:
        load_classifier_model()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Train and save model to saved_models/best_model.keras first.",
            )

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    uncertainty_result = compute_mc_dropout_uncertainty(
        model=model,
        img_array=img_array,
        n_iterations=20,
        class_names=CLASS_NAMES,
    )

    pred_index = CLASS_NAMES.index(uncertainty_result["predicted_class"])
    heatmap, _ = make_gradcam_heatmap(img_array, model, pred_index=pred_index)
    original = (img_array[0] * 255.0).astype(np.uint8)
    overlaid = overlay_heatmap(original, heatmap)

    _, overlay_buf = cv2.imencode(".png", overlaid)
    gradcam_bytes = overlay_buf.tobytes()

    pdf_content = generate_clinical_pdf_report(
        prediction_data=uncertainty_result,
        original_img_bytes=image_bytes,
        gradcam_img_bytes=gradcam_bytes,
    )

    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=clinical_diagnostic_report.pdf"},
    )


@app.get("/samples")
def get_sample_scans():
    """Retrieve preloaded authentic sample MRI scans for 1-click testing."""
    sample_files = {
        "glioma": "data/raw/glioma/Testing_Te-gl_1.jpg",
        "meningioma": "data/raw/meningioma/Testing_Te-aug-me_1.jpg",
        "pituitary": "data/raw/pituitary/Testing_Te-pi_1.jpg",
        "no_tumor": "data/raw/no_tumor/Testing_Te-no_1.jpg",
    }
    result = {}
    for cls, path in sample_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            result[cls] = {
                "class": cls,
                "filename": os.path.basename(path),
                "image_b64": b64,
            }
    return result


@app.post("/triage", response_model=List[TriageItem])
async def simulate_triage_queue():
    """Simulate a multi-patient emergency department triage queue."""
    global model
    if model is None:
        load_classifier_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model unavailable")

    triage_results = []
    sample_dirs = {
        "glioma": glob.glob("data/raw/glioma/*.jpg")[:3],
        "meningioma": glob.glob("data/raw/meningioma/*.jpg")[:3],
        "pituitary": glob.glob("data/raw/pituitary/*.jpg")[:2],
        "no_tumor": glob.glob("data/raw/no_tumor/*.jpg")[:2],
    }

    for true_cls, flist in sample_dirs.items():
        for fpath in flist:
            try:
                with open(fpath, "rb") as fp:
                    raw_bytes = fp.read()
                img_array = preprocess_image(raw_bytes)
                res = compute_mc_dropout_uncertainty(model, img_array, n_iterations=10, class_names=CLASS_NAMES)
                
                pred_cls = res["predicted_class"]
                conf = res["confidence"]
                entropy = res["predictive_entropy"]
                status_str = res["clinical_status"]

                if "HIGH" in status_str or (pred_cls != "no_tumor" and conf > 0.95):
                    priority = "🚨 STAT Emergency Review"
                    level = 1
                elif pred_cls != "no_tumor":
                    priority = "⚠️ Priority Oncology"
                    level = 2
                else:
                    priority = "🟢 Routine Clearance"
                    level = 3

                triage_results.append(TriageItem(
                    filename=os.path.basename(fpath),
                    predicted_class=pred_cls,
                    confidence=conf,
                    predictive_entropy=entropy,
                    clinical_status=status_str,
                    triage_priority=priority,
                    priority_level=level,
                ))
            except Exception:
                continue

    # Sort by priority level (1 first) then by entropy descending
    triage_results.sort(key=lambda x: (x.priority_level, -x.predictive_entropy))
    return triage_results


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "features": ["gradcam", "mc_dropout_uncertainty", "clinical_pdf_reports", "pacs_viewer", "triage_queue"],
    }


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}


# Mount modern static Web SPA
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "web")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")



