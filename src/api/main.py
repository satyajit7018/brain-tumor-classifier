import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Response
from PIL import Image
from pydantic import BaseModel

from src.eval.gradcam import make_gradcam_heatmap, overlay_heatmap
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
    title="Brain Tumor MRI Classifier API",
    description="Multi-class brain tumor classification with Grad-CAM explainability, Bayesian uncertainty, and automated PDF reporting.",
    version="1.1.0",
    lifespan=lifespan,
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
    disclaimer: str = "Research and educational project only. Not intended for clinical or diagnostic use."


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
async def predict(file: UploadFile = File(...)):
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

    # Compute Bayesian uncertainty via Monte Carlo Dropout
    uncertainty_result = compute_mc_dropout_uncertainty(
        model=model,
        img_array=img_array,
        n_iterations=20,
        class_names=CLASS_NAMES,
    )

    pred_index = CLASS_NAMES.index(uncertainty_result["predicted_class"])

    # Generate Grad-CAM heatmap with auto layer resolution
    heatmap, _ = make_gradcam_heatmap(img_array, model, pred_index=pred_index)
    original = (img_array[0] * 255.0).astype(np.uint8)
    overlaid = overlay_heatmap(original, heatmap)

    # Encode to base64 PNG
    _, buffer = cv2.imencode(".png", overlaid)
    overlay_b64 = base64.b64encode(buffer).decode("utf-8")

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "features": ["gradcam", "mc_dropout_uncertainty", "clinical_pdf_reports"],
    }


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}


