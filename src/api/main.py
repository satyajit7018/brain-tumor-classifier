import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from pydantic import BaseModel

from src.eval.gradcam import make_gradcam_heatmap, overlay_heatmap

app = FastAPI(title="Brain Tumor Classifier API")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "resnet50"  # update to match whichever model is loaded

model = None  # load with tf.keras.models.load_model("saved_models/best_model.keras")


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_overlay: str  # base64-encoded PNG


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array)
    pred_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_index])

    heatmap, _ = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER, pred_index)
    original = (img_array[0] * 255).astype(np.uint8)
    overlaid = overlay_heatmap(original, heatmap)

    _, buffer = cv2.imencode(".png", overlaid)
    overlay_b64 = base64.b64encode(buffer).decode("utf-8")

    return PredictionResponse(
        predicted_class=CLASS_NAMES[pred_index],
        confidence=confidence,
        gradcam_overlay=overlay_b64,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
