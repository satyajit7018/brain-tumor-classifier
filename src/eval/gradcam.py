"""Grad-CAM: shows which regions of the MRI the model attended to for a
given prediction. Run this on both correct and incorrect predictions,
a case where the model is right for the wrong reason is worth documenting.
"""

import cv2
import numpy as np
import tensorflow as tf


def find_target_conv_layer(model: tf.keras.Model, layer_name: str = None):
    """Locate the target conv layer or feature map tensor in the model,
    searching top-level layers and nested sub-models.
    """
    if layer_name:
        # Check direct model layers
        try:
            layer = model.get_layer(layer_name)
            return layer.output
        except (ValueError, AttributeError):
            pass

        # Check submodel layers (e.g. ResNet50 / EfficientNet submodels)
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
                try:
                    sub_layer = layer.get_layer(layer_name)
                    # To get output in outer model context:
                    return layer.output
                except (ValueError, AttributeError):
                    pass

    # Automatic fallback: find the last 4D conv/feature map layer
    for layer in reversed(model.layers):
        if hasattr(layer, "output") and len(getattr(layer.output, "shape", [])) == 4:
            return layer.output
        if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if hasattr(sub_layer, "output") and len(getattr(sub_layer.output, "shape", [])) == 4:
                    return layer.output

    raise ValueError(f"Could not find a convolutional feature map layer in {model.name}")


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str = None, pred_index=None):
    """Generate Grad-CAM heatmap showing where the model attended.
    Args:
        img_array: Preprocessed input tensor of shape (1, H, W, 3)
        model: Trained Keras model
        last_conv_layer_name: Optional name of the target conv layer
        pred_index: Optional target class index (defaults to argmax prediction)
    Returns:
        heatmap: 2D numpy array normalized to [0.0, 1.0]
        pred_index: Class index evaluated
    """
    target_output_tensor = find_target_conv_layer(model, last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_output_tensor, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    original_uint8 = np.uint8(original_img * 255) if original_img.max() <= 1.0 else original_img.astype(np.uint8)
    overlaid = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlaid

