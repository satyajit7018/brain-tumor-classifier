"""Clinical Uncertainty Estimation via Monte Carlo Dropout (Bayesian Inference).
Runs multiple stochastic forward passes with dropout enabled at inference time to
quantify epistemic uncertainty, predictive entropy, and flag ambiguous clinical scans.
"""

from typing import Dict, Any, List
import numpy as np
import tensorflow as tf


def compute_mc_dropout_uncertainty(
    model: tf.keras.Model,
    img_array: np.ndarray,
    n_iterations: int = 20,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """Execute Monte Carlo Dropout inference across n_iterations.
    Args:
        model: Trained Keras model with Dropout layer(s)
        img_array: Preprocessed image tensor of shape (1, 224, 224, 3) in [0.0, 1.0]
        n_iterations: Number of stochastic passes (default: 20)
        class_names: List of class labels (e.g. ['glioma', 'meningioma', 'pituitary', 'no_tumor'])
    Returns:
        Dictionary containing mean probabilities, standard deviations, predictive entropy,
        and clinical safety risk classification.
    """
    if class_names is None:
        class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

    predictions = []
    for _ in range(n_iterations):
        # Enable training=True so Dropout is active
        pred = model(img_array, training=True)
        predictions.append(pred.numpy()[0])

    predictions = np.array(predictions)  # Shape: (n_iterations, num_classes)

    mean_probs = np.mean(predictions, axis=0)
    std_probs = np.std(predictions, axis=0)

    pred_index = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_index])

    # Epistemic uncertainty metrics
    epistemic_uncertainty = float(np.mean(std_probs))
    max_class_std = float(np.max(std_probs))

    # Normalized Shannon Predictive Entropy (0.0 = completely certain, 1.0 = maximum ambiguity)
    eps = 1e-12
    num_classes = len(class_names)
    entropy = -float(np.sum(mean_probs * np.log2(mean_probs + eps)))
    max_entropy = np.log2(num_classes)
    normalized_entropy = float(entropy / max_entropy)

    # Clinical Alert Assessment
    if normalized_entropy >= 0.60 or max_class_std >= 0.15:
        clinical_status = "HIGH_RISK_RADIOLOGIST_REVIEW"
        status_description = "High predictive ambiguity. Automated diagnosis should NOT be used without specialist radiologist validation."
    elif normalized_entropy >= 0.35 or max_class_std >= 0.08:
        clinical_status = "MODERATE_AMBIGUITY"
        status_description = "Moderate uncertainty detected. Review scan attention regions and clinical history."
    else:
        clinical_status = "LOW_RISK_CONFIDENT"
        status_description = "Low epistemic uncertainty. Model predictions demonstrate stable stochastic convergence."

    prob_dict = {class_names[i]: float(mean_probs[i]) for i in range(num_classes)}
    std_dict = {class_names[i]: float(std_probs[i]) for i in range(num_classes)}

    return {
        "predicted_class": class_names[pred_index],
        "confidence": confidence,
        "mean_probabilities": prob_dict,
        "std_probabilities": std_dict,
        "epistemic_uncertainty": epistemic_uncertainty,
        "predictive_entropy": normalized_entropy,
        "clinical_status": clinical_status,
        "status_description": status_description,
        "n_passes": n_iterations,
    }
