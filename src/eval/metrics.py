"""Evaluation suite. False negative rate is reported as the primary metric,
not accuracy, since missing a tumor is a categorically worse error than a
false alarm in this context.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
NO_TUMOR_INDEX = CLASS_NAMES.index("no_tumor")


def evaluate_model(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    # ROC-AUC per class, one-vs-rest
    y_true_onehot = np.eye(len(CLASS_NAMES))[y_true]
    auc_per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        try:
            auc_per_class[cls] = roc_auc_score(y_true_onehot[:, i], y_pred_probs[:, i])
        except ValueError:
            auc_per_class[cls] = None

    false_negative_rate = compute_false_negative_rate(y_true, y_pred)

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_auc_per_class": auc_per_class,
        "false_negative_rate": false_negative_rate,
    }


def compute_false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of actual tumor cases (any class other than no_tumor) that
    were predicted as no_tumor. This is the number to report front and
    center, a missed tumor is worse than a false alarm.
    """
    actual_tumor_mask = y_true != NO_TUMOR_INDEX
    if actual_tumor_mask.sum() == 0:
        return 0.0
    predicted_no_tumor = y_pred[actual_tumor_mask] == NO_TUMOR_INDEX
    return float(predicted_no_tumor.sum() / actual_tumor_mask.sum())
