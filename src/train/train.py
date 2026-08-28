"""Training entry point. Runs k-fold cross-validation and reports mean and
variance across folds, a single train/test split accuracy is not enough
evidence on its own.
"""

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.baseline_cnn import build_baseline_cnn
from src.models.transfer_models import build_resnet50, build_efficientnet_b0

MODEL_BUILDERS = {
    "baseline_cnn": build_baseline_cnn,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
}


def run_kfold(model_name: str, images: np.ndarray, labels: np.ndarray, k: int = 5, epochs: int = 15):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images, labels)):
        print(f"Fold {fold_idx + 1}/{k}")
        model = MODEL_BUILDERS[model_name]()

        x_train, x_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1,
        )

        _, val_acc = model.evaluate(x_val, y_val, verbose=0)
        fold_accuracies.append(val_acc)

    results = {
        "model": model_name,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
    }
    return results


if __name__ == "__main__":
    # Wire this up to your loaded dataset arrays once data/raw is populated
    # with the Kaggle Brain Tumor MRI Dataset.
    print("Load images/labels as numpy arrays, then call run_kfold(model_name, images, labels)")
