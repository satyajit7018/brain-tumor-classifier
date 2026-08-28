"""Dataset loading and splitting. Checks class balance explicitly instead
of assuming the Kaggle Brain Tumor MRI Dataset is balanced.
"""

import os
from collections import Counter

import tensorflow as tf

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def check_class_balance(data_dir: str) -> dict:
    """Count images per class. Run this before training and record the
    result, do not assume the dataset is balanced.
    """
    counts = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.isdir(cls_dir):
            counts[cls] = len(
                [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            )
        else:
            counts[cls] = 0
    return counts


def load_datasets(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.2, seed=42):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        class_names=CLASS_NAMES,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        class_names=CLASS_NAMES,
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def compute_class_weights(counts: dict) -> dict:
    """Inverse-frequency class weights, pass to model.fit(class_weight=...)
    when the balance check shows a meaningful skew.
    """
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {}
    for i, cls in enumerate(CLASS_NAMES):
        count = counts.get(cls, 1)
        weights[i] = total / (n_classes * max(count, 1))
    return weights


if __name__ == "__main__":
    counts = check_class_balance("data/raw")
    print("Class balance:", counts)
    print("Total:", sum(counts.values()))
