"""Dataset loading and splitting. Checks class balance explicitly instead
of assuming the Kaggle Brain Tumor MRI Dataset is balanced.
"""

import os
from collections import Counter

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# download_dataset.py keeps the Kaggle Training/Testing origin as a filename
# prefix; Testing_* files form the held-out test set.
TEST_PREFIX = "Testing_"



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


def _md5(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def split_by_source(data_dir: str, test_prefix: str = TEST_PREFIX):
    """Split data_dir into (train_items, test_items) using the dataset authors'
    original Training/Testing split, which download_dataset.py preserves as a
    filename prefix. Each item is a (path, label_index) tuple.

    Test images that are byte-identical to any training image are dropped, so
    no test scan is ever seen during training. Returns
    (train_items, test_items, n_dropped_duplicates).
    """
    train_items, test_items = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            item = (os.path.join(class_dir, fname), class_idx)
            (test_items if fname.startswith(test_prefix) else train_items).append(item)

    train_hashes = {_md5(path) for path, _ in train_items}
    clean_test = [item for item in test_items if _md5(item[0]) not in train_hashes]
    return train_items, clean_test, len(test_items) - len(clean_test)


def load_images(items, img_size=IMG_SIZE):
    """Load (path, label) items into (images uint8 [N,H,W,3], labels int64 [N])."""
    import numpy as np
    from PIL import Image

    images = np.empty((len(items), *img_size, 3), dtype=np.uint8)
    labels = np.empty((len(items),), dtype=np.int64)
    for i, (path, label) in enumerate(items):
        with Image.open(path) as img:
            images[i] = np.asarray(img.convert("RGB").resize(img_size))
        labels[i] = label
    return images, labels


def load_datasets(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.15, seed=42):
    """Build train/val tf.data pipelines from the TRAINING split only. The
    held-out test split is never touched here; evaluate it with
    scripts/evaluate_final.py.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    train_items, _, _ = split_by_source(data_dir)
    images, labels = load_images(train_items, img_size)
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=val_split, stratify=labels, random_state=seed
    )

    def to_ds(x, y, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b))
        return ds.prefetch(tf.data.AUTOTUNE)

    return to_ds(x_train, y_train, True), to_ds(x_val, y_val, False)


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


def load_dataset_as_numpy(data_dir: str, img_size=IMG_SIZE, max_samples_per_class: int = None):
    """Load all images and labels from class subdirectories into numpy arrays.
    Returns:
        images: np.ndarray of shape (N, H, W, 3) in [0.0, 1.0]
        labels: np.ndarray of shape (N,) integer class indices
    """
    import numpy as np
    from PIL import Image

    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        filenames = [
            f for f in sorted(os.listdir(class_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if max_samples_per_class:
            filenames = filenames[:max_samples_per_class]

        for fname in filenames:
            fpath = os.path.join(class_dir, fname)
            try:
                with Image.open(fpath) as img:
                    img_rgb = img.convert("RGB").resize(img_size)
                    arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
                    images.append(arr)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Warning: could not load {fpath}: {e}")

    if not images:
        return np.empty((0, *img_size, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


if __name__ == "__main__":
    counts = check_class_balance("data/raw")
    print("Class balance:", counts)
    print("Total:", sum(counts.values()))

