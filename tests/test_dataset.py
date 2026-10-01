"""Unit tests for dataset loading, class balancing, and weight calculations."""

import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from src.data.dataset import (
    CLASS_NAMES,
    check_class_balance,
    compute_class_weights,
    load_dataset_as_numpy,
    split_by_source,
)


class TestDatasetModule(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        for cls in CLASS_NAMES:
            cls_path = os.path.join(self.test_dir, cls)
            os.makedirs(cls_path, exist_ok=True)
            # Create a sample test image
            img = Image.new("RGB", (64, 64), color=(100, 100, 100))
            img.save(os.path.join(cls_path, "sample1.jpg"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_check_class_balance(self):
        counts = check_class_balance(self.test_dir)
        for cls in CLASS_NAMES:
            self.assertEqual(counts[cls], 1)
        self.assertEqual(sum(counts.values()), 4)

    def test_compute_class_weights_balanced(self):
        counts = {cls: 100 for cls in CLASS_NAMES}
        weights = compute_class_weights(counts)
        for i in range(len(CLASS_NAMES)):
            self.assertAlmostEqual(weights[i], 1.0, places=4)

    def test_compute_class_weights_imbalanced(self):
        counts = {"glioma": 500, "meningioma": 250, "pituitary": 200, "no_tumor": 50}
        weights = compute_class_weights(counts)
        # Rare class (no_tumor) should have the highest weight
        no_tumor_idx = CLASS_NAMES.index("no_tumor")
        glioma_idx = CLASS_NAMES.index("glioma")
        self.assertGreater(weights[no_tumor_idx], weights[glioma_idx])

    def test_load_dataset_as_numpy(self):
        images, labels = load_dataset_as_numpy(self.test_dir, img_size=(32, 32))
        self.assertEqual(len(images), 4)
        self.assertEqual(len(labels), 4)
        self.assertEqual(images.shape, (4, 32, 32, 3))
        self.assertTrue(0.0 <= images.min() <= images.max() <= 1.0)


class TestSplitBySource(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        for i, cls in enumerate(CLASS_NAMES):
            cls_path = os.path.join(self.test_dir, cls)
            os.makedirs(cls_path, exist_ok=True)
            Image.new("RGB", (8, 8), color=(i, 1, 1)).save(os.path.join(cls_path, "Training_a.png"))
            Image.new("RGB", (8, 8), color=(i, 2, 2)).save(os.path.join(cls_path, "Testing_b.png"))
        # A test image byte-identical to a training image must be dropped.
        shutil.copy(
            os.path.join(self.test_dir, "glioma", "Training_a.png"),
            os.path.join(self.test_dir, "glioma", "Testing_dup.png"),
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_train_and_test_are_disjoint(self):
        train, test, dropped = split_by_source(self.test_dir)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 4)
        self.assertEqual(dropped, 1)
        self.assertTrue(all(os.path.basename(p).startswith("Training_") for p, _ in train))
        self.assertTrue(all(os.path.basename(p).startswith("Testing_") for p, _ in test))
        self.assertFalse({p for p, _ in train} & {p for p, _ in test})


if __name__ == "__main__":
    unittest.main()
