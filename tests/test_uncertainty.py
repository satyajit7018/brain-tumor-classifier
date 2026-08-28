"""Unit tests for Monte Carlo Dropout Bayesian Uncertainty Estimation."""

import unittest
import numpy as np
import tensorflow as tf

from src.eval.uncertainty import compute_mc_dropout_uncertainty


class TestUncertaintyModule(unittest.TestCase):
    def setUp(self):
        # Create a toy model with Dropout
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4, activation="softmax"),
        ])
        self.dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)

    def test_compute_mc_dropout_uncertainty(self):
        result = compute_mc_dropout_uncertainty(
            model=self.model,
            img_array=self.dummy_img,
            n_iterations=5,
            class_names=["glioma", "meningioma", "pituitary", "no_tumor"],
        )

        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("mean_probabilities", result)
        self.assertIn("std_probabilities", result)
        self.assertIn("epistemic_uncertainty", result)
        self.assertIn("predictive_entropy", result)
        self.assertIn("clinical_status", result)
        self.assertEqual(len(result["mean_probabilities"]), 4)
        self.assertEqual(len(result["std_probabilities"]), 4)
        self.assertAlmostEqual(sum(result["mean_probabilities"].values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
