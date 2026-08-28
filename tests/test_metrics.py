"""Unit tests for evaluation metrics and false-negative rate calculations."""

import unittest
import numpy as np

from src.eval.metrics import (
    CLASS_NAMES,
    NO_TUMOR_INDEX,
    compute_false_negative_rate,
    evaluate_model,
)


class TestMetricsModule(unittest.TestCase):
    def test_false_negative_rate_zero(self):
        # All tumor cases are predicted correctly as tumors
        y_true = np.array([0, 1, 2, 3])  # 0,1,2 are tumors, 3 is no_tumor
        y_pred = np.array([0, 1, 2, 3])
        fnr = compute_false_negative_rate(y_true, y_pred)
        self.assertEqual(fnr, 0.0)

    def test_false_negative_rate_high(self):
        # 2 out of 3 tumors are missed (predicted as no_tumor)
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([NO_TUMOR_INDEX, NO_TUMOR_INDEX, 2, 3])
        fnr = compute_false_negative_rate(y_true, y_pred)
        self.assertAlmostEqual(fnr, 2.0 / 3.0, places=4)

    def test_evaluate_model_output_structure(self):
        # 10 samples
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        # Synthetic probabilities
        y_pred_probs = np.zeros((10, 4))
        for i, t in enumerate(y_true):
            y_pred_probs[i, t] = 0.8
            y_pred_probs[i, (t + 1) % 4] = 0.2

        results = evaluate_model(y_true, y_pred_probs)
        self.assertIn("confusion_matrix", results)
        self.assertIn("classification_report", results)
        self.assertIn("roc_auc_per_class", results)
        self.assertIn("false_negative_rate", results)
        self.assertEqual(len(results["confusion_matrix"]), 4)


if __name__ == "__main__":
    unittest.main()
