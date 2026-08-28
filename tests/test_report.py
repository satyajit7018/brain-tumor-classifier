"""Unit tests for automated clinical PDF report generation."""

import io
import unittest
from PIL import Image

from src.eval.report_generator import generate_clinical_pdf_report


class TestReportGeneratorModule(unittest.TestCase):
    def setUp(self):
        # Generate dummy PNG images
        img = Image.new("RGB", (128, 128), color=(80, 80, 120))
        buf1 = io.BytesIO()
        img.save(buf1, format="PNG")
        self.dummy_orig_bytes = buf1.getvalue()

        img2 = Image.new("RGB", (128, 128), color=(200, 50, 50))
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        self.dummy_grad_bytes = buf2.getvalue()

        self.dummy_prediction_data = {
            "predicted_class": "glioma",
            "confidence": 0.885,
            "mean_probabilities": {"glioma": 0.885, "meningioma": 0.05, "pituitary": 0.04, "no_tumor": 0.025},
            "std_probabilities": {"glioma": 0.03, "meningioma": 0.01, "pituitary": 0.01, "no_tumor": 0.005},
            "epistemic_uncertainty": 0.0137,
            "predictive_entropy": 0.28,
            "clinical_status": "LOW_RISK_CONFIDENT",
            "status_description": "Low epistemic uncertainty. Model predictions demonstrate stable stochastic convergence.",
        }

    def test_generate_clinical_pdf_report(self):
        pdf_bytes = generate_clinical_pdf_report(
            prediction_data=self.dummy_prediction_data,
            original_img_bytes=self.dummy_orig_bytes,
            gradcam_img_bytes=self.dummy_grad_bytes,
            case_id="TEST-CASE-001",
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
