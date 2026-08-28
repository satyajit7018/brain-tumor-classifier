"""Unit tests for FastAPI endpoints."""

import unittest
from fastapi.testclient import TestClient
from src.api.main import app, CLASS_NAMES


class TestAPIModule(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model_loaded", data)
        self.assertEqual(data["classes"], CLASS_NAMES)

    def test_classes_endpoint(self):
        response = self.client.get("/classes")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["classes"], CLASS_NAMES)

    def test_predict_endpoint_contract(self):
        import io
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        # If model exists it returns 200 with complete clinical schema; if not loaded 503
        self.assertIn(response.status_code, [200, 503])
        if response.status_code == 200:
            data = response.json()
            self.assertIn("predicted_class", data)
            self.assertIn("confidence", data)
            self.assertIn("probabilities", data)
            self.assertIn("epistemic_uncertainty", data)
            self.assertIn("gradcam_overlay", data)

    def test_report_endpoint_contract(self):
        import io
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post("/report", files={"file": ("test.jpg", buf, "image/jpeg")})
        self.assertIn(response.status_code, [200, 503])
        if response.status_code == 200:
            self.assertEqual(response.headers.get("content-type"), "application/pdf")
            self.assertGreater(len(response.content), 1000)


    def test_samples_endpoint(self):
        response = self.client.get("/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("neuroscan_requests_total", data)
        self.assertIn("neuroscan_cache_hits_total", data)
        self.assertIn("neuroscan_avg_latency_ms", data)

    def test_predict_with_tta(self):
        import io
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post("/predict?use_tta=true&colormap=inferno", files={"file": ("test.jpg", buf, "image/jpeg")})
        self.assertIn(response.status_code, [200, 503])
        if response.status_code == 200:
            data = response.json()
            self.assertIn("predicted_class", data)
            self.assertIn("confidence", data)


if __name__ == "__main__":
    unittest.main()


