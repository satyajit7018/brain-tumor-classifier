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


if __name__ == "__main__":
    unittest.main()
