"""Product API tests for successful, empty, invalid, and failed inference."""

from __future__ import annotations

import io
import unittest

import torch
from fastapi.testclient import TestClient
from PIL import Image

import api_classification as api


class _FixedModel(torch.nn.Module):
    def forward(self, images):
        return torch.tensor([[2.0, 1.0]], device=images.device).repeat(images.shape[0], 1)


class _BrokenModel(torch.nn.Module):
    def forward(self, _images):
        raise RuntimeError("deliberate test failure")


class ClassificationApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        image = Image.new("RGB", (120, 80), "gray")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        cls.image_bytes = buffer.getvalue()

    def setUp(self):
        self.original_model = api.classifier.model

    def tearDown(self):
        api.classifier.model = self.original_model

    def test_success_returns_real_two_category_contract(self):
        api.classifier.model = _FixedModel().to(api.classifier.device).eval()
        response = self.client.post(
            "/api/predict",
            files={"file": ("sample.png", self.image_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["schema_version"], "osteoinsight.classification.v1")
        self.assertEqual(body["predicted_category"], "cancer")
        self.assertEqual([item["category"] for item in body["scores"]], ["cancer", "normal"])
        self.assertAlmostEqual(sum(item["score"] for item in body["scores"]), 1.0, places=5)
        self.assertIn("not the probability of cancer", " ".join(body["warnings"]).lower())

    def test_empty_upload_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "empty_file")

    def test_invalid_image_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("fake.png", b"not an image", "image/png")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"]["code"], "invalid_image")

    def test_inference_failure_returns_clear_error(self):
        api.classifier.model = _BrokenModel().to(api.classifier.device).eval()
        response = self.client.post(
            "/api/predict",
            files={"file": ("sample.png", self.image_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "inference_failed")


if __name__ == "__main__":
    unittest.main()
