"""Archived contract tests for the preserved localization research API."""

from __future__ import annotations

import io
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import api


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeBoxes:
    xyxy = _FakeTensor([[20, 10, 120, 60]])
    conf = _FakeTensor([0.73])


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeModel:
    def predict(self, *_args, **_kwargs):
        return [_FakeResult()]


class _EmptyResult:
    boxes = None


class _EmptyModel:
    def predict(self, *_args, **_kwargs):
        return [_EmptyResult()]


@unittest.skip("Archived localization API is no longer the product entrypoint")
class LocalizationApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        image = Image.new("RGB", (200, 100), "black")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        cls.image_bytes = buffer.getvalue()

    def setUp(self):
        self.original_model = api.detector._model

    def tearDown(self):
        api.detector._model = self.original_model

    def upload(self):
        return {"file": ("test.png", self.image_bytes, "image/png")}

    def test_unavailable_detector_returns_503_without_legacy_fallback(self):
        api.detector._model = None
        response = self.client.post("/api/predict?threshold=0.25", files=self.upload())
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["detail"]["legacy_classifier_fallback"])

    def test_detection_serializes_normalized_and_pixel_boxes(self):
        api.detector._model = _FakeModel()
        response = self.client.post("/api/predict?threshold=0.25", files=self.upload())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["image_width"], 200)
        self.assertEqual(body["image_height"], 100)
        self.assertEqual(body["detections"][0]["bbox"], [0.1, 0.1, 0.6, 0.6])
        self.assertEqual(
            body["detections"][0]["bbox_named"],
            {"x_min": 0.1, "y_min": 0.1, "x_max": 0.6, "y_max": 0.6},
        )
        self.assertEqual(
            body["detections"][0]["bbox_pixels"],
            {"x_min": 20, "y_min": 10, "x_max": 120, "y_max": 60},
        )
        self.assertEqual(body["flag"], "potential_lesion_requires_professional_review")
        self.assertTrue(body["warnings"])

    def test_no_detection_is_not_reported_as_clearance(self):
        api.detector._model = _EmptyModel()
        response = self.client.post("/api/predict?threshold=0.25", files=self.upload())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detections"], [])
        self.assertEqual(body["flag"], "no_detection_above_threshold_not_a_clearance")
        self.assertTrue(any("does not rule out" in warning.lower() for warning in body["warnings"]))


if __name__ == "__main__":
    unittest.main()
