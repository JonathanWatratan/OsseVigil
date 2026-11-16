#!/usr/bin/env python3
"""OsteoInsight localization-only FastAPI service.

The legacy Roboflow classifier is intentionally not imported or used here.
Only a trained BTXRD object detector may produce `/api/predict` responses.
"""

from __future__ import annotations

import io
import os
import hashlib
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field


API_SCHEMA_VERSION = "osteoinsight.predict.v1"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "btxrd_yolov8n_best.pt"
DISCLAIMER = (
    "Research prototype only. Boxes are model outputs trained to reproduce BTXRD lesion annotations; "
    "they do not diagnose cancer, determine malignancy, stage disease, establish urgency, or recommend treatment. "
    "No detection does not mean lesion-free or cancer-free."
)


class NormalizedBox(BaseModel):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)


class PixelBox(BaseModel):
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)


class Detection(BaseModel):
    label: Literal["potential_bone_lesion"] = "potential_bone_lesion"
    class_id: Literal[0] = 0
    score: float = Field(ge=0, le=1, description="Uncalibrated detector confidence")
    bbox: list[float] = Field(
        min_length=4, max_length=4,
        description="Normalized [x_min, y_min, x_max, y_max] coordinates",
    )
    bbox_named: NormalizedBox
    bbox_pixels: PixelBox


class ImageInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ModelInfo(BaseModel):
    name: Literal["btxrd-yolov8n-lesion-localizer"] = "btxrd-yolov8n-lesion-localizer"
    task: Literal["bone_lesion_localization"] = "bone_lesion_localization"
    training_dataset: Literal["BTXRD"] = "BTXRD"


class PredictResponse(BaseModel):
    schema_version: Literal["osteoinsight.predict.v1"] = API_SCHEMA_VERSION
    model: ModelInfo
    model_status: Literal["ready"] = "ready"
    model_version: str
    image: ImageInfo
    image_width: int
    image_height: int
    operating_threshold: float
    inference_ms: float = Field(ge=0)
    detections: list[Detection]
    warnings: list[str]
    flag: Literal[
        "potential_lesion_requires_professional_review",
        "no_detection_above_threshold_not_a_clearance",
    ]
    disclaimer: str = DISCLAIMER


class DetectorService:
    def __init__(self, checkpoint: Path, device: str = ""):
        self.checkpoint = checkpoint
        self.device = device or None
        self._model = None
        self._model_version = "unavailable"
        self.load_error = ""
        if checkpoint.is_file():
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self._model_version = f"btxrd-yolov8n-{digest.hexdigest()[:12]}"
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(checkpoint))
            except Exception as exc:  # surfaced through health and 503 response
                self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        if not self.ready:
            raise RuntimeError(self.load_error or f"Detector checkpoint not found: {self.checkpoint}")
        width, height = image.size
        result = self._model.predict(
            np.asarray(image.convert("RGB")), conf=threshold,
            device=self.device, verbose=False,
        )[0]
        output = []
        if result.boxes is None:
            return output
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, scores):
            x1, y1, x2, y2 = map(float, box)
            x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
            y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
            if x2 <= x1 or y2 <= y1:
                continue
            normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
            output.append(Detection(
                score=float(np.clip(score, 0, 1)),
                bbox=normalized,
                bbox_named=NormalizedBox(x_min=normalized[0], y_min=normalized[1], x_max=normalized[2], y_max=normalized[3]),
                bbox_pixels=PixelBox(
                    x_min=int(round(x1)), y_min=int(round(y1)),
                    x_max=int(round(x2)), y_max=int(round(y2)),
                ),
            ))
        return output


checkpoint = Path(os.environ.get("OSTEOINSIGHT_DETECTOR_PATH", DEFAULT_CHECKPOINT)).expanduser().resolve()
detector = DetectorService(checkpoint, os.environ.get("OSTEOINSIGHT_DEVICE", ""))
app = FastAPI(
    title="OsteoInsight Lesion Localization API",
    version="1.0.0",
    description="Research-only BTXRD bone-lesion localization. No clinical diagnosis endpoint is provided.",
)
origins = [value.strip() for value in os.environ.get(
    "OSTEOINSIGHT_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
).split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ready" if detector.ready else "model_not_ready",
        "task": "bone_lesion_localization",
        "model_status": "ready" if detector.ready else "model_not_ready",
        "model_version": detector.model_version,
        "checkpoint": str(detector.checkpoint),
        "detail": detector.load_error or ("loaded" if detector.ready else "BTXRD detector checkpoint is missing"),
        "legacy_classifier_enabled": False,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="One JPEG or PNG bone radiograph"),
    threshold: float = Query(0.25, ge=0.05, le=0.95, description="Declared detector score threshold"),
):
    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_not_ready",
                "message": "The BTXRD lesion detector has not been trained or loaded.",
                "checkpoint": str(detector.checkpoint),
                "legacy_classifier_fallback": False,
            },
        )
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail={"code": "empty_file", "message": "Uploaded file is empty."})
    if len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": "Maximum upload size is 25 MB."})
    try:
        with Image.open(io.BytesIO(payload)) as decoded:
            image = ImageOps.exif_transpose(decoded).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=415, detail={"code": "invalid_image", "message": "Upload must be a readable JPEG or PNG image."})
    started = time.perf_counter()
    detections = detector.predict(image, threshold)
    inference_ms = (time.perf_counter() - started) * 1000
    warnings = [
        "Detection scores are uncalibrated model outputs, not probabilities of cancer.",
        "Bounding boxes localize image regions similar to BTXRD annotations and do not establish malignancy.",
    ]
    if not detections:
        warnings.append("No detection above threshold does not rule out a lesion or cancer.")
    return PredictResponse(
        model=ModelInfo(), model_version=detector.model_version,
        image=ImageInfo(width=image.width, height=image.height),
        image_width=image.width, image_height=image.height,
        operating_threshold=threshold, inference_ms=inference_ms,
        detections=detections, warnings=warnings,
        flag=(
            "potential_lesion_requires_professional_review"
            if detections else "no_detection_above_threshold_not_a_clearance"
        ),
    )
