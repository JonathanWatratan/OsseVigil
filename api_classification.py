#!/usr/bin/env python3
"""Research-only FastAPI service for the archived ResNet18 classifier.

The two outputs reproduce the source dataset categories (``cancer`` and
``normal``). They are not clinical probabilities or diagnoses, and this model
has not been validated on a fracture challenge set or independent patients.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from osteoinsight.common import select_device
from osteoinsight.data import make_transforms
from osteoinsight.modeling import load_checkpoint


API_SCHEMA_VERSION = "osteoinsight.classification.v1"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "osteoinsight_best.pth"
DISCLAIMER = (
    "Research prototype only. These are uncalibrated model category scores, not the probability of cancer "
    "and not a diagnosis. The model does not reliably distinguish fractures, infection, benign lesions, "
    "or other out-of-distribution findings. A qualified professional must interpret the radiograph."
)


class CategoryScore(BaseModel):
    category: Literal["cancer", "normal"]
    display_name: Literal["Cancer", "Normal"]
    score: float = Field(ge=0, le=1)
    percentage: float = Field(ge=0, le=100)


class ImageInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: Literal["JPEG", "PNG"]


class ModelInfo(BaseModel):
    name: Literal["osteoinsight-resnet18"] = "osteoinsight-resnet18"
    task: Literal["dataset_category_classification"] = "dataset_category_classification"
    training_dataset: Literal["Roboflow Bone Cancer Detection v1"] = "Roboflow Bone Cancer Detection v1"


class PredictResponse(BaseModel):
    schema_version: Literal["osteoinsight.classification.v1"] = API_SCHEMA_VERSION
    model: ModelInfo
    model_status: Literal["ready"] = "ready"
    model_version: str
    image: ImageInfo
    inference_ms: float = Field(ge=0)
    predicted_category: Literal["cancer", "normal"]
    predicted_display_name: Literal["Cancer", "Normal"]
    scores: list[CategoryScore]
    interpretation: str
    uncertainty_status: Literal["specialist_review_required"] = "specialist_review_required"
    warnings: list[str]
    disclaimer: str = DISCLAIMER


class ClassifierService:
    def __init__(self, checkpoint: Path, requested_device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = "cpu"
        self.model = None
        self.meta: dict = {}
        self.idx_to_class: dict[int, str] = {}
        self.transform = None
        self.model_version = "unavailable"
        self.load_error = ""
        if not checkpoint.is_file():
            self.load_error = f"Classifier checkpoint not found: {checkpoint}"
            return
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.model_version = f"resnet18-{digest.hexdigest()[:12]}"
        try:
            self.device = select_device(requested_device)
            self.model, self.meta = load_checkpoint(checkpoint, self.device)
            class_to_idx = self.meta.get("class_to_idx", {})
            if class_to_idx != {"cancer": 0, "normal": 1}:
                raise ValueError(f"Unexpected class mapping: {class_to_idx!r}")
            self.idx_to_class = {index: name for name, index in class_to_idx.items()}
            _, self.transform = make_transforms(int(self.meta.get("image_size", 160)))
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None and self.transform is not None

    def predict(self, image: Image.Image) -> list[CategoryScore]:
        if not self.ready:
            raise RuntimeError(self.load_error or "Classifier is not ready")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].detach().cpu().tolist()
        return [
            CategoryScore(
                category=self.idx_to_class[index],
                display_name=self.idx_to_class[index].capitalize(),
                score=float(value),
                percentage=float(value * 100),
            )
            for index, value in enumerate(probabilities)
        ]


checkpoint = Path(os.environ.get("OSTEOINSIGHT_CLASSIFIER_PATH", DEFAULT_CHECKPOINT)).expanduser().resolve()
classifier = ClassifierService(checkpoint, os.environ.get("OSTEOINSIGHT_DEVICE", "auto"))
app = FastAPI(
    title="OsteoInsight Dataset-Category Classification API",
    version="2.0.0",
    description="Research-only ResNet18 inference for the source dataset's cancer/normal categories.",
)
origins = [value.strip() for value in os.environ.get(
    "OSTEOINSIGHT_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
).split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ready" if classifier.ready else "model_not_ready",
        "task": "dataset_category_classification",
        "model_status": "ready" if classifier.ready else "model_not_ready",
        "model_version": classifier.model_version,
        "checkpoint": str(classifier.checkpoint),
        "device": classifier.device,
        "detail": classifier.load_error or ("loaded" if classifier.ready else "classifier unavailable"),
        "class_names": ["cancer", "normal"],
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(..., description="One JPEG or PNG bone radiograph")):
    if not classifier.ready:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_not_ready", "message": "The ResNet18 classifier has not loaded."},
        )
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail={"code": "empty_file", "message": "Uploaded file is empty."})
    if len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": "Maximum upload size is 25 MB."})
    try:
        with Image.open(io.BytesIO(payload)) as decoded:
            source_format = (decoded.format or "").upper()
            if source_format not in {"JPEG", "PNG"}:
                raise ValueError("unsupported image format")
            image = ImageOps.exif_transpose(decoded).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=415,
            detail={"code": "invalid_image", "message": "Upload must be a readable JPEG or PNG image."},
        )
    started = time.perf_counter()
    try:
        scores = classifier.predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "inference_failed", "message": f"Model inference failed: {type(exc).__name__}."},
        ) from exc
    inference_ms = (time.perf_counter() - started) * 1000
    top = max(scores, key=lambda item: item.score)
    warnings = [
        "Model category score — not the probability of cancer.",
        "The model was trained on dataset-provided labels without independently verified clinical diagnoses.",
        "Fractures and other non-cancer abnormalities were not represented by a validated challenge set and may receive incorrect high scores.",
        "No validated out-of-distribution rejection threshold is available; specialist review is required for every result.",
    ]
    return PredictResponse(
        model=ModelInfo(),
        model_version=classifier.model_version,
        image=ImageInfo(width=image.width, height=image.height, format=source_format),
        inference_ms=inference_ms,
        predicted_category=top.category,
        predicted_display_name=top.display_name,
        scores=scores,
        interpretation=(
            f"This image is most similar to the source dataset's {top.display_name} category. "
            "That comparison is not a confirmed medical finding."
        ),
        warnings=warnings,
    )
