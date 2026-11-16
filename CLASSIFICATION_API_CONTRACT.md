# OsteoInsight frontend contract — classification v1

The current frontend calls the ResNet18 dataset-category service. It must describe the two values as model category scores, never clinical probabilities.

## Health

`GET /api/health`

Ready response fields include:

```json
{
  "status": "ready",
  "task": "dataset_category_classification",
  "model_status": "ready",
  "model_version": "resnet18-<checkpoint-hash>",
  "device": "cpu",
  "class_names": ["cancer", "normal"]
}
```

## Prediction

`POST /api/predict` with multipart field `file`. Accepted decoded formats are JPEG and PNG; maximum size is 25 MiB.

```json
{
  "schema_version": "osteoinsight.classification.v1",
  "model": {
    "name": "osteoinsight-resnet18",
    "task": "dataset_category_classification",
    "training_dataset": "Roboflow Bone Cancer Detection v1"
  },
  "model_status": "ready",
  "model_version": "resnet18-<checkpoint-hash>",
  "image": {"width": 640, "height": 640, "format": "PNG"},
  "inference_ms": 25.0,
  "predicted_category": "normal",
  "predicted_display_name": "Normal",
  "scores": [
    {"category": "cancer", "display_name": "Cancer", "score": 0.2, "percentage": 20.0},
    {"category": "normal", "display_name": "Normal", "score": 0.8, "percentage": 80.0}
  ],
  "interpretation": "This image is most similar to the source dataset's Normal category. That comparison is not a confirmed medical finding.",
  "uncertainty_status": "specialist_review_required",
  "warnings": ["Model category score — not the probability of cancer."],
  "disclaimer": "Research prototype only..."
}
```

The frontend must always show both score bars, the non-diagnostic label, uncertainty warnings, and the medical disclaimer. It must not convert `normal` into “cancer-free” or convert `cancer` into a confirmed diagnosis.

## Errors

- `400 empty_file`: no upload bytes.
- `413 file_too_large`: over 25 MiB.
- `415 invalid_image`: unreadable or non-JPEG/PNG content.
- `500 inference_failed`: checkpoint execution failed.
- `503 model_not_ready`: checkpoint missing or incompatible.

No mock scores or silent fallback predictions are permitted.
