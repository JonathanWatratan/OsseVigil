# Archived OsteoInsight frontend contract — lesion localization v1

> This contract is preserved for the BTXRD localization research branch and is not used by the current MVP. The canonical product contract is [`CLASSIFICATION_API_CONTRACT.md`](CLASSIFICATION_API_CONTRACT.md).

The primary frontend must call the BTXRD detector API. It must not call or display the legacy Roboflow classifier as a clinical cancer probability.

## Endpoint

`POST /api/predict?threshold=0.25`

Content type: `multipart/form-data`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `file` | JPEG or PNG | yes | One bone radiograph |
| `threshold` | query float, 0.05–0.95 | no | Declared uncalibrated detector-score threshold; default 0.25 |

## Successful response

```json
{
  "schema_version": "osteoinsight.predict.v1",
  "model": {
    "name": "btxrd-yolov8n-lesion-localizer",
    "task": "bone_lesion_localization",
    "training_dataset": "BTXRD"
  },
  "model_status": "ready",
  "model_version": "btxrd-yolov8n-<checkpoint-hash>",
  "image": {"width": 2048, "height": 1536},
  "image_width": 2048,
  "image_height": 1536,
  "operating_threshold": 0.25,
  "inference_ms": 48.2,
  "detections": [
    {
      "label": "potential_bone_lesion",
      "class_id": 0,
      "score": 0.81,
      "bbox": [0.23, 0.18, 0.51, 0.62],
      "bbox_named": {"x_min": 0.23, "y_min": 0.18, "x_max": 0.51, "y_max": 0.62},
      "bbox_pixels": {"x_min": 471, "y_min": 276, "x_max": 1044, "y_max": 952}
    }
  ],
  "warnings": ["Detection scores are uncalibrated model outputs, not probabilities of cancer."],
  "flag": "potential_lesion_requires_professional_review",
  "disclaimer": "Research prototype only..."
}
```

`bbox` is the canonical normalized array in `[x_min, y_min, x_max, y_max]` order, with every value in `[0,1]` relative to the uploaded image. `bbox_named` is a readability adapter carrying the same coordinates. The frontend draws an ellipse and connector derived from this real box; it does not invent a heatmap or exact tumor boundary. `bbox_pixels` is included for debugging and non-responsive canvases.

When no box crosses the declared threshold, `detections` is empty and `flag` is `no_detection_above_threshold_not_a_clearance`. The UI must not translate that state to “normal,” “cancer-free,” or “no tumor.”

## Model unavailable

Until the real BTXRD detector checkpoint is trained and loaded, the endpoint returns HTTP `503`:

```json
{
  "detail": {
    "code": "model_not_ready",
    "message": "The BTXRD lesion detector has not been trained or loaded.",
    "legacy_classifier_fallback": false
  }
}
```

The frontend should show “Localization model is not ready” and must not fall back to the legacy classifier.

## Required UI language

- Positive flag: “Potential lesion — requires professional review.”
- Empty result: “No model detection above the selected threshold. This does not rule out a lesion.”
- Score label: “Model detection score,” not “cancer probability” or “clinical certainty.”
- Never infer malignancy, stage, surgical urgency, chemotherapy eligibility, treatment, or prognosis from this response.

## Health check

`GET /api/health`

Ready services return `status: ready`. The field `legacy_classifier_enabled` is always `false`.
