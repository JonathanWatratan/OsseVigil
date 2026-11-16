# BTXRD localization MVP status

## Implemented

- Read-only release discovery for `images/`, `Annotations/`, and `dataset.csv`/`.xlsx`
- LabelMe polygon/rectangle to bounding-box parsing
- Image/annotation/metadata matching and coordinate validation
- Exact SHA-256 duplicate grouping
- Conservative proxy-lesion grouping for consecutive images with identical non-view metadata
- Seeded train/validation/test assignment stratified by dataset label, center, and anatomy
- Single-class YOLO export with normal images as negative examples
- YOLOv8n training wrapper and reproducible configuration output
- Final-test evaluation for mAP@0.5, mAP@0.5:0.95, lesion recall, image sensitivity/specificity, benign/malignant subgroup recall, and false-positive/false-negative examples
- React localization workstation with original/annotated views, zoom/pan, overlay toggle, ellipse/connector labels, original-resolution PNG export, and visible non-diagnostic disclaimers
- FastAPI `GET /api/health` plus `POST /api/predict`, with canonical normalized `[x_min, y_min, x_max, y_max]` arrays and pixel-coordinate adapters
- Explicit HTTP 503 behavior when the BTXRD checkpoint is unavailable; no legacy-classifier fallback
- Frontend response schema and required safety language in `FRONTEND_API_CONTRACT.md`
- Google Colab GPU workflow in `btxrd_localization_colab.ipynb`
- Analyze-button error path sends the actual multipart request and displays backend errors instead of remaining silently inert

## Current blocker

No stable local BTXRD release directory was found. A temporary browser-download item appeared under `/Users/jonathan/Downloads` but disappeared before ZIP integrity could be checked. Training and metrics are intentionally not produced until the complete release is available and audited.

Required input is the absolute directory containing all three of:

```text
images/
Annotations/
dataset.csv  # or dataset.xlsx
```

The original release is described in Yao et al., *Scientific Data* 12, 88 (2025), DOI `10.1038/s41597-024-04311-y`, dataset DOI `10.6084/m9.figshare.27865398`.
