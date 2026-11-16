# OsteoInsight AI

OsteoInsight AI is a research-only bone X-ray classification prototype with a React/TypeScript workstation and FastAPI inference service. The active MVP uses the existing ResNet18 checkpoint to return the source dataset's two categories, **Cancer** and **Normal**. These are uncalibrated dataset-category scores, not clinical probabilities or diagnoses. BTXRD localization research files remain preserved but are not required by the current MVP.

## Run the full-stack workstation

The web application is in `frontend/` and preserves the project’s navy, blue, and cyan visual system, 1.5-second startup splash, responsive navigation, and License & Credits dialog. The workspace displays the original image beside Cancer/Normal score bars and explicit uncertainty messaging.

Terminal 1 — classification API:

```bash
python3.12 -m venv .venv          # first run only
source .venv/bin/activate
python -m pip install -r requirements.txt  # first run only
uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2 — React development server:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/`. Vite proxies `/api` to FastAPI on port 8000. A production frontend bundle can be generated with `npm run build`.

Run the reproducible checks:

```bash
python -m unittest discover -s tests
cd frontend && npm run build
```

The API loads `models/osteoinsight_best.pth` by default. Override it with `OSTEOINSIGHT_CLASSIFIER_PATH=/absolute/path/checkpoint.pth`. The checkpoint is intentionally ignored by Git; see `MODEL_ARTIFACTS.md`.

## Current MVP and scientific limits

`POST /api/predict` performs actual ResNet18 inference and returns two scores that sum to one. The UI describes the higher value as the closest dataset category—not a confirmed finding. The model has no validated fracture/other-abnormality rejection mechanism, so every result requires specialist review and unfamiliar images can receive high incorrect scores.

Start the API with:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

`GET /api/health` reports model readiness. Valid JPG/PNG uploads produce real checkpoint scores; empty, unreadable, and unsupported uploads return explicit errors.

## Classification experiment and evaluation

The read-only audit and non-destructive remediation are complete for `/Users/jonathan/Downloads/archive`. The cleaned, source-grouped exploratory manifest retains 5,823 representatives and has no known exact, confirmed-perceptual-component, or heuristic-source group crossing its new splits. A frozen-backbone ImageNet-pretrained ResNet18 baseline was trained and evaluated once on the newly designated 593-image test set.

Test macro-F1 is 0.874 and balanced accuracy is 0.874. These numbers measure reproduction of dataset-provided categories only. Patient independence, clinical label provenance, and source/anatomy independence are not established; this is not clinical diagnostic accuracy. A sourced fracture challenge set has not yet been evaluated, so no fracture false-positive rate is claimed.

The source dataset is never renamed, moved, deleted, or written to. Generated artifacts must be stored outside the dataset tree.

## Preserved BTXRD localization research

The hackathon localization pipeline is separate from the earlier Roboflow category baseline and never merges the two datasets. It expects the original BTXRD release root:

```text
BTXRD/
├── images/          # IMG000001.jpeg ...
├── Annotations/     # LabelMe JSON for tumor images
└── dataset.csv      # or dataset.xlsx
```

Prepare a read-only, leakage-aware YOLO working set:

```bash
python btxrd_prepare.py "/absolute/path/to/BTXRD" \
  --output-dir data/btxrd_detector \
  --report-dir reports/btxrd_audit \
  --seed 42
```

The converter collapses every verified lesion box to one detector class, `lesion`, while retaining benign/malignant metadata for subgroup evaluation. Normal images receive empty YOLO label files and act as negative examples. Original JPEG, JSON, and metadata files are never changed.

After the audit passes:

```bash
python train_detector.py data/btxrd_detector/data.yaml \
  --weights yolov8n.pt --epochs 100 --patience 20 \
  --output models/btxrd_yolov8n_best.pt

python evaluate_detector.py data/btxrd_detector/manifest.csv \
  --data data/btxrd_detector/data.yaml \
  --checkpoint models/btxrd_yolov8n_best.pt \
  --confidence 0.25 --iou 0.5

python app_localization.py \
  --checkpoint models/btxrd_yolov8n_best.pt \
  --confidence 0.25
```

For a Google Colab GPU run, open `btxrd_localization_colab.ipynb`, mount Drive, and set `PROJECT` plus `BTXRD_ROOT` in its path cell. The notebook runs the same audit, training, frozen-test evaluation, checkpoint hashing, and metric reporting workflow.

The declared application threshold must match the evaluated operating threshold. BTXRD has no patient identifier and the paper notes that different views may show the same lesion. The preparation script uses consecutive IDs plus identical non-view metadata as an explicitly heuristic proxy group; this reduces obvious view leakage but cannot prove patient independence.

## Environment

Use Python 3.11 or 3.12. The inspected Mac is Apple Silicon and the pipeline selects CUDA first, then Apple MPS, then CPU. The currently active Python 3.14 installation has none of the required packages, so create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.12 is unavailable or local detector training is too slow, use `btxrd_localization_colab.ipynb` in Google Colab. The older `osteoinsight_training.ipynb` documents the archived classification experiment and is not the product training workflow. Do not create a public share link for private medical data.

## 1. Audit the authorized dataset

```bash
python audit_dataset.py "/absolute/path/to/authorized/dataset" --output-dir reports
```

This produces the inventory, image hashes, approximate near-duplicate groups, corruption findings, annotation evidence, folder-category counts, identifier hints, and an evidence-based scenario recommendation. Grayscale appearance is only a heuristic; it does not medically validate modality.

Review `reports/dataset_report.md` before proceeding. Do not add `--labels-trusted` merely because class-like folders exist. Use it only after verifying the original annotation source and meaning:

```bash
python audit_dataset.py "/absolute/path/to/authorized/dataset" --output-dir reports --labels-trusted
```

### Roboflow bone-cancer dataset v1

The identified source is [Roboflow Universe version 1](https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru/dataset/1), published under CC BY 4.0 with 8,811 images. Roboflow reports 7,057 train, 882 validation, and 872 test images, with auto-orientation and stretch-resizing to 640×640.

Download the **Multiclass Classification** export and leave its `train/`, `valid/`, and `test/` folders unchanged. Then run the dedicated CSV audit:

```bash
python audit_roboflow_classification.py "/absolute/path/to/exported-dataset" \
  --output-dir reports/roboflow_audit \
  --expected-classes normal cancer
```

This command verifies binary indicator values, confirms exactly one active class per row, checks all CSV/image references, preserves the original splits, hashes images, finds cross-split duplicates and source-stem collisions, and exports shortcut-risk summaries. Here, `label_verified` means only “successfully parsed from the source CSV”; it does **not** mean independently medically verified.

For the current export, the original-split training gate remains false. Use the approved non-destructive remediation workflow instead of training on the published splits:

```bash
python remediate_dataset.py reports/roboflow_audit/dataset_manifest.csv \
  --identity-manifest reports/leakage_analysis/image_identity_manifest.csv \
  --output-dir data/cleaned --seed 42

python train.py data/cleaned/cleaned_manifest.csv \
  --architecture resnet18 --output-dir models --epochs 20 --patience 5 \
  --batch-size 64 --learning-rate 0.001 --image-size 160 \
  --freeze-backbone --device cpu --seed 42

python evaluate.py data/cleaned/cleaned_manifest.csv \
  --checkpoint models/osteoinsight_best.pth \
  --output-dir reports/evaluation --positive-class cancer --device cpu

python app.py --checkpoint models/osteoinsight_best.pth --device cpu
```

CPU is specified because the tested PyTorch/MPS combination crashed in an Apple MPS linear kernel. `--device auto` may be retried after upgrading PyTorch/macOS.

Run visual clustering across all images to review whether categories track anatomical region or source style:

```bash
python explore_embeddings.py reports/roboflow_audit/dataset_manifest.csv \
  --output-dir reports/embeddings --clusters 12
```

The resulting cluster/category table and representative montages are review aids, not diagnostic evidence. Any test metrics describe reproduction of the Roboflow CSV categories on its published split only.

## 2A. No reliable labels: visual exploration

```bash
python explore_embeddings.py reports/dataset_manifest.csv --output-dir reports/embeddings --clusters 8
python app.py --exploration-dir reports/embeddings
```

The outputs include embeddings, K-Means assignments, a PCA overview, and representative cluster montages. These clusters are visual groups—not medical diagnoses.

## 2B. Verified classification labels

Generate a leakage-aware copied manifest. Existing source splits are preserved only if group leakage checks pass; otherwise grouping prioritizes patient IDs, study IDs, duplicate groups, and image hashes.

```bash
python prepare_splits.py reports/dataset_manifest.csv --output reports/dataset_manifest_split.csv
python train.py reports/dataset_manifest_split.csv --output-dir models
python evaluate.py reports/dataset_manifest_split.csv --checkpoint models/osteoinsight_best.pth
python app.py --checkpoint models/osteoinsight_best.pth
```

Training uses conservative augmentation: resizing and small brightness/contrast jitter only. It deliberately avoids automatic flips, rotations, crops, or aggressive geometry changes because laterality and anatomy can be meaningful.

## 2C. Detection annotations or mixed sources

Stop after the audit until the boxes, class semantics, image-coordinate mapping, source provenance, and source compatibility have been reviewed. A detector must not be selected from the presence of `.txt` or JSON files alone. The implementation will be specialized to the discovered annotation format after that review so bounding boxes and class information are preserved.

## Outputs

- `reports/dataset_report.md`: audit evidence, scenario, and scientific limitations
- `reports/dataset_manifest.csv`: one row per image with hashes and metadata
- `reports/data_quality_report.csv`: corrupt, low-resolution, duplicate, and review issues
- `data/cleaned/`: cleaned manifests, quarantine, duplicate decisions, and review montages
- `models/osteoinsight_best.pth`: best validation macro-F1 checkpoint, class map included
- `models/training_history.csv` and `.png`: training/validation history
- `reports/evaluation/`: held-out metrics, confusion matrix, and incorrect predictions
- `reports/embeddings/`: exploratory features, clusters, and sample montages

## Retraining checklist

1. Preserve a copy of source provenance and label definitions.
2. Audit again and review corrupt images, duplicates, and source boundaries.
3. Confirm labels as source-verified only when justified.
4. Ensure every patient/study/duplicate group occurs in one split only.
5. Train and select by validation macro-F1; do not tune on the held-out test set.
6. Evaluate once on the test set and compare with the majority baseline.
7. Update `model_card.md` with the actual cohort, configuration, metrics, uncertainty, licensing, and limitations.

## Medical and privacy notice

Research prototype only. This system is not validated for clinical diagnosis, treatment decisions, or patient prognosis. Results require professional medical interpretation.

Do not upload private images or metadata to external services. Bone X-rays alone cannot establish why a patient did not recover from cancer treatment, and “abnormal bone” must not be equated with bone cancer.
