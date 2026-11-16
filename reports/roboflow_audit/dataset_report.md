# Roboflow classification dataset audit

Generated: 2026-11-15T22:33:42.378260+00:00
Read-only dataset root: `/Users/jonathan/Downloads/archive`

## Annotation integrity

- CSV schemas: `{"test": {"class_columns": ["cancer", "normal"], "filename_column": "filename", "path": "/Users/jonathan/Downloads/archive/test/_classes.csv", "rows": 872}, "train": {"class_columns": ["cancer", "normal"], "filename_column": "filename", "path": "/Users/jonathan/Downloads/archive/train/_classes.csv", "rows": 7057}, "valid": {"class_columns": ["cancer", "normal"], "filename_column": "filename", "path": "/Users/jonathan/Downloads/archive/valid/_classes.csv", "rows": 882}}`
- Image rows parsed: 8,811
- Readable images with exactly one active binary class: 8,811
- Split counts: {'test': 872, 'train': 7057, 'valid': 882}
- Match published 7,057/882/872 counts: {'train': True, 'valid': True, 'test': True}
- Structural annotation/image issues: 0
- Total audit flags including leakage candidates: 1,925
- Images present but absent from CSV: 0
- CSV class names are treated as source annotations, not independently verified clinical diagnoses.

## Class distribution

| Original split | cancer | normal |
|---|---:|---:|
| test | 384 | 488 |
| train | 3081 | 3976 |
| valid | 398 | 484 |

## Leakage and inconsistency checks

- Cross-split groups: {'cross_split_near_duplicate': 865, 'cross_split_source_id_collision': 688, 'cross_split_exact_duplicate': 345}
- Conflicting-label groups: {'conflicting_labels_near_duplicate': 24, 'conflicting_labels_source_id_collision': 3}
- Exact duplicate groups overall: 1,059 (2,118 images)
- Exact duplicate groups crossing published splits: 345 (690 image rows)
- Near-duplicate dHash groups are candidate-review signals and can include false positives; exact SHA-256 duplicates are definitive.
- `source_id_heuristic` removes Roboflow hash suffixes and disease words; it is not a confirmed patient identifier.
- Patient-level independence cannot be established unless original patient/study identifiers are supplied.

## Shortcut risks

- Filenames containing `normal` or `cancer`: 4,017
- Agreement of those filename tokens with CSV labels: 100.0%
- Filenames are excluded from model inputs, but source naming can reveal duplication or collection bias.
- Low-level intensity, border, resolution, source-stem, and visual-cluster statistics are exported for shortcut review.
- Filename anatomy hints (audit metadata only): ankle: cancer=0, normal=50; chest: cancer=0, normal=61; elbow: cancer=0, normal=47; femur: cancer=79, normal=96; fibula: cancer=90, normal=0; foot: cancer=87, normal=74; forearm: cancer=0, normal=53; hand: cancer=65, normal=118; humerus: cancer=45, normal=32; knee: cancer=0, normal=54; pelvis: cancer=60, normal=70; rib: cancer=4, normal=0; shoulder: cancer=0, normal=4; spine: cancer=28, normal=7; tibia: cancer=117, normal=12; ulna: cancer=25, normal=0
- Filename source-family hints: IMG-numbered: cancer=0, normal=2446; Picture-numbered: cancer=0, normal=39; bone-cancer: cancer=2543, normal=0; image-no: cancer=0, normal=1379; numeric: cancer=318, normal=65; other: cancer=712, normal=1019; pathology-named: cancer=290, normal=0
- Anatomical region is not asserted automatically. Representative cluster/error montages require human anatomical review.
- Roboflow reports auto-orientation and stretch-resizing to 640×640; original pre-export aspect ratio may not be recoverable.

## Scientific interpretation

- Supervised training gate: **FAIL**
- Gate reason: cross-split exact duplicates contaminate validation/test independence.
A held-out result on these source-provided splits measures reproduction of the dataset categories only. It must not be described as clinical diagnostic accuracy or patient-level generalization. Cross-split duplicates, unknown patient identity, source artifacts, anatomical-region imbalance, and unverified diagnostic provenance can inflate performance.
