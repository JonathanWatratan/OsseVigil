# OsteoInsight dataset report

Status: **audit complete; supervised-training gate failed**.

The local Roboflow export at `/Users/jonathan/Downloads/archive` was audited read-only. All 8,811 images are readable and all CSV annotations are valid mutually exclusive one-hot rows with columns `filename,cancer,normal`.

| Split | Cancer | Normal | Total |
|---|---:|---:|---:|
| Train | 3,081 | 3,976 | 7,057 |
| Validation | 398 | 484 | 882 |
| Test | 384 | 488 | 872 |

There are no missing images, orphan images, malformed indicators, zero-active rows, multi-active rows, or duplicate CSV filename rows.

## Blocking leakage finding

- 1,059 exact duplicate groups exist overall, containing 2,118 image rows.
- 345 exact SHA-256 duplicate groups cross the published splits, affecting 690 image rows.
- 177 test images have an exact copy outside test.
- 168 validation images have an exact copy in training.
- No reliable patient or study identifier is present, so patient-level independence cannot be established.

Approximate near-duplicate hashes and filename-derived source IDs found additional review candidates. These heuristic candidates can include false positives; exact SHA-256 duplicates are definitive.

A stricter DCT perceptual-hash plus pixel-correlation pass found 785 high-confidence non-exact cross-split pairs. Visual review confirms repeated radiographs that were re-encoded or renamed.

## Shortcut risks

- `bone-cancer` filename family: 2,543 cancer / 0 normal.
- pathology-named family: 290 cancer / 0 normal.
- `IMG-numbered` family: 0 cancer / 2,446 normal.
- `image-no` family: 0 cancer / 1,379 normal.
- 4,017 filenames contain `cancer` or `normal`, with 100% agreement to the CSV label.
- Several filename anatomy hints are class-skewed, and mean pixel/texture statistics differ by category.

Filenames are excluded from model input, but these patterns show that anatomical region and image-source artifacts may remain learnable from pixels.

No ResNet18 was trained and no performance metric was generated because the project requires training to stop when serious unresolved audit problems exist. Detailed artifacts are in `reports/roboflow_audit/`.

A derived, non-destructive leakage-reduced manifest excludes 1,077 lower-priority train/validation copies while retaining all 872 test rows. It removes known exact cross-split groups, but training remains blocked by unknown patient independence, absent clinical annotation provenance, and strong source/anatomy confounding. See `reports/leakage_analysis/` and `reports/bias_and_provenance_report.md`.
