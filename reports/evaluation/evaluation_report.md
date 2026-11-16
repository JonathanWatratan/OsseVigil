# OsteoInsight exploratory remediation and evaluation

## Remediation outcome

- Original images: 8,811; original files modified: no
- Exact duplicate groups: 1,059 overall, including 345 crossing the published splits
- Confirmed non-exact perceptual pairs: 2,245 overall (threshold: pHash distance ≤4 and 128×128 correlation ≥0.995)
- Confirmed duplicate components: 2,364
- Lower-confidence perceptual pairs quarantined for review: 198, involving 287 images
- Heuristic source-ID label conflicts: three groups / nine images, all quarantined
- Label conflicts inside confirmed duplicate components: zero
- Redundant confirmed copies excluded: 2,692
- Retained representatives: 5,823

Visual samples are stored in `data/cleaned/confirmed_pair_review_samples.jpg` and `data/cleaned/uncertain_pair_review_samples.jpg`. The confirmed sample shows repeated or re-encoded radiographs. Lower-confidence candidates were quarantined because automated similarity cannot prove image, study, or patient identity.

## New source-grouped split

| Split | Cancer | Normal | Total |
|---|---:|---:|---:|
| Train | 2,223 | 2,406 | 4,629 |
| Validation | 290 | 311 | 601 |
| Test | 286 | 307 | 593 |

Seed: 42. Known cross-split exact hashes, confirmed duplicate components, and heuristic source groups after remediation: zero. Patient/study independence remains unknown because no reliable identifiers are available.

## Training

The baseline used an ImageNet-pretrained ResNet18 with a frozen feature backbone and trained binary head at 160×160. AdamW used learning rate 0.001, weight decay 0.0001, weighted cross-entropy, batch size 64, and seed 42. Validation macro-F1 selected the checkpoint; the best value was 0.903 at epoch 20. The test set was evaluated once after selection.

Apple MPS was attempted first but crashed in the MPS linear backend. The completed run used CPU, and the checkpoint records the device and configuration.

## Test results

| Metric | ResNet18 | Majority baseline |
|---|---:|---:|
| Macro-F1 | 0.874 | 0.341 |
| Balanced accuracy | 0.874 | 0.500 |
| Accuracy | 0.874 | 0.518 |
| Cancer-category sensitivity | 0.895 | — |
| Specificity | 0.853 | — |
| ROC-AUC | 0.948 | — |

Confusion matrix (true rows, predicted columns; order `cancer`, `normal`):

| | Predicted cancer | Predicted normal |
|---|---:|---:|
| Dataset cancer | 256 | 30 |
| Dataset normal | 45 | 262 |

Cancer precision was 0.850 and F1 was 0.872. Normal precision was 0.897 and F1 was 0.875. There were 30 false negatives and 45 false positives.

## Shortcut and subgroup findings

Source family remains heavily class-coupled: `bone-cancer` and pathology-named families are cancer-only, while `IMG-numbered`, `image-no`, and `Picture-numbered` are normal-only. Only the `numeric` and `other` families contain both classes in the test set. Their exploratory balanced accuracies were 0.750 (n=19) and 0.773 (n=107), respectively.

Filename-derived anatomy groups are incomplete and unverified. Several are single-label or very small, making subgroup sensitivity/specificity uninterpretable. Full values are in `heuristic_subgroup_metrics.csv`.

## Interpretation limits

This evaluation is exploratory and source-grouped, not patient-independent. The annotations lack documented pathology, radiology-report, or independent clinical verification. Automated grouping can make both false-positive and false-negative identity decisions. Strong anatomy, acquisition, and source correlations remain after deduplication. Therefore these metrics must not be described as diagnostic accuracy, cancer prognosis, or evidence of clinical generalization.
