# OsteoInsight bias and annotation-provenance review

## Annotation evidence found

The local `README.dataset.txt` identifies a user-provided Roboflow Universe project under CC BY 4.0. `README.roboflow.txt` states that 8,811 “Xray-images” were exported in multi-class classification format after auto-orientation and stretch-resizing to 640×640, with no export-time augmentation.

The public Roboflow project lists classes `normal` and `cancer`, but explicitly says that a project description has not been published. Neither the local export nor public page documents:

- patient or study identifiers;
- source institution or cohort;
- anatomical inclusion criteria;
- annotator qualifications or agreement;
- pathology, radiology-report, or clinical confirmation;
- label definitions;
- exclusion criteria;
- whether repeated images belong to the same patient or study.

Therefore, the CSV supports only the statement “dataset-provided category.” It does not independently verify a cancer diagnosis.

## Definitive and perceptual leakage

- Exact cross-split SHA-256 groups: 345, involving 690 rows.
- High-confidence non-exact cross-split perceptual pairs: 785.
- Raw source filenames crossing splits: 688.
- Patient/study independence: not established.

Visual review of high-confidence perceptual pairs confirms repeated radiographs that were re-encoded or renamed across train, validation, and test.

A derived manifest retained every test image, then removed lower-priority train/validation copies using the policy `test > validation > train`. It excluded 1,077 rows and eliminated known exact cross-split groups without touching source files. Remaining counts are:

| Split | Cancer | Normal | Total |
|---|---:|---:|---:|
| Train | 2,721 | 3,320 | 6,041 |
| Validation | 373 | 448 | 821 |
| Test | 384 | 488 | 872 |

This correction does not establish patient independence and does not remove all possible source-related overlap.

## Source and anatomical shortcut risk

Filename metadata is excluded from model input, but it reveals strong source construction differences that may also be present in the pixels:

| Filename/source family | Cancer | Normal |
|---|---:|---:|
| `bone-cancer` | 2,543 | 0 |
| pathology-named | 290 | 0 |
| `IMG-numbered` | 0 | 2,446 |
| `image-no` | 0 | 1,379 |

Anatomy hints are also label-skewed. Chest, knee, ankle, elbow, and forearm hints appear only in normal-labeled filenames in this export; fibula, rib, ulna, and pathology-named sources are cancer-only or strongly cancer-skewed. Because the images cover multiple anatomical regions, a classifier can learn body-region or acquisition-source differences instead of pathology.

Low-level image statistics differ by label as well: mean normalized intensity is approximately 0.319 for `cancer` and 0.237 for `normal`; texture standard deviation is approximately 0.237 versus 0.155. These differences do not prove a shortcut, but they make source/acquisition discrimination plausible.

## Training decision

The supervised-training gate for the original published splits **failed**. A later exploratory ResNet18 was fitted only after non-destructive deduplication, quarantine, and source-grouped re-splitting; its evaluation remains non-clinical because patient independence, annotation provenance, and source/anatomy confounding are unresolved.
