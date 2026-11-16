# OsteoInsight AI model card

## Status

An exploratory ImageNet-pretrained ResNet18 baseline has been trained on a non-destructively cleaned manifest. The backbone was frozen and a new binary classification head was selected by validation macro-F1. It is a dataset-category research baseline, not a clinically validated cancer detector.

## Intended use

OsteoInsight AI is a research prototype for auditing bone-image datasets, exploring visual similarity when labels are unavailable, and—only with source-verified labels—training a dataset-category classifier or an annotation-preserving detector.

It is not intended for diagnosis, treatment decisions, prognosis, survival prediction, chemotherapy-response prediction, cancer-cure prediction, or tumor-necrosis assessment.

## Candidate workflows

- Verified classification labels: ImageNet-pretrained ResNet18 or EfficientNet-B0 transfer learning.
- Valid object-detection annotations: annotation-preserving detector selected only after coordinate and class validation.
- Missing or unverified labels: pretrained ResNet18 feature extraction and K-Means visual exploration. Clusters are not diagnoses.
- Mixed sources: analyze separately until label semantics, source compatibility, and licensing permit responsible combination.

## Data and labels

The audited source is [bone cancer detection v1 on Roboflow Universe](https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru/dataset/1): 8,811 readable images, published as 7,057 train / 882 validation / 872 test under CC BY 4.0. Every CSV row is valid one-hot `cancer`/`normal`, but 345 exact duplicate groups cross splits, 177 test images have an exact copy outside test, and patient/study identity is unavailable. Roboflow reports auto-orientation plus stretch-resizing to 640×640. The annotations are not independently clinically verified.

## Evaluation policy

Supervised metrics are reported only on a held-out, leakage-checked split. The evaluation includes a majority-class baseline, accuracy, macro-F1, per-class precision/recall, one-vs-rest specificity, a confusion matrix, and ROC-AUC only when mathematically appropriate. Probability outputs are explicitly labeled uncalibrated unless calibration is separately fitted and evaluated.

## Remediation and split

The automated workflow found 1,059 exact duplicate groups overall and 2,245 confirmed non-exact perceptual pairs. It retained one deterministic representative per confirmed component, quarantined 287 images involved in lower-confidence perceptual matches, and quarantined nine images in three heuristic source-ID groups with conflicting labels. No confirmed duplicate component contained a label conflict.

The retained 5,823 images were assigned by heuristic source group with seed 42: train 4,629 (2,223 cancer / 2,406 normal), validation 601 (290 / 311), and test 593 (286 / 307). No known exact hash, confirmed duplicate component, or heuristic source group crosses these splits. This does not establish patient independence.

## Baseline configuration and results

- ResNet18 with ImageNet pretrained weights; frozen backbone, trained linear head
- Input 160×160; AdamW, learning rate 0.001, weighted cross-entropy, seed 42
- 20 candidate epochs; checkpoint selected only by validation macro-F1 (0.903)
- CPU execution after the available MPS backend crashed in an Apple MPS linear kernel
- Test macro-F1 0.874; balanced accuracy 0.874
- Cancer-category sensitivity 0.895 and specificity 0.853
- Confusion matrix, true rows `cancer`, `normal`: `[[256, 30], [45, 262]]`
- Majority baseline: macro-F1 0.341; balanced accuracy 0.500
- ROC-AUC 0.948; probabilities are uncalibrated

These results are specific to the cleaned exploratory split and dataset-provided labels. Strong source-family and anatomical-region correlations remain and can inflate apparent performance.

## Known limitations

- Patient-level generalization cannot be claimed from image-level splitting.
- Exact and near-duplicates can inflate performance if allowed across partitions.
- A dataset category is not necessarily a medically meaningful diagnosis.
- Performance may fail under changes in anatomy, equipment, acquisition protocol, demographics, institution, or preprocessing.
- Visual clusters may reflect borders, markers, compression, image size, or source artifacts rather than anatomy.
- Image licensing and permission to redistribute must be established independently.

## Safety notice

Research prototype only. This system is not validated for clinical diagnosis, treatment decisions, or patient prognosis. Results require professional medical interpretation.
