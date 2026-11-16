# Fracture and unsupported-image challenge status

No properly sourced, independently labeled fracture/non-cancer-abnormality challenge set is currently present in this project. Therefore:

- no fracture false-positive rate is reported;
- no claim is made that the classifier can distinguish fracture from tumor;
- no confidence threshold is presented as a validated safety mechanism;
- no out-of-distribution rejection guarantee is implemented;
- all results are marked for specialist review.

The prior observation that a fracture-like forearm radiograph received a high Cancer-category score is treated as a known failure example, not as a medically verified benchmark case. Its appearance alone is not used to assign a fracture label.

A defensible future challenge evaluation requires images with explicit provenance, compatible licenses, and labels established independently of filenames. It must be kept separate from training and threshold selection.
