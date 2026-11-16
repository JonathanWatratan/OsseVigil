# Cleaned exploratory dataset

- Original rows: 8,811
- Retained representatives: 5,823
- Quarantined images: 296
- Redundant confirmed copies excluded: 2,692
- Exact duplicate groups: 1,059
- Confirmed non-exact near-duplicate pairs: 2,245
- Uncertain perceptual pairs: 198
- Unresolved confirmed-component label conflicts: 0
- Heuristic source-ID label-conflict groups quarantined: 3

The split is reproducible (seed 42), grouped by heuristic source ID, and has no known exact, confirmed-duplicate-component, or heuristic-source group crossing train/validation/test.
It is exploratory only: patient independence is unknown, labels lack documented clinical verification, and anatomy/source confounding remains. Deduplication does not solve those limitations.
