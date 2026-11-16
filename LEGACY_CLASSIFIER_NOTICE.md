# Roboflow classifier safety notice

The ResNet18 binary classifier is the active simplified hackathon MVP. Its checkpoint, evaluation files, and Gradio code also document the experiment and its failure modes.

Its output is deliberately limited. In particular:

- `cancer` and `normal` are dataset-provided categories, not independently verified clinical diagnoses.
- A high `cancer` category score on a fracture or other out-of-distribution image is not evidence of cancer.
- The model is exposed by the product FastAPI service only as a dataset-category research classifier.
- The API returns real checkpoint scores and never hardcodes or simulates results.
- Product output must call these values model category scores, not probabilities of cancer.

The BTXRD detector files remain archived separately. They are not required by this classification MVP.
