# Model artifacts

Large model files are deliberately excluded from Git so normal source pushes remain small and reliable.

## Required local artifact

- Path: `models/osteoinsight_best.pth`
- Architecture: ResNet18
- Classes: `cancer = 0`, `normal = 1`
- Input size: 160 × 160
- Local size: approximately 43 MiB

The current checkpoint is below GitHub's 100 MiB per-file hard limit, but it is still a generated training artifact and is not committed by default. Keep it in the path above or set:

```bash
export OSTEOINSIGHT_CLASSIFIER_PATH=/absolute/path/to/osteoinsight_best.pth
```

For team distribution, use a versioned release asset, institutional artifact store, or Git LFS. Do not place the original dataset in Git.

## Integrity

At startup the API hashes the checkpoint and exposes the first 12 SHA-256 characters in `model_version`. This lets the frontend and logs identify the exact local artifact used for inference.
