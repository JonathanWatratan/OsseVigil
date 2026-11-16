#!/usr/bin/env python3
"""Train the single-class BTXRD lesion-localization baseline with Ultralytics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Generated BTXRD data.yaml")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="", help="Ultralytics device: '', cpu, mps, 0, 0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/btxrd_detection"))
    parser.add_argument("--output", type=Path, default=Path("models/btxrd_yolov8n_best.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install detector dependencies with: python -m pip install ultralytics") from exc
    if not args.data.is_file():
        raise SystemExit(f"YOLO data file not found: {args.data}")
    model = YOLO(args.weights)
    result = model.train(
        data=str(args.data.resolve()), epochs=args.epochs, patience=args.patience,
        imgsz=args.image_size, batch=args.batch_size, workers=args.workers,
        device=args.device or None, seed=args.seed, deterministic=True,
        project=str(args.runs_dir.resolve()), name="yolov8n_lesion", exist_ok=True,
        pretrained=True, single_cls=True, plots=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training finished without best checkpoint: {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"best_source": str(best), "best_checkpoint": str(args.output.resolve())})
    (args.output.parent / "btxrd_training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Best checkpoint copied to {args.output}")


if __name__ == "__main__":
    main()
