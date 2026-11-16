#!/usr/bin/env python3
"""Evaluate BTXRD localization once on the designated test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/btxrd_yolov8n_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/btxrd_evaluation"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=float)
    lt = np.maximum(left[:, None, :2], right[None, :, :2])
    rb = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    left_area = (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1])
    right_area = (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1])
    return intersection / np.maximum(left_area[:, None] + right_area[None, :] - intersection, 1e-9)


def matched_gt_count(gt: np.ndarray, pred: np.ndarray, threshold: float) -> int:
    scores = box_iou(gt, pred)
    matched_gt, matched_pred = set(), set()
    candidates = [(scores[i, j], i, j) for i in range(len(gt)) for j in range(len(pred)) if scores[i, j] >= threshold]
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i); matched_pred.add(j)
    return len(matched_gt)


def save_examples(rows: pd.DataFrame, output: Path, title: str, maximum: int = 12):
    output.mkdir(parents=True, exist_ok=True)
    for index, row in rows.head(maximum).iterrows():
        image = cv2.imread(row["absolute_path"])
        if image is None:
            continue
        for box in json.loads(row["predicted_boxes_json"]):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 190, 255), 3)
            cv2.putText(image, f"model {box[4]:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 190, 255), 2)
        cv2.putText(image, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 240), 2)
        cv2.imwrite(str(output / f"{index:03d}_{Path(row['absolute_path']).name}"), image)


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install detector dependencies with: python -m pip install ultralytics") from exc
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    test = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    if test.empty:
        raise SystemExit("Manifest has no test images.")
    model = YOLO(str(args.checkpoint))
    validation = model.val(
        data=str(args.data.resolve()), split="test", imgsz=args.image_size,
        conf=0.001, iou=0.7, device=args.device or None, plots=True,
        project=str(output), name="ultralytics_test", exist_ok=True,
    )
    predictions = model.predict(
        source=test["absolute_path"].tolist(), conf=args.confidence, iou=0.7,
        imgsz=args.image_size, device=args.device or None, verbose=False,
    )
    rows, total_gt, matched_gt = [], 0, 0
    for item, result in zip(test.itertuples(index=False), predictions):
        gt_items = json.loads(item.boxes_json)
        gt = np.asarray([[box["x1"], box["y1"], box["x2"], box["y2"]] for box in gt_items], dtype=float).reshape(-1, 4)
        pred_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        kept = conf >= args.confidence
        pred_xyxy, conf = pred_xyxy[kept], conf[kept]
        total_gt += len(gt); matched = matched_gt_count(gt, pred_xyxy, args.iou); matched_gt += matched
        rows.append({
            "absolute_path": item.absolute_path, "image_id": item.image_id,
            "disease_group": item.disease_group, "anatomy_group": item.anatomy_group,
            "center": getattr(item, "center", ""), "ground_truth_boxes": len(gt),
            "predicted_boxes": len(pred_xyxy), "matched_ground_truth_boxes": matched,
            "predicted_boxes_json": json.dumps([list(map(float, box)) + [float(score)] for box, score in zip(pred_xyxy, conf)]),
            "image_positive": len(gt) > 0, "predicted_positive": len(pred_xyxy) > 0,
        })
    results = pd.DataFrame(rows)
    tp = int((results.image_positive & results.predicted_positive).sum())
    fn = int((results.image_positive & ~results.predicted_positive).sum())
    fp = int((~results.image_positive & results.predicted_positive).sum())
    tn = int((~results.image_positive & ~results.predicted_positive).sum())
    subgroup_rows = []
    for group, subset in results[results["disease_group"].isin(["benign", "malignant"])].groupby("disease_group"):
        subgroup_rows.append({
            "disease_group": group, "images": len(subset),
            "image_sensitivity": float(subset.predicted_positive.mean()),
            "lesions": int(subset.ground_truth_boxes.sum()),
            "lesion_recall_iou_0_5": float(subset.matched_ground_truth_boxes.sum() / max(subset.ground_truth_boxes.sum(), 1)),
        })
    metrics = {
        "operating_confidence": args.confidence, "matching_iou": args.iou,
        "map_0_5": float(validation.box.map50), "map_0_5_0_95": float(validation.box.map),
        "lesion_detection_recall": matched_gt / max(total_gt, 1),
        "image_sensitivity": tp / max(tp + fn, 1), "image_specificity": tn / max(tn + fp, 1),
        "true_positive_images": tp, "false_negative_images": fn,
        "false_positive_images": fp, "true_negative_images": tn,
        "test_images": len(results), "test_lesions": total_gt,
        "subgroups": subgroup_rows,
        "interpretation": "Dataset-annotation localization only; not a diagnosis or malignancy determination.",
    }
    results.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(output / "benign_malignant_subgroups.csv", index=False)
    false_negatives = results[results.image_positive & ~results.predicted_positive]
    false_positives = results[~results.image_positive & results.predicted_positive]
    false_negatives.to_csv(output / "false_negative_images.csv", index=False)
    false_positives.to_csv(output / "false_positive_images.csv", index=False)
    save_examples(false_negatives, output / "false_negative_examples", "False negative at declared threshold")
    save_examples(false_positives, output / "false_positive_examples", "False positive at declared threshold")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
