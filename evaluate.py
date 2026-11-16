#!/usr/bin/env python3
"""Held-out evaluation with a majority-class baseline and uncertainty artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from osteoinsight.common import DISCLAIMER, select_device
from osteoinsight.data import (
    ManifestDataset, classification_splits_from_frame, load_roboflow_csv_manifest,
    make_transforms, manifest_fingerprint,
)
from osteoinsight.modeling import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--roboflow-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/osteoinsight_best.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--positive-class", default="cancer", help="Positive category for binary sensitivity/specificity")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def save_error_montage(errors: pd.DataFrame, output: Path) -> None:
    """Create a compact visual review sheet without altering or copying source images."""
    tiles = []
    for _, row in errors.head(16).iterrows():
        try:
            with Image.open(row["path"]) as image:
                tile = ImageOps.fit(image.convert("L"), (220, 190)).convert("RGB")
            canvas = Image.new("RGB", (220, 240), "white")
            canvas.paste(tile, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 190, 220, 240), fill=(7, 26, 45))
            draw.text((5, 195), f"source: {str(row['true_label'])[:24]}", fill="white")
            draw.text((5, 213), f"pred: {str(row['predicted_label'])[:25]}", fill=(99, 211, 229))
            tiles.append(canvas)
        except Exception:
            continue
    if not tiles:
        return
    montage = Image.new("RGB", (880, 240 * 4), "white")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % 4) * 220, (index // 4) * 240))
    montage.save(output)


def binary_metrics(y_true, y_pred, positive_index: int, negative_index: int) -> dict:
    tp = int(((y_true == positive_index) & (y_pred == positive_index)).sum())
    fn = int(((y_true == positive_index) & (y_pred == negative_index)).sum())
    fp = int(((y_true == negative_index) & (y_pred == positive_index)).sum())
    tn = int(((y_true == negative_index) & (y_pred == negative_index)).sum())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
        "count": len(y_true),
    }


def subgroup_rows(predictions: pd.DataFrame, column: str, positive_index: int, negative_index: int) -> list[dict]:
    rows = []
    for value, group in predictions.groupby(column, dropna=False):
        row = {"field": column, "group": str(value), "count": len(group)}
        if group["true_index"].nunique() == 2:
            row.update(binary_metrics(
                group["true_index"].to_numpy(), group["predicted_index"].to_numpy(),
                positive_index, negative_index,
            ))
            row["note"] = "Exploratory filename-derived subgroup; not independently verified metadata."
        else:
            row["note"] = "Single-label subgroup; balanced metrics are not interpretable."
            row["accuracy"] = float((group["true_index"] == group["predicted_index"]).mean())
        rows.append(row)
    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.manifest) == bool(args.roboflow_root):
        raise SystemExit("Provide exactly one of: manifest path or --roboflow-root.")
    manifest = pd.read_csv(args.manifest) if args.manifest else load_roboflow_csv_manifest(args.roboflow_root)
    splits, class_to_idx = classification_splits_from_frame(manifest)
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if class_to_idx != checkpoint["class_to_idx"]:
        raise SystemExit("Checkpoint class mapping does not match the manifest.")
    current_fingerprint = manifest_fingerprint(manifest)
    if checkpoint.get("manifest_fingerprint") != current_fingerprint:
        raise SystemExit("Checkpoint was trained from a different manifest fingerprint.")
    _, transform = make_transforms(int(checkpoint.get("image_size", 224)))
    dataset = ManifestDataset(splits["test"], class_to_idx, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    y_true, y_pred, probabilities, paths = [], [], [], []
    with torch.inference_mode():
        for images, targets, batch_paths in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(probs.argmax(axis=1).tolist())
            probabilities.extend(probs.tolist())
            paths.extend(batch_paths)
    y_true_arr, y_pred_arr, probs_arr = np.array(y_true), np.array(y_pred), np.array(probabilities)
    names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=np.arange(len(names)), zero_division=0,
    )
    per_class = []
    for index, name in enumerate(names):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        per_class.append({
            "class": name, "precision": precision[index], "recall_sensitivity": recall[index],
            "specificity_one_vs_rest": tn / max(tn + fp, 1), "f1": f1[index], "support": int(support[index]),
        })
    baseline_class = int(pd.Series(y_true_arr).value_counts().idxmax())
    baseline_pred = np.full_like(y_true_arr, baseline_class)
    metrics = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "macro_f1": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "majority_baseline_accuracy": accuracy_score(y_true_arr, baseline_pred),
        "majority_baseline_balanced_accuracy": balanced_accuracy_score(y_true_arr, baseline_pred),
        "majority_baseline_macro_f1": f1_score(y_true_arr, baseline_pred, average="macro", zero_division=0),
        "test_images": len(y_true_arr), "device": device,
        "checkpoint_best_validation_loss": checkpoint.get("best_val_loss"),
        "disclaimer": DISCLAIMER,
    }
    if len(names) == 2:
        if args.positive_class not in class_to_idx:
            raise SystemExit(f"Positive class '{args.positive_class}' is absent; available classes: {names}")
        positive_index = class_to_idx[args.positive_class]
        negative_index = next(index for index in range(2) if index != positive_index)
        metrics.update({"positive_class": args.positive_class, **binary_metrics(y_true_arr, y_pred_arr, positive_index, negative_index)})
    try:
        if len(names) == 2:
            binary_truth = (y_true_arr == positive_index).astype(int)
            metrics["roc_auc"] = roc_auc_score(binary_truth, probs_arr[:, positive_index])
        else:
            binary = label_binarize(y_true_arr, classes=np.arange(len(names)))
            metrics["macro_roc_auc_ovr"] = roc_auc_score(binary, probs_arr, average="macro", multi_class="ovr")
    except ValueError as exc:
        metrics["roc_auc_note"] = str(exc)
    pd.DataFrame(per_class).to_csv(args.output_dir / "per_class_metrics.csv", index=False)
    predictions = pd.DataFrame({
        "path": paths, "true_index": y_true_arr, "predicted_index": y_pred_arr,
        "true_label": [names[i] for i in y_true_arr], "predicted_label": [names[i] for i in y_pred_arr],
        "confidence_uncalibrated": probs_arr.max(axis=1), "correct": y_true_arr == y_pred_arr,
    })
    metadata_columns = [
        column for column in ["source_family_hint", "anatomy_hint_unverified", "source_group", "original_split"]
        if column in splits["test"].columns
    ]
    if metadata_columns:
        metadata = splits["test"].set_index("absolute_path")[metadata_columns]
        predictions = predictions.join(metadata, on="path")
    if "sha256" in manifest.columns and manifest["sha256"].astype(str).str.len().gt(20).all():
        path_to_hash = manifest.set_index("absolute_path")["sha256"].astype(str)
        non_test_hashes = set(manifest.loc[manifest["split"] != "test", "sha256"].astype(str))
        predictions["sha256"] = predictions["path"].map(path_to_hash)
        predictions["exact_copy_seen_outside_test"] = predictions["sha256"].isin(non_test_hashes)
        clean = predictions[~predictions["exact_copy_seen_outside_test"]]
        if len(names) == 2 and len(clean) and clean["true_index"].nunique() == 2:
            metrics["exact_duplicate_excluded_test"] = binary_metrics(
                clean["true_index"].to_numpy(), clean["predicted_index"].to_numpy(), positive_index, negative_index,
            )
            metrics["exact_duplicate_excluded_test"]["warning"] = "Exploratory subset only; near duplicates and patient/source leakage may remain."
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    if len(names) == 2 and metadata_columns:
        subgroup_output = []
        for column in ["source_family_hint", "anatomy_hint_unverified"]:
            if column in predictions.columns:
                subgroup_output.extend(subgroup_rows(predictions, column, positive_index, negative_index))
        pd.DataFrame(subgroup_output).to_csv(args.output_dir / "heuristic_subgroup_metrics.csv", index=False)
    errors = predictions[~predictions["correct"]].sort_values("confidence_uncalibrated", ascending=False)
    errors.to_csv(args.output_dir / "incorrect_predictions.csv", index=False)
    if len(names) == 2:
        errors[(errors["true_index"] == positive_index) & (errors["predicted_index"] == negative_index)].to_csv(
            args.output_dir / "false_negatives.csv", index=False,
        )
        errors[(errors["true_index"] == negative_index) & (errors["predicted_index"] == positive_index)].to_csv(
            args.output_dir / "false_positives.csv", index=False,
        )
    save_error_montage(errors, args.output_dir / "incorrect_predictions_montage.jpg")
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), max(5, len(names) * 0.7)))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=names, yticklabels=names, cmap="Blues", ax=ax)
    ax.set(xlabel="Predicted dataset category", ylabel="Source label", title="Held-out confusion matrix")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    report = classification_report(y_true_arr, y_pred_arr, target_names=names, zero_division=0)
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
