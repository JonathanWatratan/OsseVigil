#!/usr/bin/env python3
"""Transfer-learning trainer for source-verified classification labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from osteoinsight.common import seed_everything, select_device
from osteoinsight.data import (
    ManifestDataset, classification_splits_from_frame, load_classification_splits,
    load_roboflow_csv_manifest, make_transforms, manifest_fingerprint,
)
from osteoinsight.modeling import build_classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", help="Audited split manifest")
    parser.add_argument("--roboflow-root", type=Path, help="Load labels directly from train/valid/test _classes.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--architecture", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Train only the pretrained classification head (fast transfer-learning baseline)",
    )
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count, targets_all, predictions_all = 0.0, 0, 0, [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            predictions = logits.argmax(1)
            correct += (predictions == targets).sum().item()
            count += len(targets)
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
    macro_f1 = f1_score(targets_all, predictions_all, average="macro", zero_division=0)
    return total_loss / max(count, 1), correct / max(count, 1), macro_f1


def extract_frozen_features(model, loader, device, architecture: str):
    model.eval()
    features_all, targets_all = [], []
    with torch.inference_mode():
        for images, targets, _ in loader:
            images = images.to(device)
            if architecture == "resnet18":
                x = model.conv1(images)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                features = torch.flatten(x, 1)
            else:
                x = model.features(images)
                x = model.avgpool(x)
                features = torch.flatten(x, 1)
            features_all.append(features.cpu())
            targets_all.append(targets.cpu())
    return torch.cat(features_all), torch.cat(targets_all)


def run_feature_epoch(head, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    head.train(training)
    total_loss, correct, count, targets_all, predictions_all = 0.0, 0, 0, [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(1)
            total_loss += loss.item() * len(targets)
            correct += (predictions == targets).sum().item()
            count += len(targets)
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
    macro_f1 = f1_score(targets_all, predictions_all, average="macro", zero_division=0)
    return total_loss / max(count, 1), correct / max(count, 1), macro_f1


def plot_history(history: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.train_accuracy, label="train")
    axes[1].plot(history.epoch, history.val_accuracy, label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def enforce_exact_duplicate_gate(manifest: pd.DataFrame) -> pd.DataFrame:
    """Refuse fitting when exact image bytes cross source-provided splits."""
    manifest = manifest.copy()
    hashes_ready = "sha256" in manifest.columns and manifest["sha256"].astype(str).str.len().gt(20).all()
    if not hashes_ready:
        values = []
        for path in manifest["absolute_path"]:
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            values.append(digest.hexdigest())
        manifest["sha256"] = values
    crossing = manifest.groupby("sha256")["split"].nunique()
    crossing = crossing[crossing > 1]
    if len(crossing):
        affected = int(manifest["sha256"].isin(crossing.index).sum())
        raise SystemExit(
            f"Training gate failed: {len(crossing)} exact duplicate groups cross train/val/test "
            f"({affected} image rows). Resolve leakage in a copied manifest; source images remain unchanged."
        )
    return manifest


def main():
    args = parse_args()
    seed_everything(args.seed)
    if bool(args.manifest) == bool(args.roboflow_root):
        raise SystemExit("Provide exactly one of: manifest path or --roboflow-root.")
    if args.manifest:
        adjacent_summary = args.manifest.parent / "leakage_reduced_manifest_summary.json"
        if adjacent_summary.exists():
            decision = json.loads(adjacent_summary.read_text(encoding="utf-8"))
            if not decision.get("training_gate_pass", False):
                raise SystemExit(f"Training gate failed: {decision.get('training_gate_reason', 'unresolved audit risk')}")
    manifest = pd.read_csv(args.manifest) if args.manifest else load_roboflow_csv_manifest(args.roboflow_root)
    manifest = enforce_exact_duplicate_gate(manifest)
    splits, class_to_idx = classification_splits_from_frame(manifest)
    verification = splits["train"]["label_verified"].astype(str).str.lower().isin({"true", "1", "yes"})
    if not verification.all():
        raise SystemExit("Refusing training because the training labels are not explicitly source-verified.")
    train_tf, eval_tf = make_transforms(args.image_size)
    datasets = {
        "train": ManifestDataset(splits["train"], class_to_idx, train_tf),
        "val": ManifestDataset(splits["val"], class_to_idx, eval_tf),
    }
    device = select_device(args.device)
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(name == "train"),
            num_workers=args.workers, pin_memory=(device == "cuda"),
        )
        for name, dataset in datasets.items()
    }
    model = build_classifier(len(class_to_idx), args.architecture, pretrained=True).to(device)
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        head = model.fc if args.architecture == "resnet18" else model.classifier
        for parameter in head.parameters():
            parameter.requires_grad = True
    train_targets = splits["train"]["label"].astype(str).map(class_to_idx).to_numpy()
    weights = compute_class_weight("balanced", classes=np.arange(len(class_to_idx)), y=train_targets)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.3)
    feature_loaders = None
    if args.freeze_backbone:
        print("Extracting frozen pretrained features once for the head-only baseline...")
        train_features, cached_train_targets = extract_frozen_features(model, loaders["train"], device, args.architecture)
        val_features, cached_val_targets = extract_frozen_features(model, loaders["val"], device, args.architecture)
        feature_loaders = {
            "train": DataLoader(TensorDataset(train_features, cached_train_targets), batch_size=512, shuffle=True),
            "val": DataLoader(TensorDataset(val_features, cached_val_targets), batch_size=512, shuffle=False),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "osteoinsight_best.pth"
    history_rows, best_loss, best_f1, stale = [], float("inf"), -1.0, 0
    for epoch in range(1, args.epochs + 1):
        if feature_loaders:
            train_loss, train_acc, train_f1 = run_feature_epoch(head, feature_loaders["train"], criterion, device, optimizer)
            val_loss, val_acc, val_f1 = run_feature_epoch(head, feature_loaders["val"], criterion, device)
        else:
            train_loss, train_acc, train_f1 = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_loss, val_acc, val_f1 = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)
        history_rows.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_acc, "val_accuracy": val_acc,
            "train_macro_f1": train_f1, "val_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}"
        )
        improved = val_f1 > best_f1 + 1e-4 or (abs(val_f1 - best_f1) <= 1e-4 and val_loss < best_loss)
        if improved:
            best_loss, best_f1, stale = val_loss, val_f1, 0
            torch.save({
                "model_state_dict": model.state_dict(), "architecture": args.architecture,
                "class_to_idx": class_to_idx, "image_size": args.image_size,
                "manifest_fingerprint": manifest_fingerprint(manifest), "best_val_loss": best_loss,
                "best_val_macro_f1": best_f1,
                "training_args": vars(args), "device_used": device,
                "label_semantics": "dataset categories from source-verified manifest; not clinical diagnoses",
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    history = pd.DataFrame(history_rows)
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    plot_history(history, args.output_dir / "training_history.png")
    (args.output_dir / "class_mapping.json").write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")
    print(
        f"Best checkpoint: {best_path} (validation macro-F1 {best_f1:.4f}; "
        f"validation loss {best_loss:.4f}; device {device})"
    )


if __name__ == "__main__":
    main()
