#!/usr/bin/env python3
"""Pretrained feature extraction and K-Means exploration for unverified datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from osteoinsight.common import DISCLAIMER, seed_everything, select_device


class ImageOnlyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = self.frame.at[index, "absolute_path"]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/embeddings"))
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_montage(paths: list[str], title: str, output: Path):
    tiles = []
    for path in paths[:16]:
        try:
            with Image.open(path) as image:
                tile = ImageOps.fit(image.convert("L"), (180, 180)).convert("RGB")
                draw = ImageDraw.Draw(tile)
                draw.rectangle((0, 158, 180, 180), fill=(5, 20, 38))
                draw.text((5, 162), Path(path).name[:25], fill="white")
                tiles.append(tile)
        except Exception:
            continue
    canvas = Image.new("RGB", (720, 40 + 180 * 4), "white")
    ImageDraw.Draw(canvas).text((10, 10), title, fill=(5, 20, 38))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 4) * 180, 40 + (index // 4) * 180))
    canvas.save(output)


def main():
    args = parse_args()
    seed_everything(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest)
    frame = frame[frame["readable"] == True].copy()  # noqa: E712
    if len(frame) < args.clusters:
        raise SystemExit("The cluster count cannot exceed the number of readable images.")
    weights = ResNet18_Weights.DEFAULT
    dataset = ImageOnlyDataset(frame, weights.transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = select_device()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    vectors, paths = [], []
    with torch.inference_mode():
        for images, batch_paths in loader:
            vectors.append(model(images.to(device)).cpu().numpy())
            paths.extend(batch_paths)
    embeddings = np.concatenate(vectors).astype("float32")
    kmeans = KMeans(n_clusters=args.clusters, n_init="auto", random_state=args.seed).fit(embeddings)
    pca = PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)
    results = pd.DataFrame({"absolute_path": paths, "cluster": kmeans.labels_, "pca_x": pca[:, 0], "pca_y": pca[:, 1]})
    source_metadata = frame.set_index("absolute_path")
    for column in ["label", "original_split", "source_id_heuristic"]:
        if column in source_metadata.columns:
            results[column] = results["absolute_path"].map(source_metadata[column])
    results.to_csv(output / "cluster_assignments.csv", index=False)
    if "label" in results.columns:
        results.groupby(["cluster", "label"]).size().unstack(fill_value=0).to_csv(output / "cluster_label_distribution.csv")
    if "original_split" in results.columns:
        results.groupby(["cluster", "original_split"]).size().unstack(fill_value=0).to_csv(output / "cluster_split_distribution.csv")
    np.save(output / "embeddings.npy", embeddings)
    joblib.dump(kmeans, output / "kmeans.joblib")
    metadata = {
        "encoder": "torchvision ResNet18 ImageNet-1K v1 penultimate-layer features",
        "cluster_count": args.clusters, "device": device, "image_count": len(results),
        "disclaimer": DISCLAIMER,
        "warning": "Clusters are visual-similarity groups, not medically validated diagnoses.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(results.pca_x, results.pca_y, c=results.cluster, cmap="tab10", s=12, alpha=0.75)
    ax.set_title("Exploratory image clusters — not diagnostic")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    fig.colorbar(scatter, ax=ax, label="Visual cluster")
    fig.tight_layout()
    fig.savefig(output / "cluster_map.png", dpi=180)
    plt.close(fig)
    for cluster, group in results.groupby("cluster"):
        centroid = kmeans.cluster_centers_[cluster]
        member_idx = group.index.to_numpy()
        distances = np.linalg.norm(embeddings[member_idx] - centroid, axis=1)
        samples = group.iloc[np.argsort(distances)[:16]]["absolute_path"].tolist()
        make_montage(samples, f"Visual cluster {cluster} — NOT a diagnosis", output / f"cluster_{cluster:02d}_samples.jpg")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
