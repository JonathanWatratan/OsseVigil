#!/usr/bin/env python3
"""Cross-split exact, source-ID, and perceptual leakage analysis.

This script is read-only with respect to dataset images. Perceptual matches are
review candidates, never patient identities or clinical conclusions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/leakage_analysis"))
    parser.add_argument("--max-phash-distance", type=int, default=6)
    parser.add_argument("--high-confidence-correlation", type=float, default=0.995)
    return parser.parse_args()


def raw_source_id(path: str) -> str:
    name = Path(path).stem.lower()
    name = re.sub(r"\.rf\.[a-f0-9-]+$", "", name)
    name = re.sub(r"_(?:jpg|jpeg|png)$", "", name)
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def phash_and_thumbnail(path: str) -> tuple[int, np.ndarray]:
    pixels = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError(f"Unreadable image: {path}")
    small = cv2.resize(pixels, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    median = float(np.median(low.ravel()[1:]))
    bits = low > median
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    thumb = cv2.resize(pixels, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return value, thumb


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.ravel(), right.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def candidate_pairs(values: list[int], splits: list[str], max_distance: int):
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    seen_pairs = set()
    for index, value in enumerate(values):
        candidates = set()
        for band in range(8):
            key = (band, (value >> (band * 8)) & 0xFF)
            candidates.update(buckets[key])
            buckets[key].append(index)
        for other in candidates:
            if splits[other] == splits[index]:
                continue
            pair = (other, index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            distance = (value ^ values[other]).bit_count()
            if distance <= max_distance:
                yield other, index, distance


def save_montage(pairs: pd.DataFrame, output: Path) -> None:
    rows = []
    for _, pair in pairs.head(12).iterrows():
        panels = []
        for side in ("left", "right"):
            try:
                with Image.open(pair[f"{side}_path"]) as image:
                    panel = ImageOps.fit(image.convert("L"), (220, 180)).convert("RGB")
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 150, 220, 180), fill=(7, 26, 45))
                draw.text((5, 155), f"{pair[f'{side}_split']}: {pair[f'{side}_label']}", fill="white")
                panels.append(panel)
            except Exception:
                panels = []
                break
        if len(panels) == 2:
            row = Image.new("RGB", (440, 180), "white")
            row.paste(panels[0], (0, 0)); row.paste(panels[1], (220, 0))
            rows.append(row)
    if rows:
        montage = Image.new("RGB", (440, 180 * len(rows)), "white")
        for index, row in enumerate(rows):
            montage.paste(row, (0, index * 180))
        montage.save(output)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, keep_default_na=False)
    frame["raw_source_id"] = frame["relative_path"].map(raw_source_id)
    phashes, thumbnails = [], []
    for path in frame["absolute_path"]:
        value, thumb = phash_and_thumbnail(path)
        phashes.append(value); thumbnails.append(thumb)
    frame["phash"] = [f"{value:016x}" for value in phashes]
    frame[["relative_path", "absolute_path", "original_split", "label", "sha256", "raw_source_id", "phash"]].to_csv(
        output / "image_identity_manifest.csv", index=False,
    )

    exact_rows = []
    for sha, group in frame.groupby("sha256"):
        if group["original_split"].nunique() > 1:
            exact_rows.append({
                "sha256": sha, "splits": "/".join(sorted(group["original_split"].unique())),
                "labels": "/".join(sorted(group["label"].unique())), "image_count": len(group),
                "paths": " | ".join(group["relative_path"]),
            })
    exact = pd.DataFrame(exact_rows)
    exact.to_csv(output / "cross_split_exact_groups.csv", index=False)

    source_rows = []
    for source_id, group in frame[frame["raw_source_id"].astype(bool)].groupby("raw_source_id"):
        if group["original_split"].nunique() > 1:
            source_rows.append({
                "raw_source_id": source_id, "splits": "/".join(sorted(group["original_split"].unique())),
                "labels": "/".join(sorted(group["label"].unique())), "image_count": len(group),
                "unique_hashes": group["sha256"].nunique(), "paths": " | ".join(group["relative_path"]),
            })
    source = pd.DataFrame(source_rows)
    source.to_csv(output / "cross_split_source_groups.csv", index=False)

    pair_rows = []
    splits = frame["original_split"].tolist()
    for left, right, distance in candidate_pairs(phashes, splits, args.max_phash_distance):
        if frame.at[left, "sha256"] == frame.at[right, "sha256"]:
            continue
        correlation = normalized_correlation(thumbnails[left], thumbnails[right])
        if correlation < 0.98:
            continue
        confidence = "high" if distance <= 4 and correlation >= args.high_confidence_correlation else "review"
        pair_rows.append({
            "left_path": frame.at[left, "absolute_path"], "right_path": frame.at[right, "absolute_path"],
            "left_relative": frame.at[left, "relative_path"], "right_relative": frame.at[right, "relative_path"],
            "left_split": splits[left], "right_split": splits[right],
            "left_label": frame.at[left, "label"], "right_label": frame.at[right, "label"],
            "phash_distance": distance, "pixel_correlation_128": correlation,
            "confidence": confidence, "same_raw_source_id": frame.at[left, "raw_source_id"] == frame.at[right, "raw_source_id"],
        })
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["confidence", "pixel_correlation_128"], ascending=[True, False])
    pairs.to_csv(output / "cross_split_perceptual_pairs.csv", index=False)
    high = pairs[pairs["confidence"] == "high"] if not pairs.empty else pairs
    save_montage(high, output / "high_confidence_perceptual_pairs.jpg")

    summary = {
        "images": len(frame), "exact_cross_split_groups": len(exact),
        "exact_cross_split_image_rows": int(exact["image_count"].sum()) if len(exact) else 0,
        "raw_source_ids_crossing_splits": len(source),
        "raw_source_groups_with_multiple_labels": int(source["labels"].str.contains("/").sum()) if len(source) else 0,
        "perceptual_cross_split_pairs": len(pairs),
        "high_confidence_nonexact_perceptual_pairs": int((pairs["confidence"] == "high").sum()) if len(pairs) else 0,
        "patient_independence": "not established; no patient/study field exists in CSV/export documentation",
    }
    (output / "leakage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Refined leakage analysis", "", f"- Images: {len(frame):,}",
        f"- Exact cross-split SHA-256 groups: {len(exact):,}",
        f"- Raw source filenames crossing splits: {len(source):,}",
        f"- Non-exact perceptual pairs for review: {len(pairs):,}",
        f"- High-confidence non-exact perceptual pairs: {summary['high_confidence_nonexact_perceptual_pairs']:,}",
        "- Patient/study independence: not established.", "",
        "Perceptual pairs use a 64-bit DCT hash plus normalized 128×128 pixel correlation. They are candidates for review, not proof of patient identity. Exact SHA-256 matches are definitive byte-for-byte duplicates.",
    ]
    (output / "leakage_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

