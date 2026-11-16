#!/usr/bin/env python3
"""Audit BTXRD and create a non-destructive, leakage-aware YOLO dataset.

Original JPEG, LabelMe JSON, and metadata files are read only. The generated
YOLO working set contains image symlinks plus new text labels and manifests.
All lesion subtypes are collapsed to one localization class named ``lesion``;
benign/malignant metadata is retained only for stratification and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
VIEW_COLUMNS = {"frontal", "lateral", "oblique"}
IDENTIFIER_COLUMNS = {"image_id", "image", "filename", "file", "id"}
PATHOLOGY_COLUMNS = {"tumor", "benign", "malignant"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and prepare BTXRD for lesion detection")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/btxrd_detector"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/btxrd_audit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["symlink", "none"], default="symlink")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_release_paths(root: Path):
    directories = {path.name.lower(): path for path in root.iterdir() if path.is_dir()}
    images = directories.get("images")
    annotations = directories.get("annotations")
    metadata_candidates = [
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
        and "dataset" in path.stem.lower()
    ]
    if not images or not annotations or not metadata_candidates:
        raise SystemExit(
            "Expected BTXRD root containing images/, Annotations/, and dataset.csv or dataset.xlsx. "
            f"Found directories={sorted(directories)} metadata={[p.name for p in metadata_candidates]}"
        )
    return images, annotations, sorted(metadata_candidates)[0]


def read_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    frame.columns = [normalize_name(column) for column in frame.columns]
    image_column = next((column for column in frame.columns if column in IDENTIFIER_COLUMNS), None)
    if image_column is None:
        raise SystemExit(f"No image identifier column found in {path}; columns={frame.columns.tolist()}")
    frame = frame.rename(columns={image_column: "image_id"})
    frame["image_id"] = frame["image_id"].astype(str).map(lambda value: Path(value).name)
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_number(name: str) -> int | None:
    match = re.search(r"(\d+)", Path(name).stem)
    return int(match.group(1)) if match else None


def parse_labelme(path: Path, actual_width: int, actual_height: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared_width = int(payload.get("imageWidth") or actual_width)
    declared_height = int(payload.get("imageHeight") or actual_height)
    issues, boxes = [], []
    if (declared_width, declared_height) != (actual_width, actual_height):
        issues.append("annotation_image_dimensions_mismatch")
    for index, shape in enumerate(payload.get("shapes", [])):
        points = shape.get("points") or []
        if len(points) < 2:
            issues.append(f"shape_{index}_too_few_points")
            continue
        coords = np.asarray(points, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
            issues.append(f"shape_{index}_invalid_points")
            continue
        x1, y1 = coords.min(axis=0)
        x2, y2 = coords.max(axis=0)
        if x1 < 0 or y1 < 0 or x2 > actual_width or y2 > actual_height:
            issues.append(f"shape_{index}_out_of_bounds")
        if x2 <= x1 or y2 <= y1:
            issues.append(f"shape_{index}_degenerate_box")
            continue
        boxes.append({
            "label": str(shape.get("label", "lesion")), "shape_type": str(shape.get("shape_type", "polygon")),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        })
    return boxes, issues, declared_width, declared_height


def binary(row: pd.Series, name: str) -> int:
    try:
        return int(float(row.get(name, 0) or 0) > 0)
    except (TypeError, ValueError):
        return 0


def disease_group(row: pd.Series) -> str:
    if binary(row, "malignant"):
        return "malignant"
    if binary(row, "benign"):
        return "benign"
    if binary(row, "tumor"):
        return "tumor_unspecified"
    return "normal"


def anatomy_group(row: pd.Series) -> str:
    for column in ["upper_limb", "lower_limb", "pelvis"]:
        if binary(row, column):
            return column
    anatomy_columns = [
        "hand", "ulna", "radius", "humerus", "foot", "tibia", "fibula", "femur", "hip_bone",
        "ankle_joint", "knee_joint", "hip_joint", "wrist_joint", "elbow_joint", "shoulder_joint",
    ]
    active = [column for column in anatomy_columns if binary(row, column)]
    return "+".join(active) if active else "unspecified"


def metadata_signature(row: pd.Series, columns: list[str]) -> str:
    values = []
    for column in columns:
        value = row.get(column, "")
        values.append(f"{column}={'' if pd.isna(value) else str(value).strip().lower()}")
    return "|".join(values)


def assign_splits(frame: pd.DataFrame, seed: int) -> pd.Series:
    group_rows = []
    for group_id, group in frame.groupby("leakage_group", sort=True):
        group_rows.append({
            "leakage_group": group_id,
            "stratum": "|".join([
                group["disease_group"].mode().iat[0],
                str(group["center"].mode().iat[0]) if "center" in group else "unknown",
                group["anatomy_group"].mode().iat[0],
            ]),
            "count": len(group),
            "tie": hashlib.sha256(f"{seed}|{group_id}".encode()).hexdigest(),
        })
    groups = pd.DataFrame(group_rows)
    assignment = {}
    for _, subset in groups.groupby("stratum", sort=True):
        subset = subset.sort_values(["count", "tie"], ascending=[False, True])
        total = int(subset["count"].sum())
        targets = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
        current = {name: 0 for name in SPLIT_RATIOS}
        for item in subset.itertuples(index=False):
            split = max(SPLIT_RATIOS, key=lambda name: ((targets[name] - current[name]) / max(targets[name], 1), SPLIT_RATIOS[name]))
            assignment[item.leakage_group] = split
            current[split] += item.count
    return frame["leakage_group"].map(assignment)


def yolo_line(box: dict, width: int, height: int) -> str:
    cx = ((box["x1"] + box["x2"]) / 2) / width
    cy = ((box["y1"] + box["y2"]) / 2) / height
    bw = (box["x2"] - box["x1"]) / width
    bh = (box["y2"] - box["y1"]) / height
    values = [cx, cy, bw, bh]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError(f"Invalid normalized YOLO box: {values}")
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def main():
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    report = args.report_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"BTXRD directory does not exist: {root}")
    if root == output or root in output.parents:
        raise SystemExit("Working output must be outside the original BTXRD directory.")
    images_dir, annotations_dir, metadata_path = find_release_paths(root)
    metadata = read_metadata(metadata_path)
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    annotation_paths = {path.stem: path for path in annotations_dir.glob("*.json")}
    metadata_by_name = metadata.set_index("image_id", drop=False)
    metadata_by_stem = {Path(name).stem: row for name, row in metadata_by_name.iterrows()}
    rows, issues = [], []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        meta = metadata_by_stem.get(image_path.stem)
        if meta is None:
            issues.append({"image_id": image_path.name, "issue": "missing_metadata", "detail": ""})
            meta = pd.Series(dtype=object)
        annotation_path = annotation_paths.get(image_path.stem)
        boxes, annotation_issues = [], []
        if annotation_path:
            boxes, annotation_issues, _, _ = parse_labelme(annotation_path, width, height)
        group = disease_group(meta)
        tumor_expected = group != "normal"
        if tumor_expected and not annotation_path:
            issues.append({"image_id": image_path.name, "issue": "tumor_missing_annotation", "detail": ""})
        if tumor_expected and not boxes:
            issues.append({"image_id": image_path.name, "issue": "tumor_has_no_valid_box", "detail": ""})
        if not tumor_expected and boxes:
            issues.append({"image_id": image_path.name, "issue": "normal_has_boxes", "detail": str(len(boxes))})
        for value in annotation_issues:
            issues.append({"image_id": image_path.name, "issue": value, "detail": str(annotation_path)})
        row = {column: meta.get(column, "") for column in metadata.columns}
        row.update({
            "image_id": image_path.name, "absolute_path": str(image_path.resolve()),
            "annotation_path": str(annotation_path.resolve()) if annotation_path else "",
            "width": width, "height": height, "sha256": sha256(image_path),
            "bbox_count": len(boxes), "boxes_json": json.dumps(boxes, separators=(",", ":")),
            "disease_group": group, "anatomy_group": anatomy_group(meta),
            "image_number": image_number(image_path.name), "valid_for_detection": not annotation_issues,
        })
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["image_number", "image_id"], na_position="last").reset_index(drop=True)

    image_stems = {path.stem for path in image_paths}
    for stem, path in annotation_paths.items():
        if stem not in image_stems:
            issues.append({"image_id": stem, "issue": "annotation_missing_image", "detail": str(path)})
    for name in metadata["image_id"]:
        if Path(str(name)).stem not in image_stems:
            issues.append({"image_id": str(name), "issue": "metadata_missing_image", "detail": ""})

    union = UnionFind(len(frame))
    exact_groups = 0
    for _, group in frame.groupby("sha256"):
        if len(group) > 1:
            exact_groups += 1
            first = group.index[0]
            for index in group.index[1:]:
                union.union(first, index)

    # The paper states that multiple projections can depict the same lesion but
    # provides no patient ID. Consecutive IDs with identical non-view metadata
    # are therefore grouped as a conservative, explicitly heuristic proxy.
    excluded_signature = {"image_id", *VIEW_COLUMNS}
    signature_columns = [column for column in metadata.columns if column not in excluded_signature]
    frame["metadata_signature"] = frame.apply(lambda row: metadata_signature(row, signature_columns), axis=1)
    proxy_pairs = 0
    for left in range(len(frame) - 1):
        right = left + 1
        left_number, right_number = frame.at[left, "image_number"], frame.at[right, "image_number"]
        if left_number is None or right_number is None or right_number - left_number != 1:
            continue
        if frame.at[left, "metadata_signature"] == frame.at[right, "metadata_signature"]:
            union.union(left, right)
            proxy_pairs += 1
    frame["leakage_group"] = [f"group_{union.find(index):06d}" for index in frame.index]
    frame["split"] = assign_splits(frame, args.seed)

    invalid_ids = {item["image_id"] for item in issues if item["issue"] in {
        "missing_metadata", "tumor_missing_annotation", "tumor_has_no_valid_box", "normal_has_boxes",
        "annotation_image_dimensions_mismatch", "annotation_missing_image", "metadata_missing_image",
    } or "invalid" in item["issue"] or "out_of_bounds" in item["issue"] or "degenerate" in item["issue"]}
    frame["eligible"] = ~frame["image_id"].isin(invalid_ids)
    eligible = frame[frame["eligible"]].copy()

    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report / "btxrd_manifest_all.csv", index=False)
    pd.DataFrame(issues, columns=["image_id", "issue", "detail"]).to_csv(report / "btxrd_issues.csv", index=False)
    eligible.to_csv(output / "manifest.csv", index=False)
    for split in SPLIT_RATIOS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for item in eligible.itertuples(index=False):
        source = Path(item.absolute_path)
        image_target = output / "images" / item.split / source.name
        label_target = output / "labels" / item.split / f"{source.stem}.txt"
        if args.link_mode == "symlink" and not image_target.exists():
            image_target.symlink_to(source)
        boxes = json.loads(item.boxes_json)
        label_target.write_text("\n".join(yolo_line(box, item.width, item.height) for box in boxes) + ("\n" if boxes else ""), encoding="utf-8")
    yaml_text = "\n".join([
        f"path: {output}", "train: images/train", "val: images/val", "test: images/test",
        "names:", "  0: lesion", "",
    ])
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")

    exact_cross = int((eligible.groupby("sha256")["split"].nunique() > 1).sum())
    group_cross = int((eligible.groupby("leakage_group")["split"].nunique() > 1).sum())
    distribution = eligible.groupby(["split", "disease_group"]).size().unstack(fill_value=0)
    center_table = eligible.groupby(["split", "center", "disease_group"]).size().unstack(fill_value=0) if "center" in eligible else pd.DataFrame()
    anatomy_table = eligible.groupby(["split", "anatomy_group", "disease_group"]).size().unstack(fill_value=0)
    distribution.to_csv(report / "split_distribution.csv")
    center_table.to_csv(report / "center_distribution.csv")
    anatomy_table.to_csv(report / "anatomy_distribution.csv")
    summary = {
        "dataset_root": str(root), "source_files_modified": False, "images": len(frame),
        "annotations": len(annotation_paths), "metadata_rows": len(metadata), "audit_issues": len(issues),
        "eligible_images": len(eligible), "exact_duplicate_groups": exact_groups,
        "proxy_consecutive_same_metadata_pairs": proxy_pairs,
        "patient_independence": "not established; no patient identifier is supplied",
        "split_seed": args.seed, "split_distribution": distribution.to_dict("index"),
        "cross_split_exact_groups": exact_cross, "cross_split_leakage_groups": group_cross,
        "detector_class": {0: "lesion"},
        "limitations": [
            "Proxy lesion grouping can merge unrelated cases or miss same-lesion views.",
            "Centers are stratified across splits, not held out as independent institutions.",
            "Bounding boxes localize annotated lesions but do not independently identify malignancy.",
        ],
    }
    (report / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if issues:
        raise SystemExit("BTXRD audit produced blocking issues; review reports/btxrd_audit/btxrd_issues.csv before training.")


if __name__ == "__main__":
    main()
