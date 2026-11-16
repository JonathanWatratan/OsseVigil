#!/usr/bin/env python3
"""Read-only dataset discovery for OsteoInsight AI.

This script never modifies source data. It writes all artifacts to --output-dir.
Folder-derived labels remain explicitly unverified unless --labels-trusted is supplied
after the provenance has been checked by a human.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif",
}
SPLIT_NAMES = {"train": "train", "training": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}
NON_CLASS_DIRS = {"images", "image", "imgs", "labels", "annotations", "dataset", "data"}


@dataclass
class AnnotationEvidence:
    yaml_files: list[str] = field(default_factory=list)
    coco_files: list[str] = field(default_factory=list)
    csv_files: list[str] = field(default_factory=list)
    yolo_label_files: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    annotation_class_counts: dict[str, int] = field(default_factory=dict)
    csv_candidate_label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    csv_columns: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an explicitly authorized image dataset without modifying it.")
    parser.add_argument("dataset", type=Path, help="Authorized dataset folder")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Artifact output folder")
    parser.add_argument("--labels-trusted", action="store_true", help="Mark discovered labels as source-verified (requires human provenance review)")
    parser.add_argument("--near-duplicate-distance", type=int, default=5, help="Maximum 64-bit dHash Hamming distance")
    parser.add_argument("--modality-sample-size", type=int, default=200)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: Image.Image) -> int:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def image_split_and_folder_label(path: Path, root: Path) -> tuple[str | None, str | None]:
    parts = path.relative_to(root).parts[:-1]
    normalized = [part.lower() for part in parts]
    split = next((SPLIT_NAMES[p] for p in normalized if p in SPLIT_NAMES), None)
    candidates = [part for part in parts if part.lower() not in SPLIT_NAMES and part.lower() not in NON_CLASS_DIRS]
    return split, (candidates[-1] if candidates else None)


def extract_identifiers(path: Path) -> tuple[str | None, str | None]:
    text = str(path)
    patient_patterns = [r"(?i)(?:patient|pat|pt)[-_ ]?([a-z0-9]{3,})", r"(?i)(?:subject|subj)[-_ ]?([a-z0-9]{3,})"]
    study_patterns = [r"(?i)(?:study|exam|series)[-_ ]?([a-z0-9]{3,})"]
    patient = next((m.group(1) for p in patient_patterns if (m := re.search(p, text))), None)
    study = next((m.group(1) for p in study_patterns if (m := re.search(p, text))), None)
    return patient, study


def inspect_image(path: Path, root: Path) -> dict:
    split, folder_label = image_split_and_folder_label(path, root)
    patient_id, study_id = extract_identifiers(path.relative_to(root))
    row = {
        "relative_path": str(path.relative_to(root)),
        "absolute_path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "readable": False,
        "error": "",
        "width": None,
        "height": None,
        "mode": "",
        "format": "",
        "sha256": "",
        "dhash": "",
        "duplicate_group": "",
        "near_duplicate_group": "",
        "split": split or "",
        "label": folder_label or "",
        "label_source": "folder_structure" if folder_label else "",
        "label_verified": False,
        "patient_id": patient_id or "",
        "study_id": study_id or "",
        "roboflow_hint": "roboflow" in str(path).lower() or ".rf." in path.name.lower(),
        "grayscale_like": None,
    }
    try:
        row["sha256"] = sha256_file(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            row.update(width=image.width, height=image.height, mode=image.mode, format=image.format or "")
            row["dhash"] = f"{dhash(image):016x}"
            thumb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.int16)
            channel_delta = np.mean(np.max(thumb, axis=2) - np.min(thumb, axis=2))
            row["grayscale_like"] = bool(channel_delta < 4.0)
        row["readable"] = True
    except Exception as exc:  # the error is an audit result, not a reason to stop the whole scan
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def discover_annotations(root: Path, image_stems: set[str]) -> AnnotationEvidence:
    evidence = AnnotationEvidence()
    numeric_class_counts: Counter[str] = Counter()
    declared_names: dict[int, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        relative = str(path.relative_to(root))
        if path.name.lower() in {"data.yaml", "data.yml"}:
            evidence.yaml_files.append(relative)
            try:
                import yaml
                payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
                names = payload.get("names", [])
                if isinstance(names, dict):
                    names = [names[key] for key in sorted(names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
                evidence.class_names.extend(str(name) for name in names)
                declared_names.update({index: str(name) for index, name in enumerate(names)})
            except Exception as exc:
                evidence.notes.append(f"Could not parse {relative}: {exc}")
        elif suffix == ".json":
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict) and {"images", "annotations", "categories"}.issubset(payload):
                    evidence.coco_files.append(relative)
                    category_names = {item.get("id"): str(item.get("name")) for item in payload["categories"] if item.get("name")}
                    evidence.class_names.extend(category_names.values())
                    for item in payload["annotations"]:
                        name = category_names.get(item.get("category_id"), str(item.get("category_id")))
                        numeric_class_counts[name] += 1
            except Exception:
                pass
        elif suffix == ".csv":
            evidence.csv_files.append(relative)
            try:
                sample = pd.read_csv(path, nrows=10000)
                evidence.csv_columns[relative] = [str(column) for column in sample.columns]
                for column in sample.columns:
                    normalized = str(column).strip().lower()
                    if normalized in {"label", "class", "category", "target", "diagnosis", "finding"}:
                        counts = sample[column].dropna().astype(str).value_counts()
                        if 1 < len(counts) <= 100:
                            evidence.csv_candidate_label_counts[f"{relative}:{column}"] = {
                                str(name): int(count) for name, count in counts.items()
                            }
            except Exception as exc:
                evidence.notes.append(f"Could not inspect CSV {relative}: {exc}")
        elif suffix == ".txt" and path.stem in image_stems:
            try:
                lines = [line.strip().split() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
                if lines and all(len(line) >= 5 and line[0].isdigit() for line in lines):
                    evidence.yolo_label_files.append(relative)
                    for line in lines:
                        numeric_class_counts[line[0]] += 1
            except OSError:
                pass
    evidence.class_names = sorted(set(evidence.class_names))
    translated = Counter()
    for key, count in numeric_class_counts.items():
        name = declared_names.get(int(key), key) if str(key).isdigit() else key
        translated[str(name)] += count
    evidence.annotation_class_counts = dict(sorted(translated.items()))
    return evidence


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def assign_duplicate_groups(frame: pd.DataFrame, max_distance: int) -> pd.DataFrame:
    readable = frame[frame["readable"]].copy()
    exact_groups = readable.groupby("sha256").groups
    exact_id = 0
    for _, indices in exact_groups.items():
        if len(indices) > 1:
            exact_id += 1
            frame.loc[list(indices), "duplicate_group"] = f"exact_{exact_id:05d}"

    indices = list(readable.index)
    values = [int(frame.at[index, "dhash"], 16) for index in indices]
    dsu = DisjointSet(len(indices))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for position, value in enumerate(values):
        candidates: set[int] = set()
        # Eight exact 8-bit bands guarantee a candidate collision for hashes
        # within five bit flips (pigeonhole principle), avoiding an O(n^2) scan.
        for band in range(8):
            key = (band, (value >> (band * 8)) & 0xFF)
            candidates.update(buckets[key])
            buckets[key].append(position)
        for other in candidates:
            if (value ^ values[other]).bit_count() <= max_distance:
                dsu.union(position, other)
    grouped: dict[int, list[int]] = defaultdict(list)
    for position in range(len(indices)):
        grouped[dsu.find(position)].append(position)
    near_id = 0
    for members in grouped.values():
        if len(members) > 1:
            near_id += 1
            source_indices = [indices[position] for position in members]
            frame.loc[source_indices, "near_duplicate_group"] = f"near_{near_id:05d}"
    return frame


def choose_scenario(frame: pd.DataFrame, evidence: AnnotationEvidence, labels_trusted: bool) -> tuple[str, str]:
    has_detection = bool(evidence.coco_files or evidence.yolo_label_files)
    folder_labels = frame.loc[frame["label"].astype(bool), "label"].nunique()
    if has_detection:
        return "B", "Object-detection annotation files were discovered; their provenance and geometry must be validated."
    if labels_trusted and folder_labels >= 2:
        return "A", "At least two source-verified classification labels are available."
    if folder_labels >= 2:
        return "C", "Folder categories exist but were not declared source-verified; they are retained only as metadata."
    return "C", "No reliable supervised labels were discovered; use exploratory pretrained embeddings only."


def quality_rows(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, item in frame.iterrows():
        if not item["readable"]:
            rows.append({"relative_path": item["relative_path"], "issue": "unreadable", "severity": "error", "detail": item["error"]})
        if item["readable"] and (item["width"] < 128 or item["height"] < 128):
            rows.append({"relative_path": item["relative_path"], "issue": "low_resolution", "severity": "warning", "detail": f"{item['width']}x{item['height']}"})
        if item["duplicate_group"]:
            rows.append({"relative_path": item["relative_path"], "issue": "exact_duplicate", "severity": "warning", "detail": item["duplicate_group"]})
        elif item["near_duplicate_group"]:
            rows.append({"relative_path": item["relative_path"], "issue": "near_duplicate", "severity": "review", "detail": item["near_duplicate_group"]})
    return rows


def plot_distribution(frame: pd.DataFrame, output: Path) -> bool:
    labeled = frame[frame["label"].astype(bool)]
    if labeled.empty:
        return False
    import matplotlib.pyplot as plt

    counts = labeled["label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.4)))
    counts.plot.barh(ax=ax, color="#2b7a9b")
    ax.set_title("Discovered folder categories (not necessarily medically verified)")
    ax.set_xlabel("Images")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def write_report(frame: pd.DataFrame, evidence: AnnotationEvidence, scenario: str, reason: str, root: Path, output: Path, labels_trusted: bool) -> None:
    readable = frame[frame["readable"]]
    formats = readable["format"].replace("", "unknown").value_counts().to_dict()
    resolutions = readable.groupby(["width", "height"]).size().sort_values(ascending=False).head(15)
    exact_groups = frame.loc[frame["duplicate_group"].astype(bool), "duplicate_group"].nunique()
    near_groups = frame.loc[frame["near_duplicate_group"].astype(bool), "near_duplicate_group"].nunique()
    grayscale_rate = readable["grayscale_like"].mean() if len(readable) else 0.0
    modality_note = (
        "The collection is predominantly grayscale-like and may be radiographic, but pixel appearance does not prove X-ray modality."
        if grayscale_rate >= 0.95 else
        "The collection includes a material fraction of color-like images and may contain mixed modalities, screenshots, or rendered assets; source metadata review is required."
    )
    folder_depths = Counter(len(Path(item).parts) - 1 for item in frame["relative_path"])
    label_counts = frame.loc[frame["label"].astype(bool), "label"].value_counts()
    lines = [
        "# OsteoInsight dataset report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Authorized source: `{root.resolve()}`",
        "",
        "## Scientific status",
        "",
        f"- Selected workflow scenario: **{scenario}**",
        f"- Reason: {reason}",
        f"- Discovered labels explicitly marked trusted: **{labels_trusted}**",
        "- No filename or folder category is treated as a medical diagnosis by this audit.",
        "- X-ray modality cannot be medically confirmed from pixel appearance alone.",
        "",
        "## Inventory",
        "",
        f"- Image files: {len(frame):,}",
        f"- Readable: {len(readable):,}",
        f"- Unreadable/corrupt: {len(frame) - len(readable):,}",
        f"- Formats: {formats}",
        f"- Folder depths: {dict(sorted(folder_depths.items()))}",
        f"- Grayscale-like images: {grayscale_rate:.1%} (appearance heuristic only)",
        f"- Modality review: {modality_note}",
        f"- Exact duplicate groups: {exact_groups:,}",
        f"- Near-duplicate groups: {near_groups:,}",
        f"- Roboflow filename/path hints: {int(frame['roboflow_hint'].sum()):,}",
        f"- Rows with patient-like IDs: {int(frame['patient_id'].astype(bool).sum()):,}",
        f"- Rows with study-like IDs: {int(frame['study_id'].astype(bool).sum()):,}",
        "",
        "## Common resolutions",
        "",
        "| Width | Height | Count |",
        "|---:|---:|---:|",
    ]
    lines.extend(f"| {int(width)} | {int(height)} | {int(count)} |" for (width, height), count in resolutions.items())
    lines.extend([
        "",
        "## Annotation evidence",
        "",
        f"- `data.yaml`: {evidence.yaml_files or 'none'}",
        f"- YOLO label files: {len(evidence.yolo_label_files):,}",
        f"- COCO annotation files: {evidence.coco_files or 'none'}",
        f"- CSV metadata files: {evidence.csv_files or 'none'}",
        f"- Declared class names: {evidence.class_names or 'none'}",
        f"- Detection annotation counts by class: {evidence.annotation_class_counts or 'none'}",
        f"- CSV columns: {evidence.csv_columns or 'none'}",
        f"- Candidate CSV label counts (unverified): {evidence.csv_candidate_label_counts or 'none'}",
        f"- Parser notes: {evidence.notes or 'none'}",
        "",
        "## Discovered category counts",
        "",
    ])
    if label_counts.empty:
        lines.append("No folder categories were detected.")
    else:
        lines.extend(["| Category | Images |", "|---|---:|"])
        lines.extend(f"| {label} | {count} |" for label, count in label_counts.items())
        if not labels_trusted:
            lines.append("\nThese are organizational categories only and are not verified medical ground truth.")
    lines.extend([
        "",
        "## Leakage and validity notes",
        "",
        "- Exact and near-duplicate groups must remain within a single split.",
        "- Prefer patient/study-level grouping when identifiers are reliable.",
        "- Existing splits are preserved as metadata but must be checked for duplicate leakage.",
        "- No claim of patient-level generalization is valid without independent patient/study grouping.",
        "- Image licensing and redistribution permissions were not established by this audit.",
        "",
        "## Required next evidence",
        "",
        "For supervised medical classification, document the annotation source, label definitions, annotator expertise, patient/study identifiers, and a leakage-safe held-out test cohort. For detection, also validate bounding-box coordinates against image dimensions and visually inspect representative annotations.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {root}")
    if output == root or root in output.parents:
        raise SystemExit("Refusing to write reports inside the source dataset. Choose a separate --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No supported image files found under: {root}")
    rows = [inspect_image(path, root) for path in images]
    frame = assign_duplicate_groups(pd.DataFrame(rows), args.near_duplicate_distance)
    evidence = discover_annotations(root, {path.stem for path in images})
    frame["label_verified"] = bool(args.labels_trusted)
    scenario, reason = choose_scenario(frame, evidence, args.labels_trusted)
    frame.to_csv(output / "dataset_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    quality = pd.DataFrame(quality_rows(frame), columns=["relative_path", "issue", "severity", "detail"])
    quality.to_csv(output / "data_quality_report.csv", index=False)
    plot_distribution(frame, output / "class_distribution.png")
    write_report(frame, evidence, scenario, reason, root, output / "dataset_report.md", args.labels_trusted)
    summary = {
        "dataset_root": str(root), "image_count": len(frame), "readable_count": int(frame["readable"].sum()),
        "scenario": scenario, "scenario_reason": reason, "labels_trusted": bool(args.labels_trusted),
        "artifacts": [str(output / name) for name in ["dataset_report.md", "dataset_manifest.csv", "data_quality_report.csv"]],
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
