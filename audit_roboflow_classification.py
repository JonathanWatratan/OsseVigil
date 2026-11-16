#!/usr/bin/env python3
"""Audit a Roboflow multi-label classification CSV export without changing it.

Expected layout:
    dataset/train/_classes.csv
    dataset/valid/_classes.csv
    dataset/test/_classes.csv

Each CSV must contain a filename column and one binary indicator column per class.
The source split is preserved exactly in ``original_split`` and normalized to
train/val/test only for the training code.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from audit_dataset import IMAGE_EXTENSIONS, assign_duplicate_groups, inspect_image

EXPECTED_SPLIT_COUNTS = {"train": 7057, "valid": 882, "test": 872}
SPLIT_NORMALIZATION = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}
FILENAME_COLUMNS = {"filename", "file", "image", "image_name", "filepath", "path"}
POSITIVE_VALUES = {"1", "1.0", "true", "yes", "y"}
NEGATIVE_VALUES = {"0", "0.0", "false", "no", "n", "", "nan"}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Roboflow one-hot classification CSV annotations")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/roboflow_audit"))
    parser.add_argument("--expected-classes", nargs="+", default=["normal", "cancer"])
    parser.add_argument("--near-duplicate-distance", type=int, default=5)
    return parser.parse_args()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def binary_value(value) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in POSITIVE_VALUES:
        return True
    if normalized in NEGATIVE_VALUES:
        return False
    return None


def source_identifier(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"\.rf\.[a-f0-9-]+$", "", stem)
    stem = re.sub(r"_(?:jpg|jpeg|png)$", "", stem)
    # Disease words are removed only for the leakage identifier. They are never
    # interpreted as ground truth.
    stem = re.sub(r"(?:^|[_\- ])(?:normal|cancer)(?:$|[_\- ])", "_", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def anatomy_hint(filename: str) -> str:
    """Filename-only audit hint; never used as a training label or diagnosis."""
    text = filename.lower()
    vocabulary = [
        "skull", "spine", "cervical", "chest", "rib", "shoulder", "clavicle",
        "humerus", "elbow", "forearm", "radius", "ulna", "wrist", "hand",
        "pelvis", "hip", "femur", "knee", "tibia", "fibula", "ankle", "foot",
    ]
    matches = [token for token in vocabulary if re.search(rf"(?:^|[_\- ]){token}(?:$|[_\- .])", text)]
    return matches[0] if matches else "unspecified"


def source_family_hint(filename: str) -> str:
    text = filename.lower()
    if text.startswith("bone-cancer"):
        return "bone-cancer"
    if text.startswith("image-no"):
        return "image-no"
    if re.match(r"img0+", text):
        return "IMG-numbered"
    if any(token in text for token in ["osteosarcoma", "ewing", "metastasis"]):
        return "pathology-named"
    if text.startswith("picture"):
        return "Picture-numbered"
    if re.match(r"-?\d+[_\- ]", text):
        return "numeric"
    return "other"


def locate_split_csvs(root: Path) -> dict[str, Path]:
    found = {}
    for original_split in ("train", "valid", "validation", "val", "test"):
        folder = root / original_split
        if not folder.is_dir():
            continue
        candidates = sorted(folder.glob("*.csv"), key=lambda path: (path.name != "_classes.csv", path.name))
        for path in candidates:
            try:
                columns = {str(column).strip().lower() for column in pd.read_csv(path, nrows=2).columns}
            except Exception:
                continue
            if columns & FILENAME_COLUMNS:
                found[original_split] = path
                break
    return found


def pixel_statistics(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        border = np.concatenate([pixels[:8].ravel(), pixels[-8:].ravel(), pixels[:, :8].ravel(), pixels[:, -8:].ravel()])
        center = pixels[32:96, 32:96]
        return {
            "mean_intensity": float(pixels.mean()), "intensity_std": float(pixels.std()),
            "border_mean": float(border.mean()), "center_mean": float(center.mean()),
            "border_center_delta": float(border.mean() - center.mean()),
        }
    except Exception:
        return {key: np.nan for key in ["mean_intensity", "intensity_std", "border_mean", "center_mean", "border_center_delta"]}


def parse_annotations(root: Path, csvs: dict[str, Path], expected_classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_rows, issue_rows = [], []
    discovered_classes: set[str] = set()
    csv_schema = {}
    for original_split, csv_path in csvs.items():
        raw = normalize_columns(pd.read_csv(csv_path, keep_default_na=False))
        lower_to_original = {column.lower(): column for column in raw.columns}
        filename_key = next((lower_to_original[name] for name in FILENAME_COLUMNS if name in lower_to_original), None)
        if not filename_key:
            raise ValueError(f"No filename column found in {csv_path}")
        class_columns = [column for column in raw.columns if column != filename_key]
        discovered_classes.update(class_columns)
        csv_schema[original_split] = {"path": str(csv_path), "rows": len(raw), "filename_column": filename_key, "class_columns": class_columns}
        duplicated_names = raw[filename_key].astype(str).duplicated(keep=False)
        for row_number, annotation in raw.iterrows():
            filename = str(annotation[filename_key]).strip()
            parsed = {column: binary_value(annotation[column]) for column in class_columns}
            invalid_values = [column for column, value in parsed.items() if value is None]
            active = [column for column, value in parsed.items() if value is True]
            image_path = root / original_split / filename
            one_hot_valid = len(active) == 1 and not invalid_values
            if invalid_values:
                issue_rows.append({"split": original_split, "filename": filename, "issue": "non_binary_class_value", "detail": ", ".join(invalid_values)})
            if len(active) == 0:
                issue_rows.append({"split": original_split, "filename": filename, "issue": "no_active_class", "detail": "all class indicators inactive"})
            elif len(active) > 1:
                issue_rows.append({"split": original_split, "filename": filename, "issue": "multiple_active_classes", "detail": ", ".join(active)})
            if bool(duplicated_names.iloc[row_number]):
                issue_rows.append({"split": original_split, "filename": filename, "issue": "duplicate_csv_filename", "detail": "filename occurs more than once in this CSV"})
            if not image_path.is_file():
                issue_rows.append({"split": original_split, "filename": filename, "issue": "missing_image", "detail": str(image_path)})
                continue
            inspected = inspect_image(image_path, root)
            inspected.update(pixel_statistics(image_path))
            inspected.update({
                "original_split": original_split,
                "split": SPLIT_NORMALIZATION[original_split],
                "label": active[0].strip().lower() if one_hot_valid else "",
                "label_source": "roboflow_classification_csv",
                "label_verified": bool(one_hot_valid),
                "label_scope": "source annotation; clinical truth not independently verified",
                "active_class_count": len(active),
                "class_value_valid": not invalid_values,
                "source_id_heuristic": source_identifier(filename),
                "anatomy_hint_unverified": anatomy_hint(filename),
                "source_family_hint": source_family_hint(filename),
                "filename_contains_normal": bool(re.search(r"(?i)(?:^|[_\- ])normal(?:$|[_\- .])", filename)),
                "filename_contains_cancer": bool(re.search(r"(?i)(?:^|[_\- ])cancer(?:$|[_\- .])", filename)),
            })
            manifest_rows.append(inspected)
    expected = {name.strip().lower() for name in expected_classes}
    discovered = {name.strip().lower() for name in discovered_classes}
    if discovered != expected:
        issue_rows.append({"split": "all", "filename": "", "issue": "unexpected_class_schema", "detail": f"expected={sorted(expected)} discovered={sorted(discovered)}"})
    return pd.DataFrame(manifest_rows), pd.DataFrame(issue_rows, columns=["split", "filename", "issue", "detail"]), csv_schema


def add_group_issues(frame: pd.DataFrame, issues: list[dict]) -> None:
    for column, issue_name in [
        ("sha256", "exact_duplicate"), ("near_duplicate_group", "near_duplicate"),
        ("source_id_heuristic", "source_id_collision"),
    ]:
        grouped = frame[frame[column].astype(bool)].groupby(column)
        for key, group in grouped:
            if len(group) < 2:
                continue
            splits = sorted(group["original_split"].unique())
            labels = sorted(group.loc[group["label"].astype(bool), "label"].unique())
            if len(splits) > 1:
                issues.append({"split": "/".join(splits), "filename": " | ".join(group["relative_path"].head(5)), "issue": f"cross_split_{issue_name}", "detail": str(key)})
            if len(labels) > 1:
                issues.append({"split": "/".join(splits), "filename": " | ".join(group["relative_path"].head(5)), "issue": f"conflicting_labels_{issue_name}", "detail": f"{key}: {labels}"})


def orphan_images(root: Path, frame: pd.DataFrame) -> list[str]:
    referenced = {str(Path(path).resolve()) for path in frame["absolute_path"]}
    orphans = []
    for split in ("train", "valid", "validation", "val", "test"):
        folder = root / split
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and str(path.resolve()) not in referenced:
                    orphans.append(str(path.relative_to(root)))
    return sorted(orphans)


def write_report(frame: pd.DataFrame, issues: pd.DataFrame, csv_schema: dict, root: Path, output: Path, orphans: list[str]) -> None:
    valid = frame[(frame["readable"] == True) & (frame["active_class_count"] == 1) & frame["class_value_valid"]]  # noqa: E712
    class_table = valid.groupby(["original_split", "label"]).size().unstack(fill_value=0)
    class_lines = ["| Original split | " + " | ".join(str(column) for column in class_table.columns) + " |", "|---|" + "---:|" * len(class_table.columns)]
    for split, row in class_table.iterrows():
        class_lines.append("| " + str(split) + " | " + " | ".join(str(int(row[column])) for column in class_table.columns) + " |")
    split_counts = frame.groupby("original_split").size().to_dict()
    expected_match = {split: split_counts.get(split) == count for split, count in EXPECTED_SPLIT_COUNTS.items()}
    cross_split_counts = issues[issues["issue"].str.startswith("cross_split_", na=False)]["issue"].value_counts().to_dict()
    conflict_counts = issues[issues["issue"].str.startswith("conflicting_labels_", na=False)]["issue"].value_counts().to_dict()
    structural_names = {"non_binary_class_value", "no_active_class", "multiple_active_classes", "duplicate_csv_filename", "missing_image", "image_missing_from_csv", "unreadable"}
    structural_issue_count = int(issues["issue"].isin(structural_names).sum())
    exact = frame[frame["duplicate_group"].astype(bool)]
    exact_group_count = int(exact["duplicate_group"].nunique())
    exact_image_count = int(len(exact))
    cross_exact = [(name, group) for name, group in exact.groupby("duplicate_group") if group["original_split"].nunique() > 1]
    cross_exact_images = sum(len(group) for _, group in cross_exact)
    training_gate_pass = structural_issue_count == 0 and len(cross_exact) == 0
    token_rows = frame[frame["filename_contains_normal"] | frame["filename_contains_cancer"]]
    filename_matches = (
        ((token_rows["filename_contains_normal"] & (token_rows["label"] == "normal")) |
         (token_rows["filename_contains_cancer"] & (token_rows["label"] == "cancer"))).mean()
        if len(token_rows) else np.nan
    )
    anatomy_table = frame.groupby(["anatomy_hint_unverified", "label"]).size().unstack(fill_value=0)
    source_table = frame.groupby(["source_family_hint", "label"]).size().unstack(fill_value=0)
    anatomy_summary = "; ".join(
        f"{name}: " + ", ".join(f"{label}={int(value)}" for label, value in row.items())
        for name, row in anatomy_table.iterrows() if name != "unspecified"
    ) or "no filename anatomy tokens"
    source_summary = "; ".join(
        f"{name}: " + ", ".join(f"{label}={int(value)}" for label, value in row.items())
        for name, row in source_table.iterrows()
    )
    lines = [
        "# Roboflow classification dataset audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Read-only dataset root: `{root}`",
        "",
        "## Annotation integrity",
        "",
        f"- CSV schemas: `{json.dumps(csv_schema, sort_keys=True)}`",
        f"- Image rows parsed: {len(frame):,}",
        f"- Readable images with exactly one active binary class: {len(valid):,}",
        f"- Split counts: {split_counts}",
        f"- Match published 7,057/882/872 counts: {expected_match}",
        f"- Structural annotation/image issues: {structural_issue_count:,}",
        f"- Total audit flags including leakage candidates: {len(issues):,}",
        f"- Images present but absent from CSV: {len(orphans):,}",
        "- CSV class names are treated as source annotations, not independently verified clinical diagnoses.",
        "",
        "## Class distribution",
        "",
        "\n".join(class_lines),
        "",
        "## Leakage and inconsistency checks",
        "",
        f"- Cross-split groups: {cross_split_counts or 'none detected'}",
        f"- Conflicting-label groups: {conflict_counts or 'none detected'}",
        f"- Exact duplicate groups overall: {exact_group_count:,} ({exact_image_count:,} images)",
        f"- Exact duplicate groups crossing published splits: {len(cross_exact):,} ({cross_exact_images:,} image rows)",
        "- Near-duplicate dHash groups are candidate-review signals and can include false positives; exact SHA-256 duplicates are definitive.",
        "- `source_id_heuristic` removes Roboflow hash suffixes and disease words; it is not a confirmed patient identifier.",
        "- Patient-level independence cannot be established unless original patient/study identifiers are supplied.",
        "",
        "## Shortcut risks",
        "",
        f"- Filenames containing `normal` or `cancer`: {len(token_rows):,}",
        f"- Agreement of those filename tokens with CSV labels: {filename_matches:.1%}" if len(token_rows) else "- No class-name filename tokens detected.",
        "- Filenames are excluded from model inputs, but source naming can reveal duplication or collection bias.",
        "- Low-level intensity, border, resolution, source-stem, and visual-cluster statistics are exported for shortcut review.",
        f"- Filename anatomy hints (audit metadata only): {anatomy_summary}",
        f"- Filename source-family hints: {source_summary}",
        "- Anatomical region is not asserted automatically. Representative cluster/error montages require human anatomical review.",
        "- Roboflow reports auto-orientation and stretch-resizing to 640×640; original pre-export aspect ratio may not be recoverable.",
        "",
        "## Scientific interpretation",
        "",
        f"- Supervised training gate: **{'PASS' if training_gate_pass else 'FAIL'}**",
        "- Gate reason: cross-split exact duplicates contaminate validation/test independence." if not training_gate_pass else "- Gate reason: no structural or exact cross-split leakage issue detected.",
        "A held-out result on these source-provided splits measures reproduction of the dataset categories only. It must not be described as clinical diagnostic accuracy or patient-level generalization. Cross-split duplicates, unknown patient identity, source artifacts, anatomical-region imbalance, and unverified diagnostic provenance can inflate performance.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_class_distribution(frame: pd.DataFrame, output: Path) -> None:
    table = frame[frame["label"].astype(bool)].groupby(["original_split", "label"]).size().unstack(fill_value=0)
    width, height = 1100, 650
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 25), "Roboflow source-annotation distribution", fill=(7, 26, 45))
    maximum = max(int(table.to_numpy().max()), 1)
    colors = [(239, 131, 84), (79, 157, 166), (95, 86, 166)]
    baseline, chart_height = 560, 430
    splits, classes = list(table.index), list(table.columns)
    group_width = 900 / max(len(splits), 1)
    bar_width = min(100, (group_width - 40) / max(len(classes), 1))
    for split_index, split in enumerate(splits):
        group_x = 100 + split_index * group_width
        for class_index, label in enumerate(classes):
            count = int(table.loc[split, label])
            bar_height = chart_height * count / maximum
            x0 = group_x + class_index * bar_width
            draw.rectangle((x0, baseline - bar_height, x0 + bar_width - 8, baseline), fill=colors[class_index % len(colors)])
            draw.text((x0, baseline - bar_height - 20), str(count), fill=(7, 26, 45))
        draw.text((group_x, baseline + 15), str(split), fill=(7, 26, 45))
    for class_index, label in enumerate(classes):
        x = 780 + class_index * 140
        draw.rectangle((x, 30, x + 18, 48), fill=colors[class_index % len(colors)])
        draw.text((x + 24, 30), str(label), fill=(7, 26, 45))
    canvas.save(output)


def main():
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset does not exist: {root}")
    if output == root or root in output.parents:
        raise SystemExit("Output directory must be outside the source dataset tree.")
    csvs = locate_split_csvs(root)
    normalized_found = {SPLIT_NORMALIZATION[key] for key in csvs}
    if normalized_found != {"train", "val", "test"}:
        raise SystemExit(f"Expected train/valid/test classification CSVs; found: {csvs}")
    output.mkdir(parents=True, exist_ok=True)
    frame, annotation_issues, csv_schema = parse_annotations(root, csvs, args.expected_classes)
    frame = assign_duplicate_groups(frame, args.near_duplicate_distance)
    group_issues: list[dict] = []
    add_group_issues(frame, group_issues)
    issues = pd.concat([annotation_issues, pd.DataFrame(group_issues)], ignore_index=True)
    orphans = orphan_images(root, frame)
    for path in orphans:
        issues.loc[len(issues)] = {"split": Path(path).parts[0], "filename": path, "issue": "image_missing_from_csv", "detail": "orphan image"}
    frame.to_csv(output / "dataset_manifest.csv", index=False)
    issues.to_csv(output / "label_and_leakage_issues.csv", index=False)
    low_level = frame.groupby(["original_split", "label"])[
        ["width", "height", "mean_intensity", "intensity_std", "border_mean", "center_mean", "border_center_delta"]
    ].agg(["count", "mean", "std", "median"])
    low_level.to_csv(output / "shortcut_feature_summary.csv")
    frame.groupby(["anatomy_hint_unverified", "label"]).size().unstack(fill_value=0).to_csv(output / "anatomy_hint_by_label.csv")
    frame.groupby(["source_family_hint", "label"]).size().unstack(fill_value=0).to_csv(output / "source_family_by_label.csv")
    plot_class_distribution(frame, output / "class_distribution.png")
    write_report(frame, issues, csv_schema, root, output / "dataset_report.md", orphans)
    summary = {
        "dataset_root": str(root), "rows": len(frame),
        "valid_one_hot_rows": int(((frame["active_class_count"] == 1) & frame["class_value_valid"]).sum()),
        "split_counts": frame.groupby("original_split").size().to_dict(),
        "class_counts": {
            str(split): {str(label): int(count) for label, count in group.groupby("label").size().items()}
            for split, group in frame.groupby("original_split")
        },
        "issue_counts": issues["issue"].value_counts().to_dict(),
        "training_gate_pass": not bool((issues["issue"] == "cross_split_exact_duplicate").any()) and not bool(issues["issue"].isin({"non_binary_class_value", "no_active_class", "multiple_active_classes", "missing_image", "unreadable"}).any()),
        "training_gate_reason": "Cross-split exact duplicates must be resolved before fitting under the project safety rule." if bool((issues["issue"] == "cross_split_exact_duplicate").any()) else "No blocking exact-duplicate or structural annotation issue detected.",
        "source_annotation_warning": "CSV labels are not independently verified clinical diagnoses.",
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
