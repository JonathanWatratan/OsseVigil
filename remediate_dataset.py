#!/usr/bin/env python3
"""Build a non-destructive, group-aware exploratory dataset manifest.

The source Roboflow export is read only. Confirmed duplicate components are
deduplicated to one deterministic representative. Ambiguous visual matches and
conflicting heuristic source identifiers are quarantined rather than relabeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from analyze_leakage import normalized_correlation


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_manifest", type=Path)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confirmed-correlation", type=float, default=0.995)
    parser.add_argument("--review-correlation", type=float, default=0.98)
    parser.add_argument("--max-phash-distance", type=int, default=6)
    return parser.parse_args()


def thumbnail(path: str) -> np.ndarray:
    pixels = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.resize(pixels, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def candidate_pairs(values: list[int], max_distance: int):
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for index, value in enumerate(values):
        candidates: set[int] = set()
        for band in range(8):
            key = band, (value >> (band * 8)) & 0xFF
            candidates.update(buckets[key])
            buckets[key].append(index)
        for other in candidates:
            pair = (other, index)
            if pair in seen:
                continue
            seen.add(pair)
            distance = (value ^ values[other]).bit_count()
            if distance <= max_distance:
                yield other, index, distance


def assign_group_splits(frame: pd.DataFrame, seed: int) -> pd.Series:
    """Assign whole source groups to approximately 80/10/10 stratified splits."""
    work = frame.copy()
    work["stratum"] = (
        work["label"].astype(str) + "|" + work["source_family_hint"].astype(str)
        + "|" + work["anatomy_hint_unverified"].astype(str)
    )
    group_rows = []
    for source_group, group in work.groupby("source_group", sort=True):
        group_rows.append({
            "source_group": source_group,
            "stratum": group["stratum"].mode().iat[0],
            "count": len(group),
            "tie": hashlib.sha256(f"{seed}|{source_group}".encode()).hexdigest(),
        })
    groups = pd.DataFrame(group_rows)
    assignment: dict[str, str] = {}
    ratios = {"train": 0.80, "val": 0.10, "test": 0.10}
    for _, subset in groups.groupby("stratum", sort=True):
        subset = subset.sort_values(["count", "tie"], ascending=[False, True])
        total = int(subset["count"].sum())
        targets = {name: total * ratio for name, ratio in ratios.items()}
        current = {name: 0 for name in ratios}
        for row in subset.itertuples(index=False):
            # Largest proportional deficit wins; stable order breaks ties.
            chosen = max(ratios, key=lambda name: ((targets[name] - current[name]) / max(targets[name], 1), ratios[name]))
            assignment[str(row.source_group)] = chosen
            current[chosen] += int(row.count)
    return work["source_group"].map(assignment)


def table_dict(frame: pd.DataFrame, *columns: str) -> dict:
    return frame.groupby(list(columns)).size().unstack(fill_value=0).to_dict("index")


def save_pair_montage(pairs: pd.DataFrame, output: Path, title: str, maximum: int = 12) -> None:
    if pairs.empty:
        return
    chosen = pairs.iloc[np.linspace(0, len(pairs) - 1, min(maximum, len(pairs)), dtype=int)]
    rows = []
    for pair in chosen.itertuples(index=False):
        panels = []
        for path, label in [(pair.left_path, pair.left_label), (pair.right_path, pair.right_label)]:
            with Image.open(path) as image:
                panel = ImageOps.fit(image.convert("L"), (240, 180)).convert("RGB")
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 152, 240, 180), fill=(7, 26, 45))
            draw.text((6, 158), f"dataset label: {label}", fill="white")
            panels.append(panel)
        row = Image.new("RGB", (480, 210), "white")
        row.paste(panels[0], (0, 30)); row.paste(panels[1], (240, 30))
        draw = ImageDraw.Draw(row)
        draw.text((6, 6), f"{title}; pHash={pair.phash_distance}; corr={pair.pixel_correlation_128:.5f}", fill=(7, 26, 45))
        rows.append(row)
    montage = Image.new("RGB", (480, 210 * len(rows)), "white")
    for index, row in enumerate(rows):
        montage.paste(row, (0, index * 210))
    montage.save(output, quality=92)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.audit_manifest, keep_default_na=False).reset_index(drop=True)
    identity = pd.read_csv(args.identity_manifest, keep_default_na=False)
    phash_by_path = identity.set_index("absolute_path")["phash"].to_dict()
    if set(frame["absolute_path"]) != set(identity["absolute_path"]):
        raise SystemExit("Audit and identity manifests do not describe the same images.")
    frame["phash"] = frame["absolute_path"].map(phash_by_path)
    frame["row_id"] = np.arange(len(frame))
    union = UnionFind(len(frame))

    exact_groups = []
    exact_conflict_roots: set[int] = set()
    for digest, group in frame.groupby("sha256"):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        for other in indices[1:]:
            union.union(indices[0], other)
        conflict = group["label"].nunique() > 1
        exact_groups.append({
            "sha256": digest, "image_count": len(group),
            "labels": "/".join(sorted(group["label"].unique())),
            "original_splits": "/".join(sorted(group["original_split"].unique())),
            "label_conflict": conflict,
            "paths": " | ".join(group["relative_path"]),
        })
        if conflict:
            exact_conflict_roots.add(indices[0])

    phashes = [int(value, 16) for value in frame["phash"]]
    thumbs = [thumbnail(path) for path in frame["absolute_path"]]
    confirmed_rows, review_rows = [], []
    review_indices: set[int] = set()
    for left, right, distance in candidate_pairs(phashes, args.max_phash_distance):
        if frame.at[left, "sha256"] == frame.at[right, "sha256"]:
            continue
        correlation = normalized_correlation(thumbs[left], thumbs[right])
        if correlation < args.review_correlation:
            continue
        row = {
            "left_row_id": left, "right_row_id": right,
            "left_path": frame.at[left, "absolute_path"], "right_path": frame.at[right, "absolute_path"],
            "left_label": frame.at[left, "label"], "right_label": frame.at[right, "label"],
            "phash_distance": distance, "pixel_correlation_128": correlation,
            "label_conflict": frame.at[left, "label"] != frame.at[right, "label"],
        }
        if distance <= 4 and correlation >= args.confirmed_correlation:
            confirmed_rows.append(row)
            union.union(left, right)
        else:
            review_rows.append(row)
            review_indices.update([left, right])

    frame["duplicate_root"] = [union.find(index) for index in frame.index]
    component_ids = {root: f"confirmed_{number:05d}" for number, root in enumerate(sorted(frame["duplicate_root"].unique()), 1)}
    frame["confirmed_duplicate_component"] = frame["duplicate_root"].map(component_ids)
    quarantine_reason: dict[int, set[str]] = defaultdict(set)

    # Any uncertainty touching a confirmed component quarantines the entire component.
    uncertain_roots = {union.find(index) for index in review_indices}
    for index in frame.index:
        if union.find(index) in uncertain_roots:
            quarantine_reason[index].add("uncertain_perceptual_match")

    for _, group in frame.groupby("duplicate_root"):
        if group["label"].nunique() > 1:
            for index in group.index:
                quarantine_reason[index].add("conflicting_labels_in_confirmed_duplicate_component")

    source_conflict_groups = []
    for source_id, group in frame[frame["source_id_heuristic"].astype(bool)].groupby("source_id_heuristic"):
        if group["label"].nunique() > 1:
            source_conflict_groups.append({
                "source_id_heuristic": source_id, "image_count": len(group),
                "labels": "/".join(sorted(group["label"].unique())),
                "paths": " | ".join(group["relative_path"]),
            })
            for index in group.index:
                quarantine_reason[index].add("conflicting_labels_in_heuristic_source_group")

    frame["quarantine_reason"] = [";".join(sorted(quarantine_reason[index])) for index in frame.index]
    frame["cleaning_status"] = "candidate"
    frame.loc[frame["quarantine_reason"].astype(bool), "cleaning_status"] = "quarantined"

    # Keep one reproducible, highest-byte-count representative per confirmed component.
    candidates = frame[frame["cleaning_status"] == "candidate"]
    for _, group in candidates.groupby("duplicate_root"):
        if len(group) < 2:
            continue
        ordered = group.sort_values(["file_size_bytes", "relative_path"], ascending=[False, True])
        frame.loc[ordered.index[1:], "cleaning_status"] = "excluded_redundant_confirmed_duplicate"
    retained = frame[frame["cleaning_status"] == "candidate"].copy()
    retained["source_group"] = retained["source_id_heuristic"].astype(str)
    empty_source = ~retained["source_group"].astype(bool)
    retained.loc[empty_source, "source_group"] = retained.loc[empty_source, "confirmed_duplicate_component"]
    retained["split"] = assign_group_splits(retained, args.seed)

    # Assert the automated boundary properties before writing a training manifest.
    exact_cross = retained.groupby("sha256")["split"].nunique().gt(1).sum()
    source_cross = retained.groupby("source_group")["split"].nunique().gt(1).sum()
    component_cross = retained.groupby("confirmed_duplicate_component")["split"].nunique().gt(1).sum()
    if exact_cross or source_cross or component_cross:
        raise SystemExit(f"Split invariant failed: exact={exact_cross} source={source_cross} component={component_cross}")

    columns = list(frame.columns)
    retained.to_csv(output / "cleaned_manifest.csv", index=False)
    for split in ("train", "val", "test"):
        retained[retained["split"] == split].to_csv(output / f"{split}_manifest.csv", index=False)
    frame[frame["cleaning_status"] == "quarantined"].to_csv(output / "quarantine_manifest.csv", index=False)
    frame[frame["cleaning_status"] == "excluded_redundant_confirmed_duplicate"].to_csv(output / "excluded_confirmed_duplicates.csv", index=False)
    pd.DataFrame(exact_groups).to_csv(output / "exact_duplicate_groups.csv", index=False)
    confirmed_frame = pd.DataFrame(confirmed_rows)
    review_frame = pd.DataFrame(review_rows)
    confirmed_frame.to_csv(output / "confirmed_near_duplicate_pairs.csv", index=False)
    review_frame.to_csv(output / "uncertain_perceptual_pairs.csv", index=False)
    save_pair_montage(confirmed_frame, output / "confirmed_pair_review_samples.jpg", "confirmed automated duplicate")
    save_pair_montage(review_frame, output / "uncertain_pair_review_samples.jpg", "quarantined for review")
    pd.DataFrame(source_conflict_groups).to_csv(output / "source_label_conflicts.csv", index=False)
    pd.crosstab(retained["source_family_hint"], retained["label"]).to_csv(output / "source_family_by_label.csv")
    pd.crosstab(retained["anatomy_hint_unverified"], retained["label"]).to_csv(output / "anatomy_hint_by_label.csv")

    duplicate_components = frame.groupby("duplicate_root").size()
    conflicting_components = frame.groupby("duplicate_root")["label"].nunique().gt(1)
    summary = {
        "seed": args.seed,
        "source_dataset_modified": False,
        "source_dataset": str(Path(frame["absolute_path"].iat[0]).parents[1]),
        "original_images": len(frame),
        "retained_images": len(retained),
        "confirmed_exact_duplicate_groups": len(exact_groups),
        "confirmed_exact_cross_split_groups": int(sum("/" in row["original_splits"] for row in exact_groups)),
        "confirmed_nonexact_near_duplicate_pairs": len(confirmed_rows),
        "confirmed_duplicate_components": int((duplicate_components > 1).sum()),
        "uncertain_perceptual_pairs": len(review_rows),
        "quarantined_images": int((frame["cleaning_status"] == "quarantined").sum()),
        "excluded_redundant_duplicate_images": int((frame["cleaning_status"] == "excluded_redundant_confirmed_duplicate").sum()),
        "unresolved_label_conflict_components": int(conflicting_components.sum()),
        "heuristic_source_label_conflict_groups": len(source_conflict_groups),
        "class_distribution": table_dict(retained, "split", "label"),
        "known_cross_split_exact_groups_after": int(exact_cross),
        "known_cross_split_duplicate_components_after": int(component_cross),
        "known_cross_split_source_groups_after": int(source_cross),
        "evaluation_scope": "exploratory source-grouped image classification; not patient-independent or clinical",
        "remaining_limitations": [
            "No patient or study identifiers are available, so patient independence cannot be established.",
            "Dataset-provided cancer/normal labels lack documented independent clinical verification.",
            "Source family and anatomical region remain strongly associated with class labels.",
            "Heuristic source IDs and automated perceptual thresholds can both make grouping errors.",
            "Deduplication does not solve source, anatomy, acquisition, or spectrum confounding.",
        ],
        "exploratory_training_gate_pass": True,
        "exploratory_training_gate_reason": "Known exact/confirmed perceptual copies were deduplicated; uncertain matches and label-conflicting heuristic source groups were quarantined; source groups do not cross splits.",
    }
    (output / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Cleaned exploratory dataset", "",
        f"- Original rows: {len(frame):,}", f"- Retained representatives: {len(retained):,}",
        f"- Quarantined images: {summary['quarantined_images']:,}",
        f"- Redundant confirmed copies excluded: {summary['excluded_redundant_duplicate_images']:,}",
        f"- Exact duplicate groups: {len(exact_groups):,}",
        f"- Confirmed non-exact near-duplicate pairs: {len(confirmed_rows):,}",
        f"- Uncertain perceptual pairs: {len(review_rows):,}",
        f"- Unresolved confirmed-component label conflicts: {summary['unresolved_label_conflict_components']:,}",
        f"- Heuristic source-ID label-conflict groups quarantined: {len(source_conflict_groups):,}", "",
        "The split is reproducible (seed 42), grouped by heuristic source ID, and has no known exact, confirmed-duplicate-component, or heuristic-source group crossing train/validation/test.",
        "It is exploratory only: patient independence is unknown, labels lack documented clinical verification, and anatomy/source confounding remains. Deduplication does not solve those limitations.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
