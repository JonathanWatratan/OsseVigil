#!/usr/bin/env python3
"""Create leakage-aware train/validation/test assignments in a copied manifest.

The source images and original manifest are never modified. This command refuses to
create supervised splits unless every used label is explicitly marked verified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from osteoinsight.data import load_roboflow_csv_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--roboflow-root", type=Path, help="Read source train/valid/test CSV annotations directly")
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_manifest_split.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def grouping_key(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    def first_value(row):
        for prefix, column in [
            ("patient", "patient_id"), ("study", "study_id"),
            ("near", "near_duplicate_group"), ("duplicate", "duplicate_group"),
        ]:
            value = str(row.get(column, "")).strip()
            if value and value.lower() != "nan":
                return f"{prefix}:{value}"
        return f"image:{row['sha256']}"
    groups = frame.apply(first_value, axis=1)
    method = "patient/study IDs where present, then duplicate groups, then unique image hash"
    return groups, method


def group_split(indices, labels, groups, holdout_size: float, seed: int):
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=32, test_size=holdout_size, random_state=seed)
    best = None
    all_classes = set(labels)
    for left, right in splitter.split(indices, labels, groups):
        score = len(all_classes - set(labels.iloc[left])) + len(all_classes - set(labels.iloc[right]))
        candidate = (score, left, right)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if score == 0:
            break
    if best is None or best[0] > 0:
        raise ValueError("Could not construct splits containing every class while preserving groups.")
    return best[1], best[2]


def main():
    args = parse_args()
    if bool(args.manifest) == bool(args.roboflow_root):
        raise SystemExit("Provide exactly one of: manifest path or --roboflow-root.")
    if args.roboflow_root:
        frame = load_roboflow_csv_manifest(args.roboflow_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame["split_method"] = "preserved source-provided Roboflow train/valid/test split"
        frame.to_csv(args.output, index=False)
        print(frame.groupby(["original_split", "label"]).size().unstack(fill_value=0))
        print(f"Wrote {args.output}; original Roboflow splits preserved. Run the duplicate audit separately.")
        return
    frame = pd.read_csv(args.manifest, keep_default_na=False)
    usable = frame[(frame["readable"] == True) & frame["label"].astype(bool)].copy()  # noqa: E712
    if usable.empty:
        raise SystemExit("No readable labeled images are available.")
    verified = usable["label_verified"].astype(str).str.lower().isin({"true", "1", "yes"})
    if not verified.all():
        raise SystemExit("Refusing supervised split: one or more labels are not explicitly source-verified.")
    groups, method = grouping_key(usable)
    usable["split_group"] = groups
    normalized_splits = usable["split"].astype(str).str.lower()
    preserved = normalized_splits.isin({"train", "val", "test"}).all() and set(normalized_splits) == {"train", "val", "test"}
    if preserved:
        leakage = usable.groupby("split_group")["split"].nunique()
        if (leakage > 1).any():
            bad = leakage[leakage > 1].index[:10].tolist()
            raise SystemExit(f"Existing splits leak groups across partitions; examples: {bad}")
        split_method = "preserved source-provided train/val/test split after group leakage check"
    else:
        usable["split"] = ""
        indices = pd.Series(range(len(usable)))
        train_val, test = group_split(indices, usable["label"], groups, args.test_size, args.seed)
        relative_val = args.val_size / (1.0 - args.test_size)
        tv = usable.iloc[train_val]
        train_rel, val_rel = group_split(
            pd.Series(range(len(tv))), tv["label"].reset_index(drop=True),
            tv["split_group"].reset_index(drop=True), relative_val, args.seed + 1,
        )
        usable.iloc[test, usable.columns.get_loc("split")] = "test"
        usable.iloc[train_val[train_rel], usable.columns.get_loc("split")] = "train"
        usable.iloc[train_val[val_rel], usable.columns.get_loc("split")] = "val"
        split_method = f"generated with {method}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    usable["split_method"] = split_method
    usable.to_csv(args.output, index=False)
    print(usable.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"Wrote {args.output} using {split_method}.")


if __name__ == "__main__":
    main()
