#!/usr/bin/env python3
"""Create a derived manifest that removes known cross-split image leakage.

No source image, CSV, or original split assignment is modified. Test images have
highest retention priority, followed by validation, then training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIORITY = {"train": 0, "valid": 1, "validation": 1, "val": 1, "test": 2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--perceptual-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/leakage_analysis"))
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, keep_default_na=False)
    frame["eligible_after_known_leakage_filter"] = True
    frame["exclusion_reason"] = ""
    path_to_index = {str(path): index for index, path in frame["absolute_path"].items()}

    # Definitive byte-identical groups: retain only members in the highest-priority
    # split represented in the group.
    for _, group in frame.groupby("sha256"):
        if group["original_split"].nunique() < 2:
            continue
        keep_priority = max(PRIORITY[str(split)] for split in group["original_split"])
        remove = group[group["original_split"].map(PRIORITY) < keep_priority].index
        frame.loc[remove, "eligible_after_known_leakage_filter"] = False
        frame.loc[remove, "exclusion_reason"] = "exact_duplicate_of_higher_priority_split"

    pairs = pd.read_csv(args.perceptual_pairs)
    high = pairs[pairs["confidence"] == "high"]
    for _, pair in high.iterrows():
        left_index = path_to_index.get(str(pair["left_path"]))
        right_index = path_to_index.get(str(pair["right_path"]))
        if left_index is None or right_index is None:
            continue
        left_priority = PRIORITY[str(pair["left_split"])]
        right_priority = PRIORITY[str(pair["right_split"])]
        if left_priority == right_priority:
            continue
        remove_index = left_index if left_priority < right_priority else right_index
        frame.at[remove_index, "eligible_after_known_leakage_filter"] = False
        previous = str(frame.at[remove_index, "exclusion_reason"])
        reason = "high_confidence_perceptual_copy_of_higher_priority_split"
        frame.at[remove_index, "exclusion_reason"] = f"{previous};{reason}".strip(";")

    eligible = frame[frame["eligible_after_known_leakage_filter"]].copy()
    excluded = frame[~frame["eligible_after_known_leakage_filter"]].copy()
    frame.to_csv(output / "manifest_with_leakage_decisions.csv", index=False)
    eligible.to_csv(output / "dataset_manifest_leakage_reduced.csv", index=False)
    excluded.to_csv(output / "excluded_cross_split_copies.csv", index=False)

    # Confirm known exact leakage is removed.
    residual_exact = eligible.groupby("sha256")["original_split"].nunique()
    residual_exact_count = int((residual_exact > 1).sum())
    counts_before = frame.groupby(["original_split", "label"]).size().unstack(fill_value=0).to_dict("index")
    counts_after = eligible.groupby(["original_split", "label"]).size().unstack(fill_value=0).to_dict("index")
    summary = {
        "policy": "test > validation > train; remove lower-priority exact or high-confidence perceptual copies",
        "source_files_modified": False,
        "original_rows": len(frame), "eligible_rows": len(eligible), "excluded_rows": len(excluded),
        "counts_before": counts_before, "counts_after": counts_after,
        "residual_exact_cross_split_groups": residual_exact_count,
        "remaining_limitations": [
            "patient/study independence cannot be established",
            "source family and anatomical region are strongly confounded with labels",
            "annotation provenance lacks clinical verification details",
            "perceptual review thresholds cannot prove patient identity",
        ],
        "training_gate_pass": False,
        "training_gate_reason": "Known image-copy leakage is reduced, but source/anatomy confounding and patient independence remain unresolved.",
    }
    (output / "leakage_reduced_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
