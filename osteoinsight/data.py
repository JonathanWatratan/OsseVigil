from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestColumns:
    path: str = "absolute_path"
    label: str = "label"
    split: str = "split"
    patient: str = "patient_id"
    study: str = "study_id"
    duplicate_group: str = "duplicate_group"
    near_duplicate_group: str = "near_duplicate_group"


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["absolute_path"]) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, self.class_to_idx[str(row["label"])], str(row["absolute_path"])


def manifest_fingerprint(frame: pd.DataFrame) -> str:
    columns = [c for c in ["relative_path", "sha256", "label", "split"] if c in frame.columns]
    payload = frame[columns].fillna("").sort_values(columns[0]).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_transforms(image_size: int = 224):
    from torchvision import transforms

    # Conservative transforms: no flips or geometry changes by default because
    # laterality and anatomy can be medically meaningful.
    train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.08, contrast=0.08)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train, evaluate


def load_classification_splits(manifest_path: str | Path):
    frame = pd.read_csv(manifest_path)
    return classification_splits_from_frame(frame)


def classification_splits_from_frame(frame: pd.DataFrame):
    required = {"absolute_path", "label", "split", "readable"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame[(frame["readable"] == True) & frame["label"].notna()].copy()  # noqa: E712
    if frame.empty:
        raise ValueError("No readable, labeled images are present in the manifest.")
    classes = sorted(frame["label"].astype(str).unique())
    class_to_idx = {name: index for index, name in enumerate(classes)}
    splits = {name: group.copy() for name, group in frame.groupby("split")}
    for needed in ("train", "val", "test"):
        if needed not in splits or splits[needed].empty:
            raise ValueError(f"Manifest has no non-empty '{needed}' split.")
    return splits, class_to_idx


def load_roboflow_csv_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Load source-provided one-hot labels directly from Roboflow split CSVs.

    Filenames are used only to locate images; no label is inferred from a name.
    This lightweight loader does not replace the full duplicate/leakage audit.
    """
    root = Path(dataset_root).expanduser().resolve()
    split_map = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}
    filename_names = {"filename", "file", "image", "image_name", "filepath", "path"}
    positive = {"1", "1.0", "true", "yes", "y"}
    negative = {"0", "0.0", "false", "no", "n", "", "nan"}
    rows = []
    for original_split in ("train", "valid", "validation", "val", "test"):
        folder = root / original_split
        if not folder.is_dir():
            continue
        csv_candidates = sorted(folder.glob("*.csv"), key=lambda path: (path.name != "_classes.csv", path.name))
        if not csv_candidates:
            raise ValueError(f"No classification CSV found in {folder}")
        csv_path = csv_candidates[0]
        annotations = pd.read_csv(csv_path, keep_default_na=False)
        annotations.columns = [str(column).strip() for column in annotations.columns]
        filename_column = next((column for column in annotations.columns if column.lower() in filename_names), None)
        if filename_column is None:
            raise ValueError(f"No filename column in {csv_path}")
        class_columns = [column for column in annotations.columns if column != filename_column]
        for row_number, item in annotations.iterrows():
            active = []
            for column in class_columns:
                value = str(item[column]).strip().lower()
                if value in positive:
                    active.append(column.strip().lower())
                elif value not in negative:
                    raise ValueError(f"Non-binary value {item[column]!r} in {csv_path}:{row_number + 2} column {column}")
            if len(active) != 1:
                raise ValueError(f"Expected exactly one active class in {csv_path}:{row_number + 2}; found {active}")
            image_path = folder / str(item[filename_column]).strip()
            if not image_path.is_file():
                raise FileNotFoundError(f"CSV image is missing: {image_path}")
            rows.append({
                "relative_path": str(image_path.relative_to(root)), "absolute_path": str(image_path.resolve()),
                "readable": True, "label": active[0], "label_verified": True,
                "label_source": "roboflow_classification_csv",
                "label_scope": "source annotation; clinical truth not independently verified",
                "original_split": original_split, "split": split_map[original_split],
                "sha256": "not_computed_in_lightweight_loader",
            })
    frame = pd.DataFrame(rows)
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError(f"Expected original train/valid/test splits; found {sorted(frame['split'].unique())}")
    return frame
