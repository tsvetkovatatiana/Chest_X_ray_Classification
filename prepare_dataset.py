"""
Prepare the Chest X-Ray dataset (Muhammad Rehan, Kaggle) for binary classification.

Steps:
  1. Keep only the Normal and Pneumonia classes (Tuberculosis is dropped).
  2. Combine the train and val folders into one training pool.
  3. Write CSV files with image paths and labels:
       - metadata.csv  -> train + val images (for training / cross-validation)
       - test_data.csv      -> test images (for final evaluation)
       - fold_{k}_train.csv / fold_{k}_val.csv -> optional stratified K-fold splits

Expected layout after unzipping the Kaggle download (folder names are matched
case-insensitively, so "Normal", "NORMAL" and "normal" all work):

    <data_root>/
        train/  normal/  pneumonia/  tuberculosis/
        val/    normal/  pneumonia/  tuberculosis/
        test/   normal/  pneumonia/  tuberculosis/

Usage:
    python prepare_dataset.py --data_root data/chest_xray --out_dir data/CSVs
    python prepare_dataset.py --data_root data/chest_xray --out_dir data/CSVs --folds 0   # no K-fold files
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

KEEP_CLASSES = {"normal", "pneumonia"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TRAIN_SPLITS = {"train", "val", "valid", "validation"}
TEST_SPLITS = {"test"}


def find_split_dirs(data_root: Path) -> dict:
    """Return {split_name: path} for the split folders found under data_root."""
    splits = {}
    for entry in data_root.iterdir():
        if entry.is_dir():
            name = entry.name.lower()
            if name in TRAIN_SPLITS | TEST_SPLITS:
                splits[name] = entry
    return splits


def collect_images(split_dir: Path, split_name: str) -> list:
    """Collect image rows for the kept classes inside one split folder."""
    rows = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name.strip().lower()
        if label not in KEEP_CLASSES:
            print(f"  skipping class folder: {class_dir}")
            continue

        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "Name": img_path.name,
                    "Path": img_path.as_posix(),
                    "Label": label,
                    "Dataset_type": "TEST" if split_name in TEST_SPLITS else "TRAIN",
                    "Source_split": split_name,
                })
    return rows


def is_readable(path: str) -> bool:
    """True if OpenCV can decode the image (same reader as dataset.py)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)  # works with any Windows path
        if data.size == 0:
            return False
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) is not None
    except Exception:
        return False


def drop_unreadable(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Remove images OpenCV cannot open and report them."""
    if df.empty:
        return df
    ok = df["Path"].apply(is_readable)
    bad = df[~ok]
    if len(bad):
        print(f"{name}: removed {len(bad)} unreadable images:")
        for p in bad["Path"].head(20):
            print(f"  {p}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
    return df[ok].reset_index(drop=True)


def print_summary(name: str, df: pd.DataFrame) -> None:
    counts = df["Label"].value_counts().to_dict()
    print(f"{name}: {len(df)} images {counts}")


def main():
    parser = argparse.ArgumentParser(description="Build CSVs for Normal vs Pneumonia classification.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder that contains the train/, val/ and test/ folders.")
    parser.add_argument("--out_dir", type=str, default="data/CSVs",
                        help="Where to save the CSV files.")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of stratified folds to create (0 to skip).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    splits = find_split_dirs(data_root)
    if not splits:
        raise FileNotFoundError(f"No train/val/test folders found in {data_root}")
    print("Found splits:", ", ".join(sorted(splits)))

    train_rows, test_rows = [], []
    for split_name, split_dir in splits.items():
        rows = collect_images(split_dir, split_name)
        (test_rows if split_name in TEST_SPLITS else train_rows).extend(rows)

    metadata = pd.DataFrame(train_rows)
    test = pd.DataFrame(test_rows)

    if metadata.empty:
        raise RuntimeError("No Normal/Pneumonia images found in the train/val folders.")

    # Remove exact duplicate paths only. Train and val can contain different images
    # with the same file name (e.g. normal-1.jpg), so never deduplicate by name.
    before = len(metadata)
    metadata = metadata.drop_duplicates(subset=["Path"]).reset_index(drop=True)
    if len(metadata) != before:
        print(f"Removed {before - len(metadata)} duplicate images from train/val.")

    print("Checking that every image can be opened (this takes a minute)...")
    metadata = drop_unreadable(metadata, "train/val")
    test = drop_unreadable(test, "test")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(out_dir / "metadata.csv", index=False)
    print_summary("metadata.csv (train + val)", metadata)

    if not test.empty:
        test.to_csv(out_dir / "test_data.csv", index=False)
        print_summary("test_data.csv", test)
    else:
        print("Warning: no test images found, test_data.csv was not created.")

    if args.folds and args.folds > 1:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(metadata, metadata["Label"]), 1):
            metadata.iloc[train_idx].to_csv(out_dir / f"fold_{fold}_train.csv", index=False)
            metadata.iloc[val_idx].to_csv(out_dir / f"fold_{fold}_val.csv", index=False)
        print(f"Created {args.folds} stratified folds in {out_dir}")

    print(f"Done. CSV files saved to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()