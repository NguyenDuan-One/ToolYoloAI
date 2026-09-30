"""
dataset_splitter.py
-------------------
Shuffles and splits a dataset (images + labels) into train / val / test sets.

Expected input folder structure:
    source_folder/
        images/   (e.g., .jpg, .png)
        labels/   (e.g., .txt for YOLO)

Output folder structure:
    output_folder/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
"""

import shutil
import random
from pathlib import Path
from typing import Callable


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(
    source_folder: str,
    output_folder: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Shuffle and split a dataset into train, val, and test sets.

    The test set receives whatever remains after train + val are allocated,
    so train_ratio + val_ratio must be strictly less than 1.

    Args:
        source_folder:     Root folder containing 'images/' and 'labels/' subdirs.
        output_folder:     Root folder where
                           'images/{train,val,test}' and
                           'labels/{train,val,test}' are created.
        train_ratio:       Fraction of data for training  (e.g. 0.7).
        val_ratio:         Fraction of data for validation (e.g. 0.2).
        seed:              Random seed for reproducibility.
        progress_callback: Optional callback(current, total) for UI progress.

    Returns:
        dict with keys "train", "val", "test" and their pair counts as values.

    Raises:
        FileNotFoundError: If source images/ or labels/ folder is missing.
        ValueError:        If ratios are invalid or no matched pairs found.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1.0 "
            "(the remainder becomes the test set)."
        )

    src = Path(source_folder)
    images_dir = src / "images"
    labels_dir = src / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"'images' folder not found in: {source_folder}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"'labels' folder not found in: {source_folder}")

    # Collect image/label pairs that have matching stems
    image_files = [
        f for f in sorted(images_dir.iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    pairs = [
        (img, labels_dir / (img.stem + ".txt"))
        for img in image_files
        if (labels_dir / (img.stem + ".txt")).exists()
    ]

    if not pairs:
        raise ValueError("No matched image/label pairs found.")

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs   = pairs[train_end:val_end]
    test_pairs  = pairs[val_end:]

    # Create all output sub-directories up front
    out = Path(output_folder)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Track global progress across all three splits
    all_pairs = [
        ("train", train_pairs),
        ("val",   val_pairs),
        ("test",  test_pairs),
    ]
    done = 0
    total = n

    for split_name, pair_list in all_pairs:
        for img_path, lbl_path in pair_list:
            shutil.copy2(img_path, out / "images" / split_name / img_path.name)
            shutil.copy2(lbl_path, out / "labels" / split_name / lbl_path.name)
            done += 1
            if progress_callback:
                progress_callback(done, total)

    return {
        "train": len(train_pairs),
        "val":   len(val_pairs),
        "test":  len(test_pairs),
    }

