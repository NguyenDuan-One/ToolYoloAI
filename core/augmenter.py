"""
augmenter.py
------------
Generates augmented image variants from a folder of source images.

Design philosophy:
  - Each enabled augmentation produces ONE additional image variant per source.
  - All parameters map 1-to-1 to their albumentations equivalents.
  - The "enabled" flag is a simple boolean – callers manage this via the GUI.

Augmentation groups supported:
  1. Rotation       – up to 4 specific angles (skip if angle == 0)
  2. Color/Exposure – brightness, contrast, hue, saturation, value
  3. Blur           – motion blur, Gaussian blur
  4. Noise          – Gaussian noise
  5. Fog            – weather-like fog overlay
"""

import cv2
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import albumentations as A


# ---------------------------------------------------------------------------
# Configuration dataclass – maps to GUI inputs
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    """All parameters controlling augmentation behaviour."""

    # -- Rotation --
    rotation_enabled: bool = False
    rotation_angles: list[float] = field(default_factory=lambda: [0, 0, 0, 0])

    # -- Color / Exposure (RandomBrightnessContrast + HueSaturationValue) --
    color_enabled: bool = False
    brightness_limit: float = 0.25
    contrast_limit: float = 0.25

    hsv_enabled: bool = False
    hue_shift_limit: int = 10
    sat_shift_limit: int = 20
    val_shift_limit: int = 15

    # -- Blur --
    motion_blur_enabled: bool = False
    motion_blur_limit: int = 7

    gaussian_blur_enabled: bool = False
    gaussian_blur_limit: int = 5

    # -- Noise --
    gauss_noise_enabled: bool = False
    gauss_noise_var_limit: tuple[int, int] = (5, 35)

    # -- Fog --
    fog_enabled: bool = False
    fog_coef_lower: float = 0.05
    fog_coef_upper: float = 0.20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by `angle` degrees around its center (keeps original size).

    Empty corners are filled with BLACK (0) instead of mirror-reflected pixels.
    Mirror fill creates fake pixel data that can degrade AI model training.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),    # black fill – clearly marks non-image area
    )



def _build_transforms(cfg: AugConfig) -> list[tuple[str, A.BasicTransform]]:
    """
    Build a list of (label, albumentations transform) pairs based on `cfg`.

    One entry in this list → one image variant will be produced.
    """
    transforms: list[tuple[str, A.BasicTransform]] = []

    if cfg.color_enabled:
        transforms.append((
            "color",
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=1.0,
            ),
        ))

    if cfg.hsv_enabled:
        transforms.append((
            "hsv",
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=cfg.sat_shift_limit,
                val_shift_limit=cfg.val_shift_limit,
                p=1.0,
            ),
        ))

    if cfg.motion_blur_enabled:
        limit = cfg.motion_blur_limit
        if limit % 2 == 0:
            limit += 1          # kernel must be odd
        transforms.append((
            "motion_blur",
            A.MotionBlur(blur_limit=limit, p=1.0),
        ))

    if cfg.gaussian_blur_enabled:
        limit = cfg.gaussian_blur_limit
        if limit % 2 == 0:
            limit += 1          # kernel must be odd
        transforms.append((
            "gaussian_blur",
            A.GaussianBlur(blur_limit=(limit, limit), p=1.0),
        ))

    if cfg.gauss_noise_enabled:
        transforms.append((
            "gauss_noise",
            A.GaussNoise(var_limit=cfg.gauss_noise_var_limit, p=1.0),
        ))

    if cfg.fog_enabled:
        transforms.append((
            "fog",
            A.RandomFog(
                fog_coef_range=(cfg.fog_coef_lower, cfg.fog_coef_upper),
                alpha_coef=0.1,
                p=1.0,
            ),
        ))


    return transforms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augment_folder(
    input_folder: str,
    output_folder: str,
    cfg: AugConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Apply augmentations to all images in `input_folder` and save results
    to `output_folder`.

    Strategy:
      • Each enabled albumentations transform → 1 extra image per source.
      • Each non-zero rotation angle → 1 extra image per source.
      • The original image is always copied unchanged.

    Args:
        input_folder:      Source folder with the original images.
        output_folder:     Destination folder for augmented images.
        cfg:               AugConfig with all parameters and enabled flags.
        progress_callback: Optional callback(current, total).

    Returns:
        Total number of images written (including originals).
    """
    src = Path(input_folder)
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in src.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise ValueError(f"No images found in: {input_folder}")

    transforms = _build_transforms(cfg)
    active_angles = [a for a in cfg.rotation_angles if a != 0] if cfg.rotation_enabled else []

    total = len(image_files)
    written = 0

    for idx, img_path in enumerate(image_files):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue                        # skip unreadable files

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        stem = img_path.stem
        ext = img_path.suffix

        # Always copy the original
        cv2.imwrite(str(out / img_path.name), image_bgr)
        written += 1

        # Apply each albumentations transform
        for label, transform in transforms:
            augmented = transform(image=image_rgb)["image"]
            result_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            out_name = f"{stem}_aug_{label}{ext}"
            cv2.imwrite(str(out / out_name), result_bgr)
            written += 1

        # Rotation variants
        for angle in active_angles:
            rotated = _rotate_image(image_bgr, angle)
            out_name = f"{stem}_rot{int(angle)}{ext}"
            cv2.imwrite(str(out / out_name), rotated)
            written += 1

        if progress_callback:
            progress_callback(idx + 1, total)

    return written
