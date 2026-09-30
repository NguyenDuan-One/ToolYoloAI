"""
auto_labeler.py
---------------
Automatically generates YOLO-format label (.txt) files for a folder of images
using an Ultralytics YOLO model (YOLOv8 / YOLO11).

Design:
  - One .txt file per image, same stem, in the output labels folder.
  - Each line: <class_id> <cx> <cy> <w> <h>  (normalised 0–1).
  - Images with zero detections get an empty .txt file so the dataset
    structure stays consistent.
  - Accepts either a pretrained model name ("yolov8n.pt", "yolo11n.pt", …)
    or an absolute path to a custom-trained weights file.
"""
import os
import ssl
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable

from ultralytics import YOLO


# Supported image extensions (same set used across the project)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bundled COCO class names as a convenience for display
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
}


def _download_weights(model_name: str):
    """
    Explicitly download model weights from GitHub assets using an unverified SSL context.
    This bypasses local certificate issues common on some Windows environments.
    """
    # Try latest tags sequentially for maximum compatibility
    for tag in ["v8.4.0", "v8.3.0"]:
        url = f"https://github.com/ultralytics/assets/releases/download/{tag}/{model_name}"
        dest = Path(model_name)
        ctx = ssl._create_unverified_context()
        
        print(f"--- Explicitly downloading {model_name} from {url} ...")
        try:
            with urllib.request.urlopen(url, context=ctx) as response:
                with open(dest, 'wb') as f:
                    f.write(response.read())
            print(f"--- Download complete: {dest}")
            return # Success
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue # Try next tag
            raise e
        except Exception as e:
            raise e

    raise FileNotFoundError(
        f"Weights for '{model_name}' are not available for automatic download yet.\n"
        f"TIP: Please ensure you selected a valid variant (e.g., {model_name.replace('.pt', 'n.pt')} if not already)."
    )


def get_model_classes(model_name: str) -> dict[int, str]:
    """
    Load a YOLO model and return its class names mapping {id: name}.
    """
    # If it's a preset and missing, try explicit download first
    if not Path(model_name).exists() and model_name.endswith(".pt") and "yolo" in model_name.lower():
        try:
            _download_weights(model_name)
        except Exception as e:
            print(f"--- Explicit download failed: {e}")

    try:
        model = YOLO(model_name)
        if not hasattr(model, 'names') or not model.names:
             raise ValueError("Model loaded successfully but has no class names.")
        return model.names
    except (FileNotFoundError, Exception) as e:
        err_msg = str(e)
        if "names" in err_msg.lower() or "not found" in err_msg.lower() or isinstance(e, FileNotFoundError):
            raise Exception(
                f"Error loading model '{model_name}': {e}\n"
                "TIP: Your 'ultralytics' library might be outdated. Please run: pip install -U ultralytics"
            ) from e
        raise e


def auto_label(
    input_folder: str,
    output_folder: str,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_classes: list[str | int] | None = None,
    device: str = "",                   # "" = auto (GPU if available, else CPU)
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Run YOLO inference on every image in `input_folder` and write YOLO-format
    label files to `output_folder`.

    Args:
        target_classes:    Optional list of class IDs (int) or names (str) to keep.
                           If None or empty, all classes detected by the model are kept.
        ... (others same)

    Returns:
        dict with keys:
            "processed", "labeled", "empty", "classes",
            "model_names" – dict {id: name} from the model
    """
    src = Path(input_folder)
    out = Path(output_folder)

    if not src.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    out.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in src.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        raise ValueError(f"No images found in: {input_folder}")

    # Load model
    model = YOLO(model_name)
    model_names = model.names  # {id: name}

    # Resolve target_classes to a set of integer IDs
    active_ids: set[int] | None = None
    if target_classes:
        active_ids = set()
        # Create a reverse mapping {name.lower(): id} for easier lookup
        name_to_id = {str(v).lower(): k for k, v in model_names.items()}
        for item in target_classes:
            item_str = str(item).strip().lower()
            if item_str in name_to_id:
                active_ids.add(name_to_id[item_str])
            elif item_str.isdigit():
                active_ids.add(int(item_str))

    stats = {
        "processed": 0,
        "labeled": 0,
        "empty": 0,
        "classes": {},
        "model_names": model_names
    }
    total = len(image_files)

    for idx, img_path in enumerate(image_files):
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False,
        )

        label_lines: list[str] = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())

                # Filter by active_ids if specified
                if active_ids is not None and cls_id not in active_ids:
                    continue

                cx, cy, bw, bh = box.xywhn[0].tolist()
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # Accumulate class stats
                cls_name = model_names.get(cls_id, str(cls_id))
                stats["classes"][cls_name] = stats["classes"].get(cls_name, 0) + 1

            if label_lines:
                stats["labeled"] += 1
            else:
                stats["empty"] += 1
        else:
            stats["empty"] += 1

        # Write label file (always write, even if empty, for consistency)
        label_path = out / (img_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        stats["processed"] += 1
        if progress_callback:
            progress_callback(idx + 1, total)

    return stats


def generate_yaml(output_path: str, model_names: dict, train_path: str = "../train"):
    """
    Generate a data.yaml file required for YOLO training.
    """
    path = Path(output_path)
    ids = sorted(model_names.keys())
    nc = len(ids)
    names_list = [model_names[i] for i in ids]

    content = [
        f"path: {train_path}  # root directory of images",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {nc}",
        f"names: {names_list}"
    ]

    path.write_text("\n".join(content), encoding="utf-8")


def generate_classes_txt(output_path: str, model_names: dict):
    """
    Generate a classes.txt file with one class per line (index = class ID).
    """
    path = Path(output_path)
    ids = sorted(model_names.keys())
    # Ensure every ID from 0 to max(ids) is represented to avoid gaps
    max_id = max(ids) if ids else -1
    lines = []
    for i in range(max_id + 1):
        lines.append(model_names.get(i, f"class_{i}"))
    
    path.write_text("\n".join(lines), encoding="utf-8")

