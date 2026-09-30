"""
video_converter.py
------------------
Extracts frames from a video file and saves them as images.

Modes:
  - "seconds":  Extract one frame every N seconds.
  - "frames":   Extract every N-th frame from the video.
"""

import cv2
import os
from pathlib import Path
from typing import Callable


def extract_frames(
    video_path: str,
    output_folder: str,
    mode: str = "seconds",          # "seconds" or "frames"
    value: float = 1.0,             # seconds interval OR frame step
    prefix: str = "frame",
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Extract frames from a video file.

    Args:
        video_path:        Path to the input video file.
        output_folder:     Directory where extracted images are saved.
        mode:              "seconds" to extract by time interval,
                           "frames" to extract every N-th frame.
        value:             Time interval (seconds) or frame step count.
        prefix:            Filename prefix for saved images.
        progress_callback: Optional callback(current, total) for UI progress.

    Returns:
        Number of images saved.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError:        If mode is invalid or value <= 0.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if value <= 0:
        raise ValueError("value must be greater than 0.")
    if mode not in ("seconds", "frames"):
        raise ValueError("mode must be 'seconds' or 'frames'.")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        cap.release()
        raise ValueError("Could not read FPS from the video. The file may be corrupted.")

    # Compute the step in frame index units
    if mode == "seconds":
        frame_step = max(1, int(fps * value))
    else:  # "frames"
        frame_step = max(1, int(value))

    saved_count = 0
    frame_index = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        filename = output_dir / f"{prefix}_{frame_index:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        saved_count += 1

        if progress_callback:
            progress_callback(frame_index, total_frames)

        frame_index += frame_step

    cap.release()
    return saved_count
