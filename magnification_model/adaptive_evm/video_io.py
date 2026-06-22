from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_video_4d(video_path: str | Path) -> tuple[np.ndarray, float]:
    """Read a video as an RGB array with shape (T, H, W, C) in [0, 1]."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Input video does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        frame_rate = 30.0

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_bgr.ndim == 2:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float64) / 255.0)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("No video frames were read. Check the input video.")
    return np.stack(frames, axis=0), float(frame_rate)


def write_video_4d(
    video4d: np.ndarray, output_path: str | Path, frame_rate: float
) -> None:
    """Write an RGB array with shape (T, H, W, C) to a video."""
    if video4d is None or video4d.size == 0:
        raise ValueError("video4d is empty and cannot be written to a video.")
    if video4d.ndim == 3:
        video4d = video4d[..., np.newaxis]
    if video4d.ndim != 4:
        raise ValueError("video4d must be a four-dimensional array with shape (T, H, W, C).")

    t_num, h, w, channels = video4d.shape
    if t_num == 0:
        raise ValueError("The video contains zero frames.")
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        frame_rate = 30.0

    # Common codecs typically require even frame dimensions.
    new_h, new_w = h + h % 2, w + w % 2
    if (new_h, new_w) != (h, w):
        video4d = np.pad(
            video4d,
            ((0, 0), (0, new_h - h), (0, new_w - w), (0, 0)),
            mode="edge",
        )
        h, w = new_h, new_w

    if channels == 1:
        video4d = np.repeat(video4d, 3, axis=3)
    elif channels != 3:
        raise ValueError(f"Video frames must have 1 or 3 channels; got {channels}.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if path.suffix.lower() == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(path), fourcc, float(frame_rate), (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {path}")

    clean_video = np.clip(
        np.nan_to_num(video4d, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
    )
    try:
        for frame_rgb in clean_video:
            frame_uint8 = (frame_rgb * 255.0).round().astype(np.uint8)
            writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
