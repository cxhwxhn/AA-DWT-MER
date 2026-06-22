from __future__ import annotations

import numpy as np


def rgb_video_to_gray(video4d: np.ndarray) -> np.ndarray:
    if video4d.ndim == 3:
        return video4d.astype(np.float64)
    if video4d.ndim != 4:
        raise ValueError("video4d must be a four-dimensional array with shape (T, H, W, C).")
    if video4d.shape[-1] == 1:
        return video4d[..., 0].astype(np.float64)
    return (
        0.2989 * video4d[..., 0]
        + 0.5870 * video4d[..., 1]
        + 0.1140 * video4d[..., 2]
    ).astype(np.float64)


def calculate_high_frequency_energy(
    gray_video: np.ndarray, spatial_threshold: float
) -> float:
    if gray_video.ndim != 3:
        raise ValueError("gray_video must be a three-dimensional array with shape (T, H, W).")

    t_num, height, width = gray_video.shape
    spectrum = np.fft.fftshift(np.fft.fftn(np.transpose(gray_video, (1, 2, 0))))
    u = np.arange(-np.floor(width / 2), np.ceil(width / 2)) / width
    v = np.arange(-np.floor(height / 2), np.ceil(height / 2)) / height
    u_grid, v_grid = np.meshgrid(u, v)
    spatial_mask = np.sqrt(u_grid**2 + v_grid**2) > spatial_threshold / 100.0
    high_spectrum = spectrum * np.repeat(spatial_mask[..., np.newaxis], t_num, axis=2)
    return float(2.6 * np.sum(np.abs(high_spectrum) ** 2) / high_spectrum.size)
