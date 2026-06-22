from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Params
from .metrics import calculate_high_frequency_energy, rgb_video_to_gray
from .model import prepare_dynamic_evm_model, reconstruct_amplified_video
from .video_io import read_video_4d, write_video_4d


def adaptive_dynamic_weak_texture_amplification(
    input_video: str | Path, output_video: str | Path, params: Params
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Amplify weak dynamic textures and automatically select the best alpha."""
    video4d, frame_rate = read_video_4d(input_video)
    original_energy = calculate_high_frequency_energy(
        rgb_video_to_gray(video4d), params.spatial_high_freq_threshold
    )
    model = prepare_dynamic_evm_model(video4d, frame_rate, params)

    alpha = float(params.alpha0)
    best_alpha = alpha
    best_error = float("inf")
    best_video: np.ndarray | None = None
    log_data: list[list[float]] = []

    for iteration in range(1, params.max_iter + 1):
        enhanced = reconstruct_amplified_video(model, alpha)
        amplified_energy = calculate_high_frequency_energy(
            rgb_video_to_gray(enhanced), params.spatial_high_freq_threshold
        )
        ratio = (amplified_energy - params.k * original_energy) / (
            original_energy + params.epsilon
        )
        error = abs(ratio - params.target_r)
        log_data.append([alpha, ratio])
        print(f"Iteration {iteration}/{params.max_iter}: alpha={alpha:.4f}, R={ratio:.6f}")

        if np.isfinite(error) and error < best_error:
            best_error, best_alpha, best_video = error, alpha, enhanced.copy()
        if error <= params.tol:
            break

        feedback_step = params.beta * (ratio - params.target_r)
        if abs(feedback_step) < 1.0:
            feedback_step = np.sign(feedback_step) * 1.0
        if abs(feedback_step) > 5.0:
            feedback_step = np.sign(feedback_step) * 5.0
        alpha_new = 0.8 * (alpha - feedback_step)
        if not np.isfinite(alpha_new):
            alpha_new = alpha - 2.0 if ratio > params.target_r else alpha + 2.0
        alpha = float(np.clip(alpha_new, params.alpha_min, params.alpha_max))

    if best_video is None:
        raise RuntimeError(
            "No valid amplified video was produced. Check the input video and parameters."
        )

    log_table = pd.DataFrame(log_data, columns=["Alpha", "HighFreqRatio_R"])
    write_video_4d(best_video, output_video, frame_rate)
    return best_video, best_alpha, log_table
