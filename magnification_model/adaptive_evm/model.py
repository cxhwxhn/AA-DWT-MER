from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Params
from .filters import temporal_bandpass_4d
from .metrics import rgb_video_to_gray
from .pyramid import build_laplacian_pyramid, reconstruct_laplacian_pyramid


@dataclass
class DynamicEVMModel:
    base_pyramid: list[np.ndarray]
    dynamic_pyramid: list[np.ndarray]
    num_levels: int
    frame_count: int


def prepare_dynamic_evm_model(
    video4d: np.ndarray, frame_rate: float, params: Params
) -> DynamicEVMModel:
    frame_count, height, width, _ = video4d.shape
    max_level = max(1, int(np.floor(np.log2(min(height, width)))) - 3)
    num_levels = min(params.num_levels, max_level)

    gray_video = rgb_video_to_gray(video4d)
    masks = np.zeros((frame_count, height, width), dtype=bool)
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    for index in range(1, frame_count):
        moving = np.abs(gray_video[index] - gray_video[index - 1]) > params.tau / 255.0
        masks[index] = cv2.filter2D(moving.astype(np.float64), -1, kernel) > 0.3

    base_pyramid: list[np.ndarray] = []
    mask_pyramid: list[np.ndarray] = []
    for index, frame in enumerate(video4d):
        pyramid = build_laplacian_pyramid(frame, num_levels)
        for level, layer in enumerate(pyramid):
            layer_h, layer_w, channels = layer.shape
            if index == 0:
                base_pyramid.append(
                    np.zeros((frame_count, layer_h, layer_w, channels), dtype=np.float64)
                )
                mask_pyramid.append(
                    np.zeros((frame_count, layer_h, layer_w, 1), dtype=np.float64)
                )
            base_pyramid[level][index] = layer
            mask_pyramid[level][index, ..., 0] = cv2.resize(
                masks[index].astype(np.float64),
                (layer_w, layer_h),
                interpolation=cv2.INTER_NEAREST,
            )

    dynamic_pyramid = []
    for base, mask in zip(base_pyramid, mask_pyramid):
        filtered = temporal_bandpass_4d(
            base, frame_rate, params.low_freq, params.high_freq
        )
        dynamic_pyramid.append(filtered * mask)

    return DynamicEVMModel(base_pyramid, dynamic_pyramid, num_levels, frame_count)


def reconstruct_amplified_video(model: DynamicEVMModel, alpha: float) -> np.ndarray:
    _, height, width, channels = model.base_pyramid[0].shape
    result = np.zeros((model.frame_count, height, width, channels), dtype=np.float64)
    for index in range(model.frame_count):
        pyramid = [
            model.base_pyramid[level][index]
            + alpha * model.dynamic_pyramid[level][index]
            for level in range(model.num_levels)
        ]
        result[index] = np.clip(reconstruct_laplacian_pyramid(pyramid), 0.0, 1.0)
    return result
