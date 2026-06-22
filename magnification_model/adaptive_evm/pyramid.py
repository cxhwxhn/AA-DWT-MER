from __future__ import annotations

import cv2
import numpy as np


def build_laplacian_pyramid(img: np.ndarray, num_levels: int) -> list[np.ndarray]:
    gaussian = [img]
    for _ in range(1, num_levels):
        previous = gaussian[-1]
        size = (max(1, previous.shape[1] // 2), max(1, previous.shape[0] // 2))
        gaussian.append(cv2.resize(previous, size, interpolation=cv2.INTER_LINEAR))

    laplacian: list[np.ndarray] = []
    for current, smaller in zip(gaussian[:-1], gaussian[1:]):
        up = cv2.resize(
            smaller, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        laplacian.append(current - up)
    laplacian.append(gaussian[-1])
    return laplacian


def reconstruct_laplacian_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    image = pyramid[-1]
    for target in reversed(pyramid[:-1]):
        image = cv2.resize(
            image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        image = image + target
    return image
