from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    """Algorithm parameters."""

    tau: float = 8.0
    low_freq: float = 0.3
    high_freq: float = 6.0
    num_levels: int = 4
    target_r: float = 0.11
    alpha0: float = 50.0
    alpha_min: float = 1.0
    alpha_max: float = 80.0
    beta: float = 0.1
    max_iter: int = 100
    tol: float = 1e-5
    spatial_high_freq_threshold: float = 15.0
    k: float = 1.0
    epsilon: float = 1e-8
