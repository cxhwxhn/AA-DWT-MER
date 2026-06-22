from __future__ import annotations

import numpy as np


def temporal_bandpass_4d(
    video4d: np.ndarray, frame_rate: float, low_freq: float, high_freq: float
) -> np.ndarray:
    """Apply an FFT band-pass filter along the time axis."""
    t_num = video4d.shape[0]
    frequencies = np.fft.fftfreq(t_num, d=1.0 / frame_rate)
    mask = ((np.abs(frequencies) >= low_freq) & (np.abs(frequencies) <= high_freq))
    mask = mask.reshape(t_num, 1, 1, 1)
    return np.real(np.fft.ifft(np.fft.fft(video4d, axis=0) * mask, axis=0))
