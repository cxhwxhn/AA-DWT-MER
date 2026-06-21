# -*- coding: utf-8 -*-
"""
Python version of the MATLAB code for adaptive dynamic weak-texture information amplification.

Dependencies:
    pip install numpy opencv-python pandas

Usage:
    python adaptive_dynamic_weak_texture_amplification.py

You can also modify input_video and output_video in the __main__ block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class Params:
    tau: float = 8.0                         # 动态区域阈值
    low_freq: float = 0.3                    # 时域带通滤波下限 Hz
    high_freq: float = 6.0                   # 时域带通滤波上限 Hz
    num_levels: int = 4                      # Laplacian 金字塔层数

    target_r: float = 0.11                   # 目标高频增量比
    alpha0: float = 50.0                     # 初始放大因子
    alpha_min: float = 1.0                   # 最小放大因子
    alpha_max: float = 80.0                  # 最大放大因子
    beta: float = 0.1                       # 学习率 / 反馈步长系数
    max_iter: int = 100                      # 最大迭代次数
    tol: float = 1e-5                        # 收敛阈值

    spatial_high_freq_threshold: float = 0.15
    k: float = 1.0                           # 高频能量比例系数
    epsilon: float = 1e-8                    # 防止除零


# =====================================================================
# 主函数：动态弱纹理信息自适应放大
# =====================================================================
def adaptive_dynamic_weak_texture_amplification(
    input_video: str | Path,
    output_video: str | Path,
    params: Params,
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """
    输入视频 -> 构建动态弱纹理 EVM 模型 -> 自适应寻找最优 alpha -> 保存增强视频。

    Returns
    -------
    enhanced_video : np.ndarray
        增强后视频，shape = (T, H, W, C)，取值范围 [0, 1]。
    alpha_opt : float
        最优放大因子。
    log_table : pandas.DataFrame
        迭代日志。
    """
    video4d, frame_rate = read_video_4d(input_video)
    print(f"视频读取完成：{video4d.shape[0]} 帧，帧率 {frame_rate:.2f} FPS。")

    video_gray = rgb_video_to_gray(video4d)

    # 计算原始视频的空间高频能量
    e_original = calculate_high_frequency_energy(
        video_gray, params.spatial_high_freq_threshold
    )
   

    # 构建动态弱纹理放大模型
    model = prepare_dynamic_evm_model(video4d, frame_rate, params)

    # 初始化 alpha
    alpha = float(params.alpha0)

    log_data: List[List[float]] = []

    # 不再使用 loss，直接用 |R - target_r| 作为评价误差
    best_error = float("inf")
    best_alpha = alpha
    best_video: np.ndarray | None = None

    for iter_idx in range(1, params.max_iter + 1):
        enhanced_video = reconstruct_amplified_video(model, alpha)

        e_amplified = calculate_high_frequency_energy(
            rgb_video_to_gray(enhanced_video), params.spatial_high_freq_threshold
        )

        r_value = (e_amplified - params.k * e_original) / (
            e_original + params.epsilon
        )

        error = abs(r_value - params.target_r)

        # 记录迭代次数、alpha 与 R
        log_data.append([iter_idx, alpha, r_value])

        print(
            f"Iter {iter_idx}: alpha = {alpha:.4f}, HighFreqRatio_R = {r_value:.6f}"
        )

        # 记录当前最优结果
        if error < best_error:
            best_error = error
            best_alpha = alpha
            best_video = enhanced_video.copy()

        # 判断是否收敛
        if error < params.tol:
            print("达到收敛条件。")
            break

        # ================== 加速型反馈更新 ==================
        error_r = r_value - params.target_r

        # 当 R 大于目标值时，高频噪声偏多，需要降低 alpha
        # 当 R 小于目标值时，放大不足，需要提高 alpha
        feedback_step = params.beta * error_r

        # 设置最小更新步长，避免 alpha 下降太慢
        min_step = 1.0
        if abs(feedback_step) < min_step:
            feedback_step = np.sign(feedback_step) * min_step

        # 设置最大更新步长，防止 alpha 一次变化过大
        max_step = 5.0
        if abs(feedback_step) > max_step:
            feedback_step = np.sign(feedback_step) * max_step

        # 更新 alpha
        alpha_new = 0.8*(alpha - feedback_step)

        # 如果更新异常，则采用简单反馈
        if not np.isfinite(alpha_new):
            if r_value > params.target_r:
                alpha_new = alpha - 2.0
            else:
                alpha_new = alpha + 2.0

        # 限制 alpha 范围
        alpha_new = min(max(alpha_new, params.alpha_min), params.alpha_max)

        # 更新 alpha
        alpha = alpha_new

    log_table = pd.DataFrame(
        log_data,
        columns=["Iter", "Alpha", "HighFreqRatio_R"],
    )

    alpha_opt = best_alpha
    if best_video is None:
        raise RuntimeError("未得到有效的增强视频，请检查输入视频或参数设置。")

    enhanced_video = best_video
    print(f"最终采用的最优 alpha = {alpha_opt:.4f}")

    # 保存最终增强视频
    write_video_4d(enhanced_video, output_video, frame_rate)
    print(f"增强后视频已保存至：{output_video}")

    return enhanced_video, alpha_opt, log_table


# =====================================================================
# 读取视频为 T × H × W × C，取值范围 [0, 1]，RGB 顺序
# =====================================================================
def read_video_4d(video_path: str | Path) -> Tuple[np.ndarray, float]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"输入视频不存在：{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入视频：{video_path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate is None or not np.isfinite(frame_rate) or frame_rate <= 0:
        frame_rate = 30.0

    frames: List[np.ndarray] = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # OpenCV 默认是 BGR，这里转换成 RGB，与 MATLAB 的 RGB 读入更一致
        if frame_bgr.ndim == 2:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_rgb = frame_rgb.astype(np.float64) / 255.0
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("未读取到任何视频帧，请检查输入视频。")

    video4d = np.stack(frames, axis=0)  # T, H, W, C
    return video4d, float(frame_rate)


# =====================================================================
# 构建动态弱纹理 EVM 模型
# =====================================================================
def prepare_dynamic_evm_model(
    video4d: np.ndarray,
    frame_rate: float,
    params: Params,
) -> Dict[str, object]:
    t_num, h, w, c = video4d.shape

    # 自动限制金字塔层数，防止图像尺寸过小
    max_level = max(1, int(np.floor(np.log2(min(h, w)))) - 3)
    num_levels = min(params.num_levels, max_level)
    print(f"使用 Laplacian 金字塔层数：{num_levels}")

    gray_video = rgb_video_to_gray(video4d)

    # ================== Step 1 & Step 2：动态区域掩膜 ==================
    mask_video = np.zeros((t_num, h, w), dtype=bool)

    # 输入视频已经归一化到 [0,1]，因此 tau 需要除以 255
    tau_norm = params.tau / 255.0

    kernel = np.ones((3, 3), dtype=np.float64) / 9.0

    for t in range(1, t_num):
        diff_frame = np.abs(gray_video[t] - gray_video[t - 1])
        mask = diff_frame > tau_norm

        # 平滑动态区域，减少孤立噪声点
        mask_smooth = cv2.filter2D(mask.astype(np.float64), -1, kernel)
        mask_video[t] = mask_smooth > 0.3

    # ================== Step 3：Laplacian 金字塔分解 ==================
    base_pyr: List[np.ndarray] = []
    mask_pyr: List[np.ndarray] = []

    for t in range(t_num):
        frame = video4d[t]
        pyr = build_laplacian_pyramid(frame, num_levels)

        for level in range(num_levels):
            ph, pw, pc = pyr[level].shape

            if t == 0:
                base_pyr.append(np.zeros((t_num, ph, pw, pc), dtype=np.float64))
                mask_pyr.append(np.zeros((t_num, ph, pw, 1), dtype=np.float64))

            base_pyr[level][t] = pyr[level]

            resized_mask = cv2.resize(
                mask_video[t].astype(np.float64),
                (pw, ph),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_pyr[level][t, :, :, 0] = resized_mask

    # ================== Step 3：时域带通滤波 ==================
    dynamic_pyr: List[np.ndarray] = []

    for level in range(num_levels):
        filtered = temporal_bandpass_4d(
            base_pyr[level], frame_rate, params.low_freq, params.high_freq
        )

        # mask_pyr[level]: T,H,W,1，可自动广播到 T,H,W,C
        dynamic = filtered * mask_pyr[level]
        dynamic_pyr.append(dynamic)

    model: Dict[str, object] = {
        "base_pyr": base_pyr,
        "dynamic_pyr": dynamic_pyr,
        "num_levels": num_levels,
        "t_num": t_num,
    }
    return model


# =====================================================================
# Step 4：使用 alpha 重构增强视频
# =====================================================================
def reconstruct_amplified_video(model: Dict[str, object], alpha: float) -> np.ndarray:
    base_pyr: List[np.ndarray] = model["base_pyr"]  # type: ignore[assignment]
    dynamic_pyr: List[np.ndarray] = model["dynamic_pyr"]  # type: ignore[assignment]
    num_levels: int = int(model["num_levels"])
    t_num: int = int(model["t_num"])

    _, h, w, c = base_pyr[0].shape
    enhanced_video = np.zeros((t_num, h, w, c), dtype=np.float64)

    for t in range(t_num):
        pyr: List[np.ndarray] = []
        for level in range(num_levels):
            current_level = base_pyr[level][t] + alpha * dynamic_pyr[level][t]
            pyr.append(current_level)

        frame = reconstruct_laplacian_pyramid(pyr)
        frame = np.clip(frame, 0.0, 1.0)
        enhanced_video[t] = frame

    return enhanced_video


# =====================================================================
# Laplacian 金字塔分解
# =====================================================================
def build_laplacian_pyramid(img: np.ndarray, num_levels: int) -> List[np.ndarray]:
    gaussian_pyr: List[np.ndarray] = [img]

    for level in range(1, num_levels):
        prev = gaussian_pyr[level - 1]
        new_w = max(1, prev.shape[1] // 2)
        new_h = max(1, prev.shape[0] // 2)
        down = cv2.resize(prev, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        gaussian_pyr.append(down)

    laplacian_pyr: List[np.ndarray] = []
    for level in range(num_levels - 1):
        current = gaussian_pyr[level]
        next_level = gaussian_pyr[level + 1]
        up = cv2.resize(
            next_level,
            (current.shape[1], current.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        laplacian_pyr.append(current - up)

    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr


# =====================================================================
# Laplacian 金字塔重构
# =====================================================================
def reconstruct_laplacian_pyramid(pyr: List[np.ndarray]) -> np.ndarray:
    img = pyr[-1]

    for level in range(len(pyr) - 2, -1, -1):
        target = pyr[level]
        img = cv2.resize(
            img,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        img = img + target

    return img


# =====================================================================
# 沿时间维度进行带通滤波
# =====================================================================
def temporal_bandpass_4d(
    video4d: np.ndarray,
    frame_rate: float,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """
    video4d shape = T × H × W × C。
    这里沿第 0 维，也就是时间维度做 FFT 带通滤波。
    """
    t_num = video4d.shape[0]

    freq = np.fft.fftfreq(t_num, d=1.0 / frame_rate)
    band_mask = (np.abs(freq) >= low_freq) & (np.abs(freq) <= high_freq)
    band_mask = band_mask.reshape((t_num, 1, 1, 1))

    f_video = np.fft.fft(video4d, axis=0)
    f_filtered = f_video * band_mask
    filtered_video = np.real(np.fft.ifft(f_filtered, axis=0))

    return filtered_video


# =====================================================================
# 计算空间高频能量
# =====================================================================
def calculate_high_frequency_energy(
    gray_video: np.ndarray,
    spatial_threshold: float,
) -> float:
    """
    gray_video shape = T × H × W。
    MATLAB 版本是 H × W × T，这里只要频域掩膜与维度对应即可。
    """
    if gray_video.ndim != 3:
        raise ValueError("gray_video 必须是 3 维数组，shape = T × H × W。")

    t_num, h, w = gray_video.shape

    # 为了对应 MATLAB 的 H × W × T，这里转成 H × W × T 再做 fftn
    gray_hwt = np.transpose(gray_video, (1, 2, 0))

    f = np.fft.fftshift(np.fft.fftn(gray_hwt))

    u = np.arange(-np.floor(w / 2), np.ceil(w / 2)) / w
    v = np.arange(-np.floor(h / 2), np.ceil(h / 2)) / h
    u_grid, v_grid = np.meshgrid(u, v)

    h_space = np.sqrt(u_grid ** 2 + v_grid ** 2) > spatial_threshold
    h_space_3d = np.repeat(h_space[:, :, np.newaxis], t_num, axis=2)

    f_high = f * h_space_3d
    e_high = 2.6*np.sum(np.abs(f_high) ** 2) / f_high.size

    return float(e_high)


# =====================================================================
# RGB 视频转灰度视频
# =====================================================================
def rgb_video_to_gray(video4d: np.ndarray) -> np.ndarray:
    """
    输入 shape = T × H × W × C，输出 shape = T × H × W。
    """
    if video4d.ndim == 3:
        return video4d.astype(np.float64)

    if video4d.ndim != 4:
        raise ValueError("video4d 必须是 4 维数组，shape = T × H × W × C。")

    if video4d.shape[-1] == 1:
        return video4d[:, :, :, 0].astype(np.float64)

    r = video4d[:, :, :, 0]
    g = video4d[:, :, :, 1]
    b = video4d[:, :, :, 2]

    gray_video = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_video.astype(np.float64)


# =====================================================================
# 保存 T × H × W × C 视频
# 修正版：处理帧尺寸、帧率、NaN、Inf、奇数宽高等问题
# =====================================================================
def write_video_4d(
    video4d: np.ndarray,
    output_path: str | Path,
    frame_rate: float,
) -> None:
    if video4d is None or video4d.size == 0:
        raise ValueError("video4d 为空，无法写入视频。")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if video4d.ndim == 3:
        # T,H,W -> T,H,W,1
        video4d = video4d[:, :, :, np.newaxis]

    if video4d.ndim != 4:
        raise ValueError("video4d 必须是 4 维数组，shape = T × H × W × C。")

    t_num, h, w, c = video4d.shape

    if t_num < 1:
        raise ValueError("视频帧数为 0，无法写入视频。")

    if frame_rate is None or (not np.isfinite(frame_rate)) or frame_rate <= 0:
        frame_rate = 30.0
    frame_rate = float(frame_rate)

    # MPEG-4 和部分编码器通常要求宽高为偶数
    new_h = h + (h % 2)
    new_w = w + (w % 2)

    if new_h != h or new_w != w:
        print(f"检测到视频尺寸为 {h} × {w}，已自动填充为 {new_h} × {new_w}。")
        pad_bottom = new_h - h
        pad_right = new_w - w
        video4d = np.pad(
            video4d,
            pad_width=((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)),
            mode="edge",
        )
        t_num, h, w, c = video4d.shape

    # 如果是单通道，转换成三通道
    if c == 1:
        video4d = np.repeat(video4d, 3, axis=3)
        c = 3

    if c != 3:
        raise ValueError(f"输出视频必须是 1 通道或 3 通道，目前通道数为 {c}。")

    # 去除 NaN 和 Inf，并限制到 [0, 1]
    video4d = np.nan_to_num(video4d, nan=0.0, posinf=0.0, neginf=0.0)
    video4d = np.clip(video4d, 0.0, 1.0)

    # 如果文件已存在，先删除
    if output_path.exists():
        output_path.unlink()

    ext = output_path.suffix.lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(str(output_path), fourcc, frame_rate, (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频文件：{output_path}")

    try:
        for t in range(t_num):
            frame_rgb = np.clip(video4d[t], 0.0, 1.0)
            frame_uint8 = (frame_rgb * 255.0).round().astype(np.uint8)

            if frame_uint8.shape[0] != h or frame_uint8.shape[1] != w or frame_uint8.shape[2] != 3:
                raise RuntimeError(f"第 {t + 1} 帧尺寸异常，无法写入。")

            # OpenCV 写入需要 BGR 顺序
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"视频写入完成：{output_path}")


# =====================================================================
# 程序入口
# =====================================================================
if __name__ == "__main__":
    # ================== 参数设置 ==================
    input_video = "EP08_04.avi"       # 输入微表情视频
    output_video = "enhanced_EP19_03f.avi"    # 输出增强视频

    params = Params(
        tau=8,
        low_freq=0.5,
        high_freq=5.0,
        num_levels=4,
        target_r=0.11,
        alpha0=50,
        alpha_min=1,
        alpha_max=80,
        beta=0.1,
        max_iter=100,
        tol=1e-3,
        spatial_high_freq_threshold=0.15,
        k=1,
        epsilon=1e-8,
    )

    enhanced_video, alpha_opt, log_table = adaptive_dynamic_weak_texture_amplification(
        input_video, output_video, params
    )
