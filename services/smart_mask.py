from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


class SmartMaskError(RuntimeError):
    pass


def _ensure_gray(roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 3:
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_subtitle_mask(
    roi: np.ndarray,
    strength: int = 50,
    outline_px: int = 2,
) -> np.ndarray:
    if roi is None or roi.size == 0:
        raise SmartMaskError("Empty ROI")

    strength = max(0, min(100, int(strength)))
    gray = _ensure_gray(roi)
    height, width = gray.shape[:2]

    kernel_size = max(3, (min(height, width) // 120) * 2 + 1)
    kernel = _kernel(kernel_size)
    outline_kernel = _kernel(max(1, outline_px * 2 + 1))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

    mean_val = float(np.mean(blur))
    std_val = float(np.std(blur))
    if std_val < 1.0:
        std_val = 1.0

    bright_mask = cv2.threshold(
        tophat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    dark_mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    combined = cv2.bitwise_or(bright_mask, dark_mask)

    if combined.sum() == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    close_width = max(3, min(width // 4, 51))
    close_height = max(3, min(height // 8, 15))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_width | 1, close_height | 1)
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
    combined = cv2.dilate(combined, kernel, iterations=1)

    min_area = max(4, int((height * width) * 0.0004))
    max_area = int((height * width) * 0.45)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )

    clean = np.zeros_like(combined)
    kept = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = w / max(1, h)
        rel_area = area / max(1, height * width)
        if area < min_area or area > max_area:
            continue
        if rel_area > 0.15:
            continue
        if h < max(2, height // 40) or w < max(2, width // 30):
            continue
        if aspect < 0.03 or aspect > 40.0:
            continue
        clean[labels == label] = 255
        kept += 1

    if clean.sum() == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    combined = clean

    max_dilation = max(1, min(4, int(strength / 25)))
    dilation_strength = max(0, min(max_dilation, 3))
    if dilation_strength > 0:
        combined = cv2.dilate(
            combined, outline_kernel, iterations=dilation_strength
        )

    mask_coverage = float(np.count_nonzero(combined)) / float(
        combined.shape[0] * combined.shape[1]
    )
    if mask_coverage > 0.35:
        combined = clean
        if mask_coverage > 0.35 and combined.sum() > 0:
            combined = cv2.dilate(combined, outline_kernel, iterations=1)

    final_coverage = float(np.count_nonzero(combined)) / float(
        combined.shape[0] * combined.shape[1]
    )
    if final_coverage > 0.40:
        return np.zeros(gray.shape, dtype=np.uint8)

    return combined
