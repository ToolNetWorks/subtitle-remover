from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SmartMaskResult:
    mask: np.ndarray
    had_mask: bool
    reliable: bool
    coverage: float
    stats: dict


class SmartMaskError(RuntimeError):
    pass


def _ensure_gray(roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 3:
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _scale_outline(base_px: int, roi_height: int) -> int:
    """Scale outline pixels based on ROI height."""
    scale = max(0.5, roi_height / 1080.0)
    return max(1, int(round(base_px * scale)))


def build_subtitle_mask(
    roi: np.ndarray,
    strength: int = 50,
    outline_px: int = 2,
) -> SmartMaskResult:
    if roi is None or roi.size == 0:
        raise SmartMaskError("Empty ROI")

    strength = max(0, min(100, int(strength)))
    gray = _ensure_gray(roi)
    height, width = gray.shape[:2]

    # Scale outline based on resolution
    scaled_outline = _scale_outline(outline_px, height)
    outline_kernel = _kernel(max(1, scaled_outline * 2 + 1))

    # Adaptive kernel size based on ROI
    kernel_size = max(3, (min(height, width) // 120) * 2 + 1)
    kernel = _kernel(kernel_size)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

    mean_val = float(np.mean(blur))
    std_val = float(np.std(blur))
    if std_val < 1.0:
        std_val = 1.0

    # Adaptive threshold using Otsu
    _, bright_mask = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, dark_mask = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    combined = cv2.bitwise_or(bright_mask, dark_mask)

    if combined.sum() == 0:
        return SmartMaskResult(
            mask=np.zeros(gray.shape, dtype=np.uint8),
            had_mask=False,
            reliable=True,
            coverage=0.0,
            stats={"frames_with_mask": 0, "frames_without_mask": 1, "unreliable_frames": 0},
        )

    # Close gaps in mask
    close_width = max(3, min(width // 4, 51))
    close_height = max(3, min(height // 8, 15))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_width | 1, close_height | 1)
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
    combined = cv2.dilate(combined, kernel, iterations=1)

    # Component filtering
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
        return SmartMaskResult(
            mask=np.zeros(gray.shape, dtype=np.uint8),
            had_mask=False,
            reliable=True,
            coverage=0.0,
            stats={"frames_with_mask": 0, "frames_without_mask": 1, "unreliable_frames": 0},
        )

    combined = clean

    # Strength-based dilation (0-100 maps to 0-3 iterations)
    max_dilation = max(1, min(4, int(strength / 25)))
    dilation_strength = max(0, min(max_dilation, 3))
    if dilation_strength > 0:
        combined = cv2.dilate(
            combined, outline_kernel, iterations=dilation_strength
        )

    coverage = float(np.count_nonzero(combined)) / float(
        combined.shape[0] * combined.shape[1]
    )

    # Safety: if coverage is too high, try conservative mask
    if coverage > 0.35:
        combined = clean
        coverage = float(np.count_nonzero(combined)) / float(
            combined.shape[0] * combined.shape[1]
        )

    # Final safety check
    final_coverage = float(np.count_nonzero(combined)) / float(
        combined.shape[0] * combined.shape[1]
    )

    reliable = final_coverage <= 0.40
    had_mask = combined.sum() > 0

    if not reliable:
        return SmartMaskResult(
            mask=np.zeros(gray.shape, dtype=np.uint8),
            had_mask=False,
            reliable=False,
            coverage=final_coverage,
            stats={
                "frames_with_mask": 0,
                "frames_without_mask": 0,
                "unreliable_frames": 1,
            },
        )

    return SmartMaskResult(
        mask=combined,
        had_mask=had_mask,
        reliable=True,
        coverage=final_coverage,
        stats={
            "frames_with_mask": 1 if had_mask else 0,
            "frames_without_mask": 0 if had_mask else 1,
            "unreliable_frames": 0,
        },
    )
