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


def _scale_outline(base_px: int, frame_height: int) -> int:
    """Scale outline pixels based on full frame height."""
    if frame_height <= 0:
        frame_height = 1080
    scale = max(0.5, frame_height / 1080.0)
    return max(1, int(round(base_px * scale)))


def build_subtitle_mask(
    roi: np.ndarray,
    strength: int = 50,
    outline_px: int = 2,
    frame_height: int = 0,
) -> SmartMaskResult:
    if roi is None or roi.size == 0:
        raise SmartMaskError("Empty ROI")

    strength = max(0, min(100, int(strength)))
    gray = _ensure_gray(roi)
    height, width = gray.shape[:2]

    if frame_height <= 0:
        frame_height = height

    scaled_outline = _scale_outline(outline_px, frame_height)
    outline_kernel = _kernel(max(1, scaled_outline * 2 + 1))

    kernel_size = max(3, (min(height, width) // 120) * 2 + 1)
    kernel = _kernel(kernel_size)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

    # Get Otsu thresholds as baseline
    otsu_bright, _ = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_dark, _ = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adjust threshold based on strength
    # strength 0 -> threshold = Otsu * 1.5 (conservative)
    # strength 50 -> threshold = Otsu (baseline)
    # strength 100 -> threshold = Otsu * 0.5 (aggressive)
    strength_factor = strength / 100.0
    threshold_scale = 1.5 - strength_factor

    bright_thresh = max(8.0, float(otsu_bright) * threshold_scale)
    dark_thresh = max(8.0, float(otsu_dark) * threshold_scale)

    _, bright_mask = cv2.threshold(
        tophat, bright_thresh, 255, cv2.THRESH_BINARY
    )
    _, dark_mask = cv2.threshold(
        blackhat, dark_thresh, 255, cv2.THRESH_BINARY
    )

    combined = cv2.bitwise_or(bright_mask, dark_mask)

    if np.count_nonzero(combined) == 0:
        return SmartMaskResult(
            mask=np.zeros(gray.shape, dtype=np.uint8),
            had_mask=False,
            reliable=True,
            coverage=0.0,
            stats={"frames_with_mask": 0, "frames_without_mask": 1, "unreliable_frames": 0},
        )

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

    if np.count_nonzero(clean) == 0:
        return SmartMaskResult(
            mask=np.zeros(gray.shape, dtype=np.uint8),
            had_mask=False,
            reliable=True,
            coverage=0.0,
            stats={"frames_with_mask": 0, "frames_without_mask": 1, "unreliable_frames": 0},
        )

    combined = clean

    max_dilation = max(1, min(4, int(strength / 25)))
    dilation_strength = max(0, min(max_dilation, 3))
    if dilation_strength > 0:
        combined = cv2.dilate(
            combined, outline_kernel, iterations=dilation_strength
        )

    coverage = float(np.count_nonzero(combined)) / float(
        combined.shape[0] * combined.shape[1]
    )

    if coverage > 0.35:
        combined = clean
        coverage = float(np.count_nonzero(combined)) / float(
            combined.shape[0] * combined.shape[1]
        )

    final_coverage = float(np.count_nonzero(combined)) / float(
        combined.shape[0] * combined.shape[1]
    )

    reliable = final_coverage <= 0.40
    had_mask = np.count_nonzero(combined) > 0

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
