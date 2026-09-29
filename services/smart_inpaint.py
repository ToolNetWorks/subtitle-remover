from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .smart_mask import SmartMaskError, build_subtitle_mask


class SmartInpaintError(RuntimeError):
    pass


def smart_remove(
    frame: np.ndarray,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    mask_strength: int = 50,
    outline_px: int = 2,
    inpaint_radius: int = 3,
) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise SmartInpaintError("Empty frame")

    height, width = frame.shape[:2]
    left = max(0, min(int(left), width - 1))
    right = max(left + 1, min(int(right), width))
    top = max(0, min(int(top), height - 1))
    bottom = max(top + 1, min(int(bottom), height))

    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return frame

    try:
        mask = build_subtitle_mask(
            roi, strength=mask_strength, outline_px=outline_px
        )
    except SmartMaskError:
        return frame

    if mask is None or mask.sum() == 0:
        return frame

    mask_coverage = float(np.count_nonzero(mask)) / float(
        mask.shape[0] * mask.shape[1]
    )
    if mask_coverage > 0.40:
        return frame

    inpaint_radius = max(1, min(int(inpaint_radius), 5))
    try:
        inpainted = cv2.inpaint(
            roi, mask, inpaint_radius, cv2.INPAINT_TELEA
        )
    except cv2.error:
        return frame

    result = frame.copy()
    result[top:bottom, left:right] = inpainted
    return result
