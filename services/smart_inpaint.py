from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .smart_mask import SmartMaskError, SmartMaskResult, build_subtitle_mask


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
) -> tuple[np.ndarray, SmartMaskResult]:
    if frame is None or frame.size == 0:
        raise SmartInpaintError("Empty frame")

    height, width = frame.shape[:2]
    left = max(0, min(int(left), width - 1))
    right = max(left + 1, min(int(right), width))
    top = max(0, min(int(top), height - 1))
    bottom = max(top + 1, min(int(bottom), height))

    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        mask_result = SmartMaskResult(
            mask=np.zeros((bottom - top, right - left), dtype=np.uint8),
            had_mask=False,
            reliable=True,
            coverage=0.0,
            stats={"frames_with_mask": 0, "frames_without_mask": 1, "unreliable_frames": 0},
        )
        return frame, mask_result

    try:
        mask_result = build_subtitle_mask(
            roi, strength=mask_strength, outline_px=outline_px
        )
    except SmartMaskError:
        empty_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        mask_result = SmartMaskResult(
            mask=empty_mask,
            had_mask=False,
            reliable=False,
            coverage=0.0,
            stats={"frames_with_mask": 0, "frames_without_mask": 0, "unreliable_frames": 1},
        )
        return frame, mask_result

    mask = mask_result.mask
    if mask is None or mask.sum() == 0:
        return frame, mask_result

    inpaint_radius = max(1, min(int(inpaint_radius), 5))
    try:
        inpainted = cv2.inpaint(
            roi, mask, inpaint_radius, cv2.INPAINT_TELEA
        )
    except cv2.error:
        return frame, mask_result

    result = frame.copy()
    result[top:bottom, left:right] = inpainted
    return result, mask_result
