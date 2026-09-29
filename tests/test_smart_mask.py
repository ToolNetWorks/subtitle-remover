import numpy as np
import pytest

from services.smart_mask import build_subtitle_mask, SmartMaskError
from services.smart_inpaint import smart_remove, SmartInpaintError


def _make_roi(background, foreground, height=60, width=160):
    roi = np.full((height, width, 3), background, dtype=np.uint8)
    roi[10:18, 20:140] = foreground
    return roi


def test_mask_white_text_on_dark():
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    mask = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert mask.dtype == np.uint8
    assert mask.shape == roi.shape[:2]
    assert np.count_nonzero(mask) > 0
    assert np.count_nonzero(mask) < 0.35 * mask.size


def test_mask_dark_text_on_light():
    roi = _make_roi((220, 220, 220), (20, 20, 20))
    mask = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert mask.dtype == np.uint8
    assert mask.shape == roi.shape[:2]
    assert np.count_nonzero(mask) > 0
    assert np.count_nonzero(mask) < 0.35 * mask.size


def test_mask_no_text():
    roi = np.full((40, 120, 3), (100, 100, 100), dtype=np.uint8)
    mask = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert mask.sum() == 0


def test_mask_strength_range():
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    weak = build_subtitle_mask(roi, strength=0, outline_px=2)
    strong = build_subtitle_mask(roi, strength=100, outline_px=2)
    assert strong.sum() >= weak.sum()


def test_smart_remove_changes_roi():
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:80, 20:180] = roi
    result = smart_remove(frame, left=20, top=20, right=180, bottom=80, mask_strength=50)
    assert result is not frame
    assert result.shape == frame.shape


def test_smart_remove_empty_roi():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = smart_remove(frame, left=0, top=0, right=0, bottom=0, mask_strength=50)
    assert result is frame


def test_smart_remove_no_mask():
    frame = np.full((40, 120, 3), (100, 100, 100), dtype=np.uint8)
    result = smart_remove(frame, left=0, top=0, right=120, bottom=40, mask_strength=50)
    assert result is frame
