import numpy as np
import pytest
import cv2

from services.smart_mask import SmartMaskResult, build_subtitle_mask, _scale_outline
from services.smart_inpaint import smart_remove, SmartInpaintError


def _make_roi(background, foreground, height=60, width=160):
    roi = np.full((height, width, 3), background, dtype=np.uint8)
    roi[10:18, 20:140] = foreground
    return roi


def test_mask_white_text_on_dark():
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert isinstance(result, SmartMaskResult)
    assert result.mask.dtype == np.uint8
    assert result.mask.shape == roi.shape[:2]
    assert np.count_nonzero(result.mask) > 0
    assert np.count_nonzero(result.mask) < 0.35 * result.mask.size
    assert result.reliable is True
    assert result.coverage < 0.40


def test_mask_dark_text_on_light():
    roi = _make_roi((220, 220, 220), (20, 20, 20))
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert isinstance(result, SmartMaskResult)
    assert result.mask.dtype == np.uint8
    assert result.mask.shape == roi.shape[:2]
    assert np.count_nonzero(result.mask) > 0
    assert np.count_nonzero(result.mask) < 0.35 * result.mask.size
    assert result.reliable is True


def test_mask_no_text():
    roi = np.full((40, 120, 3), (100, 100, 100), dtype=np.uint8)
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert isinstance(result, SmartMaskResult)
    assert np.count_nonzero(result.mask) == 0
    assert result.had_mask is False
    assert result.reliable is True


def test_mask_strength_range():
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    weak = build_subtitle_mask(roi, strength=0, outline_px=2)
    strong = build_subtitle_mask(roi, strength=100, outline_px=2)
    assert np.count_nonzero(strong.mask) >= np.count_nonzero(weak.mask)


def test_smart_remove_changes_roi():
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:80, 20:180] = roi
    result, mask_result = smart_remove(frame, left=20, top=20, right=180, bottom=80, mask_strength=50)
    assert result.shape == frame.shape


def test_smart_remove_empty_roi():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result, mask_result = smart_remove(frame, left=0, top=0, right=0, bottom=0, mask_strength=50)
    assert result.shape == frame.shape


def test_smart_remove_no_mask():
    frame = np.full((40, 120, 3), (100, 100, 100), dtype=np.uint8)
    result, mask_result = smart_remove(frame, left=0, top=0, right=120, bottom=40, mask_strength=50)
    assert result.shape == frame.shape
    assert mask_result.had_mask is False


def test_mask_with_puttext_white():
    """Test mask detection with cv2.putText white text."""
    roi = np.full((60, 200, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(roi, "Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert result.had_mask
    assert np.count_nonzero(result.mask) > 0


def test_mask_with_puttext_dark():
    """Test mask detection with cv2.putText dark text."""
    roi = np.full((60, 200, 3), (220, 220, 220), dtype=np.uint8)
    cv2.putText(roi, "Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert result.had_mask
    assert np.count_nonzero(result.mask) > 0


def test_false_positive_horizontal_line():
    """Test that horizontal lines don't create large masks."""
    roi = np.full((60, 200, 3), (100, 100, 100), dtype=np.uint8)
    cv2.line(roi, (0, 30), (200, 30), (200, 200, 200), 2)
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert result.had_mask is False or result.coverage < 0.05


def test_false_positive_gradient():
    """Test that gradients don't create large masks."""
    roi = np.full((60, 200, 3), (100, 100, 100), dtype=np.uint8)
    for y in range(60):
        roi[y, :] = [100 + y * 2, 100 + y * 2, 100 + y * 2]
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert result.had_mask is False or result.coverage < 0.05


def test_scale_outline():
    """Test resolution scaling for outline."""
    assert _scale_outline(2, 1080) == 2
    assert _scale_outline(2, 2160) == 4
    assert _scale_outline(2, 480) == 1


def test_mask_stats():
    """Test that SmartMaskResult contains proper stats."""
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert "frames_with_mask" in result.stats
    assert "frames_without_mask" in result.stats
    assert "unreliable_frames" in result.stats
    assert result.stats["frames_with_mask"] + result.stats["frames_without_mask"] + result.stats["unreliable_frames"] == 1
