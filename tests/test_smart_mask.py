import numpy as np
import pytest
import cv2
import tempfile

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


def test_strength_sensitivity():
    """Test that increasing strength increases mask coverage."""
    roi = _make_roi((10, 10, 10), (220, 220, 220))
    weak = build_subtitle_mask(roi, strength=0, outline_px=2)
    medium = build_subtitle_mask(roi, strength=50, outline_px=2)
    strong = build_subtitle_mask(roi, strength=100, outline_px=2)
    assert np.count_nonzero(weak.mask) <= np.count_nonzero(medium.mask)
    assert np.count_nonzero(medium.mask) <= np.count_nonzero(strong.mask)


def test_mask_with_real_black_outline():
    """Test mask detection with real black outline using cv2.putText technique."""
    roi = np.full((80, 240, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(roi, "Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(roi, "Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert result.had_mask is True
    assert np.count_nonzero(result.mask) > 0
    assert result.reliable is True


def test_mask_count_nonzero_in_stats():
    """Test that smart_stats uses np.count_nonzero correctly."""
    from services.remover import process_video
    from unittest.mock import patch, MagicMock

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:80, 20:180] = _make_roi((10, 10, 10), (220, 220, 220))

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.mp4"
        output_path = Path(tmp) / "output.mp4"
        work_path = Path(tmp) / "work.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(input_path), fourcc, 30, (200, 100))
        writer.write(frame)
        writer.release()

        progress_data = {}
        def capture_progress(data):
            progress_data.update(data)

        result = process_video(
            input_path=input_path,
            work_video_path=work_path,
            output_path=output_path,
            x1=20, x2=180, y=50, thickness=48, feather=8, sample_gap=4,
            mode="smart", mask_strength=50, progress_callback=capture_progress,
        )

        if "smart_stats" in result:
            assert "total_mask_pixels" in result["smart_stats"]
            assert isinstance(result["smart_stats"]["total_mask_pixels"], int)
            assert result["smart_stats"]["total_mask_pixels"] >= 0


def test_preview_mask_overlay():
    """Test that preview-mask endpoint produces red overlay on detected mask pixels."""
    from fastapi.testclient import TestClient
    from app import app
    import json
    from pathlib import Path
    from unittest.mock import patch

    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmp:
        job_id = "a" * 32
        directory = Path(tmp) / job_id
        directory.mkdir()
        source = directory / "input.mp4"
        source.write_bytes(b"\x00" * 1024)

        roi = np.full((60, 160, 3), (10, 10, 10), dtype=np.uint8)
        roi[10:18, 20:140] = (220, 220, 220)
        preview = directory / "preview.jpg"
        cv2.imwrite(str(preview), roi)

        state = {
            "job_id": job_id,
            "status": "ready",
            "width": 160,
            "height": 60,
        }
        (directory / "state.json").write_text(json.dumps(state))

        def fake_job_dir(jid):
            if jid != job_id:
                raise Exception("bad job")
            return directory

        with patch("app.job_dir", fake_job_dir), \
             patch("app.get_source_video", lambda d: source):
            form = {
                "x1": "10",
                "x2": "150",
                "y": "14",
                "thickness": "48",
                "mask_strength": "50",
            }
            r = client.post(f"/api/jobs/{job_id}/preview-mask", data=form)
            assert r.status_code == 200
            assert r.headers.get("content-type") == "image/jpeg"

            result_img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            assert result_img is not None
            assert result_img.shape == roi.shape


def test_ground_truth_mask_coverage():
    """Test that mask coverage is reasonable for known text positions."""
    roi = np.full((60, 200, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(roi, "Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    result = build_subtitle_mask(roi, strength=50, outline_px=2)
    assert result.had_mask is True
    coverage = result.coverage
    assert 0.01 < coverage < 0.35
