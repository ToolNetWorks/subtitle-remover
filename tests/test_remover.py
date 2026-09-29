import numpy as np
import pytest

from services.remover import _clamp_region, _fast_fill, _smooth_fill, remove_strip, RemoveError


def test_clamp_region_normal():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    left, top, right, bottom = _clamp_region(200, 100, x1=50, x2=150, y=60, thickness=10)
    assert left == 50
    assert right == 150
    assert top == 55
    assert bottom == 65


def test_clamp_region_reorder_x():
    left, top, right, bottom = _clamp_region(200, 100, x1=150, x2=50, y=60, thickness=10)
    assert left == 50
    assert right == 150


def test_clamp_region_too_narrow():
    with pytest.raises(RemoveError):
        _clamp_region(200, 100, x1=50, x2=53, y=60, thickness=10)


def test_clamp_region_too_thin():
    with pytest.raises(RemoveError):
        _clamp_region(200, 100, x1=50, x2=150, y=0, thickness=1)


def test_fast_fill():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[0, :] = [255, 0, 0]
    result = _fast_fill(frame, left=0, top=2, right=10, bottom=8, sample_gap=2)
    assert result is frame
    expected = np.full((6, 10, 3), [255, 0, 0], dtype=np.uint8)
    assert np.array_equal(result[2:8, :], expected)


def test_smooth_fill():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[0, :] = [255, 0, 0]
    frame[9, :] = [0, 0, 255]
    result = _smooth_fill(frame, left=0, top=2, right=10, bottom=8, sample_gap=2, feather=0)
    assert result is frame
    top_pixel = result[2, 0]
    bottom_pixel = result[7, 0]
    assert np.array_equal(top_pixel, [255, 0, 0])
    assert np.array_equal(bottom_pixel, [0, 0, 255])


def test_remove_strip_invalid_mode():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(RemoveError):
        remove_strip(frame, x1=1, x2=8, y=5, thickness=4, mode="invalid")


def test_process_queued_double_submit_guard():
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        from unittest.mock import patch

        job_id = "a" * 32
        directory = Path(tmp) / job_id
        directory.mkdir()
        source = directory / "input.mp4"
        source.write_bytes(b"\x00" * 1024)

        state = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Đã xếp hàng.",
            "width": 320,
            "height": 240,
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
                "x2": "100",
                "y": "120",
                "thickness": "48",
                "feather": "8",
                "sample_gap": "4",
                "mode": "smooth",
            }
            r1 = client.post(f"/api/jobs/{job_id}/process", data=form)
            r2 = client.post(f"/api/jobs/{job_id}/process", data=form)
            assert r1.status_code == 200
            assert r2.status_code == 200
            body = r2.json()
            assert body["status"] == "queued"
