import numpy as np
from pathlib import Path
from services.remover import (
    calc_chunks, _cover_fill, _smooth_fill, _fast_fill,
    remove_strip, validate_chunk_file, compute_roi_mae,
)


def test_default_mode_smooth_frontend():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert 'mode: "smooth"' in html
    assert '<option value="smooth" selected>Smooth</option>' in html
    assert '$("mode").value = "smooth"' in html


def test_smooth_changes_roi():
    frame = np.full((100, 200, 3), 120, dtype=np.uint8)
    frame[45:55, 50:150] = 255
    out, _ = remove_strip(frame.copy(), x1=50, x2=150, y=50, thickness=20, mode="smooth")
    mae = float(np.mean(np.abs(frame.astype(float) - out.astype(float))))
    assert mae > 1.0


def test_outside_roi_preserved_smooth():
    frame = np.full((100, 200, 3), 120, dtype=np.uint8)
    out, _ = remove_strip(frame.copy(), x1=50, x2=150, y=50, thickness=20, mode="smooth", feather=0)
    assert np.array_equal(frame[0:10], out[0:10])
    assert np.array_equal(frame[90:100], out[90:100])


def test_cover_changes_roi():
    frame = np.full((100, 200, 3), 120, dtype=np.uint8)
    frame[45:55, 50:150] = 255
    out, _ = remove_strip(frame.copy(), x1=50, x2=150, y=50, thickness=20, mode="cover")
    mae = float(np.mean(np.abs(frame.astype(float) - out.astype(float))))
    assert mae > 1.0


def test_cover_feather_edges():
    frame = np.full((60, 100, 3), 100, dtype=np.uint8)
    frame[20:40, 10:90] = 250  # fake subtitle block inside ROI
    no_feather, _ = remove_strip(frame.copy(), x1=10, x2=90, y=30, thickness=20, mode="cover", feather=0)
    with_feather, _ = remove_strip(frame.copy(), x1=10, x2=90, y=30, thickness=20, mode="cover", feather=8)
    assert not np.array_equal(no_feather, with_feather)


def test_chunk_calculation_two_chunks():
    chunks = calc_chunks(15441, 9000)
    assert len(chunks) == 2
    assert chunks[0] == {"index": 0, "start_frame": 0, "end_frame": 8999}
    assert chunks[1] == {"index": 1, "start_frame": 9000, "end_frame": 15440}


def test_chunk_exact_ranges():
    chunks = calc_chunks(18000, 9000)
    assert len(chunks) == 2
    assert chunks[1]["end_frame"] - chunks[1]["start_frame"] + 1 == 9000


def test_last_short_chunk():
    chunks = calc_chunks(10000, 9000)
    assert len(chunks) == 2
    assert chunks[1]["end_frame"] - chunks[1]["start_frame"] + 1 == 1000


def test_no_duplicate_boundary():
    chunks = calc_chunks(15441, 9000)
    assert chunks[1]["start_frame"] == chunks[0]["end_frame"] + 1


def test_smart_warning_logic():
    from app import _smart_warning
    res = {"processed_frames": 100, "smart_stats": {"frames_with_mask": 20}}
    w = _smart_warning(res, "smart")
    assert w["warning_code"] == "smart_low_detection"
    res2 = {"processed_frames": 100, "smart_stats": {"frames_with_mask": 80}}
    w2 = _smart_warning(res2, "smart")
    assert w2["warning_code"] is None


def test_upload_streaming_endpoints_exist():
    from app import app
    routes = {(r.path, tuple(r.methods or [])) for r in app.routes}
    assert any("/api/uploads/init" in p for p, _ in routes)
    assert any("/api/uploads/{job_id}/file" in p for p, _ in routes)


def test_legacy_upload_compat():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    r = client.post("/api/uploads/init")
    assert r.status_code == 400  # needs JSON, but route exists


def test_processing_no_effect_detection():
    # Identical input/output => warning
    import cv2, tempfile
    h, w = 120, 200
    frames = [np.full((h, w, 3), 100, dtype=np.uint8) for _ in range(10)]
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.mp4"
        out = Path(tmp) / "out.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(inp), fourcc, 10, (w, h))
        for f in frames:
            wr.write(f)
        wr.release()
        wr = cv2.VideoWriter(str(out), fourcc, 10, (w, h))
        for f in frames:
            wr.write(f)
        wr.release()
        samples = compute_roi_mae(inp, out, x1=10, x2=190, y=60, thickness=20)
        assert samples
        assert max(v["roi_mae"] for v in samples.values()) < 5.0
