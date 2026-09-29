from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from services.media import extract_frame, probe_video
from services.remover import _clamp_region, remove_strip, process_video
from services.smart_inpaint import smart_remove
from services.smart_mask import build_subtitle_mask


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/var/lib/subtitle-remover/jobs")
MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
JOB_TTL_SECONDS = 6 * 60 * 60

app = FastAPI(title="Subtitle Remover", version="1.0.0")


def job_dir(job_id: str) -> Path:
    if not job_id or len(job_id) != 32:
        raise HTTPException(404, "Job không hợp lệ")
    if any(char not in "0123456789abcdef" for char in job_id.lower()):
        raise HTTPException(404, "Job không hợp lệ")
    return DATA_DIR / job_id


def write_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(404, "Job không tồn tại")
    return json.loads(path.read_text(encoding="utf-8"))


def save_upload(upload: UploadFile, destination: Path) -> None:
    written = 0
    with destination.open("wb") as output:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_VIDEO_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "Video vượt quá 4 GB")
            output.write(chunk)

    if written <= 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Video rỗng")


def get_source_video(directory: Path) -> Path:
    candidates = list(directory.glob("input.*"))
    if not candidates:
        raise HTTPException(404, "Không tìm thấy video")
    return candidates[0]


def patch_state(job_id: str, changes: dict) -> None:
    directory = job_dir(job_id)
    state_path = directory / "state.json"
    try:
        state = read_json(state_path)
    except HTTPException:
        state = {"job_id": job_id}
    state.update(changes)
    write_json(state_path, state)


def process_worker(
    job_id: str,
    *,
    x1: int,
    x2: int,
    y: int,
    thickness: int,
    feather: int,
    sample_gap: int,
    mode: str,
    mask_strength: int = 50,
) -> None:
    directory = job_dir(job_id)
    source = get_source_video(directory)
    work_video = directory / "processed-no-audio.mp4"
    output = directory / "output.mp4"

    try:
        patch_state(job_id, {
            "status": "processing",
            "progress": 0,
            "message": "Đang chuẩn bị...",
            "started_at": time.time(),
            "error": None,
        })

        def report(data: dict) -> None:
            patch_state(job_id, data)

        result = process_video(
            input_path=source,
            work_video_path=work_video,
            output_path=output,
            x1=x1,
            x2=x2,
            y=y,
            thickness=thickness,
            feather=feather,
            sample_gap=sample_gap,
            mode=mode,
            mask_strength=mask_strength,
            progress_callback=report,
        )

        patch_state(job_id, {
            **result,
            "status": "done",
            "progress": 100,
            "message": "Hoàn tất.",
            "completed_at": time.time(),
            "output_path": str(output),
        })
    except Exception as exc:
        patch_state(job_id, {
            "status": "failed",
            "progress": 0,
            "message": "Xử lý thất bại.",
            "error": str(exc),
            "completed_at": time.time(),
        })
    finally:
        work_video.unlink(missing_ok=True)


def cleanup_loop() -> None:
    while True:
        time.sleep(15 * 60)
        now = time.time()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for directory in DATA_DIR.iterdir():
            if not directory.is_dir():
                continue
            state_path = directory / "state.json"
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if state.get("status") == "processing":
                continue
            completed = float(
                state.get("completed_at")
                or state.get("created_at")
                or directory.stat().st_mtime
            )
            if now - completed <= JOB_TTL_SECONDS:
                continue
            shutil.rmtree(directory, ignore_errors=True)


@app.on_event("startup")
def on_startup() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=cleanup_loop, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "subtitle-remover", "version": "1.0.0"}


def prepare_job(job_id: str, source: Path) -> None:
    directory = job_dir(job_id)
    patch_state(job_id, {
        "status": "preparing",
        "progress": 0,
        "message": "Đang chuẩn bị preview...",
    })

    try:
        meta = probe_video(source)
        frame_path = directory / "preview.jpg"
        extract_frame(source, frame_path, min(1.0, max(0.0, meta["duration"] / 10)))
    except Exception as exc:
        patch_state(job_id, {
            "status": "failed",
            "progress": 0,
            "message": "Chuẩn bị preview thất bại.",
            "error": str(exc),
            "completed_at": time.time(),
        })
        return

    patch_state(job_id, {
        "status": "ready",
        "progress": 0,
        "message": "Sẵn sàng chọn vùng subtitle.",
        **meta,
    })


@app.post("/api/jobs")
def create_job(video: UploadFile = File(...)) -> dict:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Không hỗ trợ định dạng: {suffix or 'unknown'}")

    job_id = uuid.uuid4().hex
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / f"input{suffix}"

    try:
        save_upload(video, source)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    state = {
        "job_id": job_id,
        "status": "uploaded",
        "progress": 0,
        "message": "Đã nhận video.",
        "created_at": time.time(),
        "video_filename": video.filename,
    }
    write_json(directory / "state.json", state)

    threading.Thread(
        target=prepare_job,
        args=(job_id, source),
        daemon=True,
    ).start()

    return state


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return read_json(job_dir(job_id) / "state.json")


@app.get("/api/jobs/{job_id}/preview-image")
def preview_image(job_id: str):
    path = job_dir(job_id) / "preview.jpg"
    if not path.exists():
        raise HTTPException(404, "Preview chưa tồn tại")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/jobs/{job_id}/preview-mask")
def preview_mask(
    job_id: str,
    x1: int = Form(...),
    x2: int = Form(...),
    y: int = Form(...),
    thickness: int = Form(48),
    mask_strength: int = Form(50),
):
    directory = job_dir(job_id)
    source = get_source_video(directory)
    state = read_json(directory / "state.json")
    source_preview = directory / "preview.jpg"
    mask_preview = directory / "preview-mask.jpg"

    frame = cv2.imread(str(source_preview))
    if frame is None:
        meta = probe_video(source)
        extract_frame(
            source,
            source_preview,
            min(1.0, max(0.0, meta["duration"] / 10)),
        )
        frame = cv2.imread(str(source_preview))

    if frame is None:
        raise HTTPException(500, "Không đọc được preview frame")

    height, width = frame.shape[:2]
    left, top, right, bottom = _clamp_region(
        width,
        height,
        x1=x1,
        x2=x2,
        y=y,
        thickness=thickness,
    )

    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        raise HTTPException(400, "ROI rỗng")

    try:
        mask_result = build_subtitle_mask(
            roi, strength=mask_strength, outline_px=2, frame_height=height
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    mask = mask_result.mask
    overlay = roi.copy()
    overlay[mask == 255] = (0, 0, 255)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    frame[top:bottom, left:right] = roi
    cv2.imwrite(str(mask_preview), frame)
    return FileResponse(
        mask_preview,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/jobs/{job_id}/preview")
def create_preview(
    job_id: str,
    x1: int = Form(...),
    x2: int = Form(...),
    y: int = Form(...),
    thickness: int = Form(48),
    feather: int = Form(8),
    sample_gap: int = Form(4),
    mode: str = Form("smooth"),
    mask_strength: int = Form(50),
):
    directory = job_dir(job_id)
    source = get_source_video(directory)
    state = read_json(directory / "state.json")
    source_preview = directory / "preview.jpg"
    output_preview = directory / "preview-clean.jpg"

    frame = cv2.imread(str(source_preview))
    if frame is None:
        meta = probe_video(source)
        extract_frame(
            source,
            source_preview,
            min(1.0, max(0.0, meta["duration"] / 10)),
        )
        frame = cv2.imread(str(source_preview))

    if frame is None:
        raise HTTPException(500, "Không đọc được preview frame")

    try:
        cleaned, _ = remove_strip(
            frame,
            x1=x1,
            x2=x2,
            y=y,
            thickness=thickness,
            feather=feather,
            sample_gap=sample_gap,
            mode=mode,
            mask_strength=mask_strength,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    cv2.imwrite(str(output_preview), cleaned)
    return FileResponse(
        output_preview,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/jobs/{job_id}/process")
def start_process(
    job_id: str,
    x1: int = Form(...),
    x2: int = Form(...),
    y: int = Form(...),
    thickness: int = Form(48),
    feather: int = Form(8),
    sample_gap: int = Form(4),
    mode: str = Form("smooth"),
    mask_strength: int = Form(50),
) -> dict:
    directory = job_dir(job_id)
    state = read_json(directory / "state.json")

    if state.get("status") in {"queued", "processing"}:
        return state

    if mode not in {"fast", "smooth", "smart"}:
        raise HTTPException(400, "mode phải là fast, smooth, hoặc smart")
    if thickness < 2 or thickness > int(state["height"]):
        raise HTTPException(400, "Thickness không hợp lệ")
    if feather < 0 or feather > 100:
        raise HTTPException(400, "Feather không hợp lệ")
    if mask_strength < 0 or mask_strength > 100:
        raise HTTPException(400, "Mask strength không hợp lệ")

    output = directory / "output.mp4"
    output.unlink(missing_ok=True)

    patch_state(job_id, {
        "status": "queued",
        "progress": 0,
        "message": "Đã xếp hàng.",
        "settings": {
            "x1": x1,
            "x2": x2,
            "y": y,
            "thickness": thickness,
            "feather": feather,
            "sample_gap": sample_gap,
            "mode": mode,
            "mask_strength": mask_strength,
        },
    })

    threading.Thread(
        target=process_worker,
        kwargs={
            "job_id": job_id,
            "x1": x1,
            "x2": x2,
            "y": y,
            "thickness": thickness,
            "feather": feather,
            "sample_gap": sample_gap,
            "mode": mode,
            "mask_strength": mask_strength,
        },
        daemon=True,
    ).start()

    return read_json(directory / "state.json")


@app.get("/api/jobs/{job_id}/result")
def result_video(job_id: str):
    path = job_dir(job_id) / "output.mp4"
    if not path.exists():
        raise HTTPException(404, "Kết quả chưa tồn tại")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}/download")
def download_video(job_id: str):
    path = job_dir(job_id) / "output.mp4"
    if not path.exists():
        raise HTTPException(404, "Kết quả chưa tồn tại")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"subtitle-removed-{job_id[:8]}.mp4",
    )
