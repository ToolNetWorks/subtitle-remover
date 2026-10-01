from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from services.media import extract_frame, probe_video
from services.remover import _clamp_region, remove_strip, process_video
from services.smart_inpaint import smart_remove
from services.smart_mask import build_subtitle_mask


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/lib/subtitle-remover/jobs"))
MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
JOB_TTL_SECONDS = 6 * 60 * 60
UPLOAD_CHUNK_SIZE = 1024 * 1024
VALID_MODES = {"fast", "smooth", "smart", "cover"}
# created -> uploading -> uploaded -> preparing -> ready | failed
UPLOAD_RESUMABLE_STATUSES = {"created", "uploading", "uploaded", "failed"}

logger = logging.getLogger("subtitle-remover")

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


def _smart_warning(result: dict, mode: str) -> dict:
    if mode != "smart":
        return {"warning_code": None, "warning_message": None}
    stats = result.get("smart_stats") or {}
    processed = int(result.get("processed_frames") or 0)
    with_mask = int(stats.get("frames_with_mask") or 0)
    if processed <= 0:
        return {"warning_code": None, "warning_message": None}
    ratio = with_mask / max(1, processed)
    if ratio < 0.50:
        return {
            "warning_code": "smart_low_detection",
            "warning_message": "SMART phát hiện phụ đề ở quá ít frame. Hãy thử Smooth/Cover hoặc điều chỉnh vùng chọn.",
            "mask_ratio": round(ratio, 4),
        }
    return {"warning_code": None, "warning_message": None, "mask_ratio": round(ratio, 4)}


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
            job_dir=directory,
        )

        final = {
            **result,
            "status": "done",
            "progress": 100,
            "message": "Hoàn tất.",
            "completed_at": time.time(),
            "output_path": str(output),
        }
        # Preserve quality warning from pipeline; smart warning fills if absent.
        if not final.get("warning_code"):
            final.update(_smart_warning(result, mode))
        patch_state(job_id, final)
    except Exception as exc:
        msg = str(exc)
        code = "processing_failed"
        for candidate in ("chunk_failed", "concat_failed", "audio_mux_failed",
                          "processing_no_effect", "invalid_roi", "probe_failed"):
            if candidate in msg:
                code = candidate
                break
        patch_state(job_id, {
            "status": "failed",
            "progress": 0,
            "message": "Xử lý thất bại.",
            "error": msg,
            "error_code": code,
            "failed_chunk": getattr(exc, "args", [None])[0] if "chunk" in msg else None,
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


@app.post("/api/uploads/init")
async def init_upload(request: Request) -> dict:
    """Create job first, frontend streams raw bytes afterwards.

    Body JSON: {"filename": "clip.mp4", "filesize": 12345}
    Returns job_id + upload_url immediately (no ffprobe here).
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Body JSON không hợp lệ")

    if not isinstance(payload, dict):
        raise HTTPException(400, "Body JSON không hợp lệ")

    filename = str(payload.get("filename") or "")
    filesize = payload.get("filesize")

    # Guard: extension
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Không hỗ trợ định dạng: {suffix or 'unknown'}")

    # Guard: filesize (optional but validated when present)
    if filesize is not None:
        try:
            filesize = int(filesize)
        except (TypeError, ValueError):
            raise HTTPException(400, "Filesize không hợp lệ")
        if filesize <= 0:
            raise HTTPException(400, "Filesize không hợp lệ")
        if filesize > MAX_VIDEO_BYTES:
            raise HTTPException(413, "Video vượt quá 4 GB")

    job_id = uuid.uuid4().hex
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)

    state = {
        "job_id": job_id,
        "status": "created",
        "progress": 0,
        "message": "Đã tạo job, chờ upload.",
        "created_at": time.time(),
        "video_filename": Path(filename).name,
        "video_suffix": suffix,
        "expected_bytes": filesize,
    }
    write_json(directory / "state.json", state)
    logger.info("upload init job=%s file=%s size=%s", job_id, filename, filesize)

    return {
        **state,
        "upload_url": f"/api/uploads/{job_id}/file",
    }


@app.put("/api/uploads/{job_id}/file")
async def upload_stream(job_id: str, request: Request) -> dict:
    """Stream raw request body directly to disk, no multipart spool, no RAM hold.

    Client sends: PUT raw File bytes (Content-Type: application/octet-stream).
    Response is returned right after disk write completes; preview runs in background.
    """
    directory = job_dir(job_id)
    state_path = directory / "state.json"
    if not state_path.exists():
        raise HTTPException(404, "Job không tồn tại")
    state = read_json(state_path)

    # Guard: do not overwrite a job already past upload phase
    if state.get("status") not in UPLOAD_RESUMABLE_STATUSES:
        raise HTTPException(409, f"Job đã ở trạng thái {state.get('status')}, không thể upload lại")

    suffix = str(state.get("video_suffix") or "")
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "Job thiếu định dạng video, hãy init lại")

    # Guard: optional Content-Length pre-check (early return, no disk I/O)
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_VIDEO_BYTES:
                raise HTTPException(413, "Video vượt quá 4 GB")
            if int(content_length) <= 0:
                raise HTTPException(400, "Video rỗng")
        except ValueError:
            pass

    destination = directory / f"input{suffix}"
    patch_state(job_id, {
        "status": "uploading",
        "progress": 0,
        "message": "Đang nhận dữ liệu upload...",
        "upload_started_at": time.time(),
        "error": None,
    })

    t_start = time.monotonic()
    written = 0
    try:
        with destination.open("wb") as output:
            async for chunk in request.stream():
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_VIDEO_BYTES:
                    output.close()
                    destination.unlink(missing_ok=True)
                    patch_state(job_id, {
                        "status": "failed",
                        "message": "Upload thất bại.",
                        "error": "Video vượt quá 4 GB",
                        "completed_at": time.time(),
                    })
                    raise HTTPException(413, "Video vượt quá 4 GB")
                output.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        patch_state(job_id, {
            "status": "failed",
            "message": "Upload thất bại.",
            "error": str(exc),
            "completed_at": time.time(),
        })
        raise HTTPException(500, f"Ghi file thất bại: {exc}")

    stream_ms = (time.monotonic() - t_start) * 1000.0

    # Guard: empty body
    if written <= 0:
        destination.unlink(missing_ok=True)
        patch_state(job_id, {
            "status": "failed",
            "message": "Upload thất bại.",
            "error": "Video rỗng",
            "completed_at": time.time(),
        })
        raise HTTPException(400, "Video rỗng")

    logger.info(
        "upload stream complete job=%s bytes=%d stream_ms=%.1f",
        job_id, written, stream_ms,
    )

    new_state = {
        "status": "uploaded",
        "progress": 0,
        "message": "Đã nhận video.",
        "video_size": written,
        "uploaded_at": time.time(),
        "timing": {
            "receive_save_ms": round(stream_ms, 1),
        },
    }
    patch_state(job_id, new_state)

    # Preview runs in background — never inside upload request.
    threading.Thread(
        target=prepare_job,
        args=(job_id, destination),
        daemon=True,
    ).start()

    # Return uploaded snapshot (not re-read: prepare thread may have
    # already flipped state to preparing — polling covers that).
    return {**state, **new_state, "job_id": job_id}


@app.post("/api/jobs")
def create_job(video: UploadFile = File(...)) -> dict:
    """Legacy multipart upload (compat). Frontend mới dùng /api/uploads/* streaming."""
    t_handler_start = time.monotonic()
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Không hỗ trợ định dạng: {suffix or 'unknown'}")

    job_id = uuid.uuid4().hex
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / f"input{suffix}"

    try:
        t_save_start = time.monotonic()
        save_upload(video, source)
        save_ms = (time.monotonic() - t_save_start) * 1000.0
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    handler_ms = (time.monotonic() - t_handler_start) * 1000.0
    # NOTE: handler_ms excludes Starlette multipart spool time (full body parse
    # happens before this function runs) + Cloudflare edge->origin transfer.
    # That hidden latency is exactly what caused "100% stuck" via tunnel.
    logger.info(
        "legacy upload job=%s bytes=%d save_ms=%.1f handler_ms=%.1f",
        job_id, source.stat().st_size if source.exists() else -1, save_ms, handler_ms,
    )

    state = {
        "job_id": job_id,
        "status": "uploaded",
        "progress": 0,
        "message": "Đã nhận video.",
        "created_at": time.time(),
        "video_filename": video.filename,
        "timing": {
            "save_ms": round(save_ms, 1),
            "handler_ms": round(handler_ms, 1),
        },
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

    if mode not in {"fast", "smooth", "smart", "cover"}:
        raise HTTPException(400, "mode phải là fast, smooth, smart, hoặc cover")
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
