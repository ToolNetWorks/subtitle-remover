from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .media import mux_audio, probe_video
from .smart_inpaint import smart_remove


ProgressCallback = Callable[[dict], None]


class RemoveError(RuntimeError):
    pass


def _clamp_region(
    width: int,
    height: int,
    *,
    x1: int,
    x2: int,
    y: int,
    thickness: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(int(x1), int(x2)))
    right = min(width, max(int(x1), int(x2)))
    half = max(1, int(thickness) // 2)
    top = max(0, int(y) - half)
    bottom = min(height, int(y) + half)

    if right - left < 4:
        raise RemoveError("Subtitle region is too narrow")
    if bottom - top < 2:
        raise RemoveError("Subtitle region thickness is too small")
    return left, top, right, bottom


def _fast_fill(
    frame: np.ndarray,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    sample_gap: int,
) -> np.ndarray:
    height = frame.shape[0]
    sample_y = max(0, top - sample_gap)
    if sample_y == top and bottom + sample_gap < height:
        sample_y = bottom + sample_gap
    strip = frame[sample_y:sample_y + 1, left:right]
    if strip.size == 0:
        return frame

    frame[top:bottom, left:right] = np.repeat(
        strip,
        bottom - top,
        axis=0,
    )
    return frame


def _smooth_fill(
    frame: np.ndarray,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    sample_gap: int,
    feather: int,
) -> np.ndarray:
    height = frame.shape[0]
    top_y = max(0, top - sample_gap)
    bottom_y = min(height - 1, bottom + sample_gap)

    if top_y >= top and bottom_y <= bottom:
        return _fast_fill(
            frame,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            sample_gap=sample_gap,
        )

    top_pixels = frame[top_y, left:right].astype(np.float32)
    bottom_pixels = frame[bottom_y, left:right].astype(np.float32)
    strip_height = bottom - top

    generated = np.empty(
        (strip_height, right - left, 3),
        dtype=np.float32,
    )

    divisor = max(strip_height - 1, 1)
    for offset in range(strip_height):
        ratio = offset / divisor
        generated[offset] = (
            top_pixels * (1.0 - ratio)
            + bottom_pixels * ratio
        )

    generated = np.clip(generated, 0, 255).astype(np.uint8)

    if feather <= 0:
        frame[top:bottom, left:right] = generated
        return frame

    feather = min(int(feather), max(1, strip_height // 2))
    original = frame[top:bottom, left:right].copy()

    alpha = np.ones((strip_height, 1, 1), dtype=np.float32)
    for index in range(feather):
        value = (index + 1) / (feather + 1)
        alpha[index, 0, 0] = value
        alpha[-index - 1, 0, 0] = value

    blended = (
        generated.astype(np.float32) * alpha
        + original.astype(np.float32) * (1.0 - alpha)
    )
    frame[top:bottom, left:right] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)
    return frame


def _cover_fill(
    frame: np.ndarray,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    sample_gap: int,
    feather: int,
) -> np.ndarray:
    """BCC Wire-Remover style cover: multi-row median sample + weighted interp.

    CPU-only, no blur of whole ROI, no TELEA. Feather only blends ROI borders.
    """
    height = frame.shape[0]
    strip_h = bottom - top
    if strip_h <= 0 or right - left <= 0:
        return frame
    sample_depth = max(4, min(12, max(4, strip_h // 8)))
    gap = max(1, int(sample_gap))
    # Top sample band: rows above ROI
    top_start = max(0, top - gap - sample_depth)
    top_end = max(0, top - gap)
    # Bottom sample band: rows below ROI
    bot_start = min(height, bottom + gap)
    bot_end = min(height, bottom + gap + sample_depth)
    roi_w = right - left
    if top_end - top_start >= 2:
        top_band = frame[top_start:top_end, left:right].astype(np.float32)
        top_sample = np.median(top_band, axis=0)
    else:
        top_sample = frame[max(0, top - 1), left:right].astype(np.float32)
    if bot_end - bot_start >= 2:
        bot_band = frame[bot_start:bot_end, left:right].astype(np.float32)
        bot_sample = np.median(bot_band, axis=0)
    else:
        bot_sample = frame[min(height - 1, bottom), left:right].astype(np.float32)
    # Fallback to single-row if bands invalid
    generated = np.empty((strip_h, roi_w, 3), dtype=np.float32)
    divisor = max(strip_h - 1, 1)
    for offset in range(strip_h):
        ratio = offset / divisor
        generated[offset] = top_sample * (1.0 - ratio) + bot_sample * ratio
    generated = np.clip(generated, 0, 255).astype(np.uint8)
    if feather <= 0:
        frame[top:bottom, left:right] = generated
        return frame
    feather = min(int(feather), max(1, strip_h // 2))
    original = frame[top:bottom, left:right].copy()
    alpha = np.ones((strip_h, 1, 1), dtype=np.float32)
    for index in range(feather):
        value = (index + 1) / (feather + 1)
        alpha[index, 0, 0] = value
        alpha[-index - 1, 0, 0] = value
    blended = (
        generated.astype(np.float32) * alpha
        + original.astype(np.float32) * (1.0 - alpha)
    )
    frame[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame


def remove_strip(
    frame: np.ndarray,
    *,
    x1: int,
    x2: int,
    y: int,
    thickness: int,
    feather: int = 8,
    sample_gap: int = 4,
    mode: str = "smooth",
    mask_strength: int = 50,
) -> tuple[np.ndarray, Optional[SmartMaskResult]]:
    if frame is None or frame.size == 0:
        raise RemoveError("Empty video frame")
    if mode not in {"fast", "smooth", "smart", "cover"}:
        raise RemoveError("mode must be fast, smooth, smart, or cover")

    height, width = frame.shape[:2]
    left, top, right, bottom = _clamp_region(
        width,
        height,
        x1=x1,
        x2=x2,
        y=y,
        thickness=thickness,
    )

    if mode == "fast":
        return _fast_fill(
            frame,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            sample_gap=max(1, int(sample_gap)),
        ), None

    if mode == "smooth":
        return _smooth_fill(
            frame,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            sample_gap=max(1, int(sample_gap)),
            feather=max(0, int(feather)),
        ), None

    if mode == "cover":
        return _cover_fill(
            frame,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            sample_gap=max(1, int(sample_gap)),
            feather=max(0, int(feather)),
        ), None

    return smart_remove(
        frame,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        mask_strength=max(0, min(100, int(mask_strength))),
    )


def get_chunk_seconds() -> int:
    try:
        return max(30, int(os.getenv("SUBTITLE_CHUNK_SECONDS", "300")))
    except ValueError:
        return 300


def calc_chunks(total_frames: int, chunk_frames: int) -> list[dict]:
    if total_frames <= 0 or chunk_frames <= 0:
        return [{"index": 0, "start_frame": 0, "end_frame": max(0, total_frames - 1)}]
    chunks = []
    idx = 0
    start = 0
    while start < total_frames:
        end = min(total_frames - 1, start + chunk_frames - 1)
        chunks.append({"index": idx, "start_frame": start, "end_frame": end})
        idx += 1
        start = end + 1
    return chunks


def validate_chunk_file(path: Path, expected_frames: int, width: int, height: int) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        meta = probe_video(path)
    except Exception:
        return False
    if meta["width"] != width or meta["height"] != height:
        return False
    # Allow 5% frame tolerance for codec rounding
    if expected_frames > 0 and abs(meta["total_frames"] - expected_frames) > max(2, expected_frames * 0.05 + 2):
        # Fallback: try cv2 count
        cap = cv2.VideoCapture(str(path))
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        if abs(n - expected_frames) > max(2, expected_frames * 0.05 + 2):
            return False
    return True


def compute_roi_mae(input_path: Path, output_path: Path, *, x1: int, x2: int, y: int, thickness: int, fractions=(0.10, 0.25, 0.50, 0.75, 0.90)) -> dict:
    from .media import probe_video as _probe
    meta = _probe(input_path)
    w, h = meta["width"], meta["height"]
    left = max(0, min(int(x1), int(x2)))
    right = min(w, max(int(x1), int(x2)))
    half = max(1, int(thickness) // 2)
    top = max(0, int(y) - half)
    bottom = min(h, int(y) + half)
    cap_in = cv2.VideoCapture(str(input_path))
    cap_out = cv2.VideoCapture(str(output_path))
    total = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT) or meta["total_frames"] or 0)
    samples = {}
    for frac in fractions:
        idx = min(max(0, total - 1), int(total * frac))
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_out.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok1, f1 = cap_in.read()
        ok2, f2 = cap_out.read()
        if not ok1 or not ok2:
            continue
        roi_in = f1[top:bottom, left:right].astype(float)
        roi_out = f2[top:bottom, left:right].astype(float)
        roi_mae = float(np.mean(np.abs(roi_in - roi_out))) if roi_in.size else 0.0
        mask = np.ones(f1.shape[:2], dtype=bool)
        mask[top:bottom, left:right] = False
        out_mae = float(np.mean(np.abs(f1.astype(float)[mask] - f2.astype(float)[mask])))
        samples[f"{int(frac*100)}%"] = {"roi_mae": round(roi_mae, 2), "outside_mae": round(out_mae, 3)}
    cap_in.release()
    cap_out.release()
    return samples


def _process_single(
    *,
    input_path: Path,
    work_video_path: Path,
    output_path: Path,
    x1: int,
    x2: int,
    y: int,
    thickness: int,
    feather: int,
    sample_gap: int,
    mode: str,
    mask_strength: int = 50,
    progress_callback: ProgressCallback,
    base_processed: int = 0,
    base_total: int = 0,
    report_scale: tuple[float, float] = (0.0, 95.0),
) -> dict:
    meta = probe_video(input_path)
    width = meta["width"]
    height = meta["height"]
    fps = meta["fps"]
    total_frames = meta["total_frames"]

    _clamp_region(
        width,
        height,
        x1=x1,
        x2=x2,
        y=y,
        thickness=thickness,
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RemoveError("Cannot open input video")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(work_video_path),
        fourcc,
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RemoveError("Cannot create work video")

    started = time.time()
    processed = 0
    last_report = 0.0
    smart_stats = {
        "frames_with_mask": 0,
        "frames_without_mask": 0,
        "unreliable_frames": 0,
        "total_mask_pixels": 0,
        "average_mask_coverage": 0.0,
        "max_mask_coverage": 0.0,
    }
    mask_coverages = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            cleaned, mask_result = remove_strip(
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

            if mask_result is not None:
                smart_stats["frames_with_mask"] += mask_result.stats.get("frames_with_mask", 0)
                smart_stats["frames_without_mask"] += mask_result.stats.get("frames_without_mask", 0)
                smart_stats["unreliable_frames"] += mask_result.stats.get("unreliable_frames", 0)
                if mask_result.had_mask:
                    smart_stats["total_mask_pixels"] += int(np.count_nonzero(mask_result.mask))
                    mask_coverages.append(mask_result.coverage)
                    if mask_result.coverage > smart_stats["max_mask_coverage"]:
                        smart_stats["max_mask_coverage"] = mask_result.coverage

            writer.write(cleaned)
            processed += 1

            now = time.time()
            if now - last_report < 0.5:
                continue
            last_report = now

            ratio = (
                min(1.0, processed / total_frames)
                if total_frames > 0
                else 0.0
            )
            elapsed = max(0.0, now - started)
            eta = (
                elapsed / ratio - elapsed
                if ratio > 0.001
                else None
            )
            lo, hi = report_scale
            scaled = lo + ratio * (hi - lo)
            progress_callback({
                "status": "processing",
                "progress": round(scaled, 1),
                "processed_frames": base_processed + processed,
                "total_frames": base_total or total_frames,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "message": "Đang xoá subtitle...",
            })
    finally:
        capture.release()
        writer.release()

    if processed <= 0:
        raise RemoveError("No frames were processed")

    if mask_coverages:
        smart_stats["average_mask_coverage"] = round(float(np.mean(mask_coverages)), 4)

    # Single-path only does audio mux here; chunk path muxes once at the end.
    if base_total == 0:
        progress_callback({
            "status": "processing",
            "progress": 97.0,
            "processed_frames": processed,
            "total_frames": total_frames,
            "message": "Đang ghép audio gốc...",
        })
        mux_audio(
            processed_video=work_video_path,
            source_video=input_path,
            output_path=output_path,
        )

    elapsed = max(0.0, time.time() - started)
    result = {
        "processed_frames": processed,
        "total_frames": total_frames,
        "elapsed_seconds": round(elapsed, 1),
        "width": width,
        "height": height,
        "fps": fps,
        "duration": meta["duration"],
    }
    if mode == "smart":
        result["smart_stats"] = smart_stats
    else:
        result["smart_stats"] = smart_stats
    return result


def _process_frame_range(
    *,
    input_path: Path,
    chunk_path: Path,
    start_frame: int,
    end_frame: int,
    x1: int,
    x2: int,
    y: int,
    thickness: int,
    feather: int,
    sample_gap: int,
    mode: str,
    mask_strength: int,
    width: int,
    height: int,
    fps: float,
    progress_callback: ProgressCallback,
    base_processed: int,
    base_total: int,
    report_lo: float,
    report_hi: float,
) -> dict:
    expected = end_frame - start_frame + 1
    if validate_chunk_file(chunk_path, expected, width, height):
        return {"skipped": True, "processed": expected, "smart": None}
    last_error: Exception | None = None
    for attempt in range(3):  # initial + 2 retries
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RemoveError("Cannot open input video")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(chunk_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RemoveError("Cannot create chunk video")
            smart = {"frames_with_mask": 0, "frames_without_mask": 0, "unreliable_frames": 0,
                     "total_mask_pixels": 0, "coverages": []}
            done = 0
            started = time.time()
            last_report = 0.0
            try:
                for idx in range(start_frame, end_frame + 1):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RemoveError(f"Missing frame {idx} in chunk")
                    cleaned, mask_result = remove_strip(
                        frame, x1=x1, x2=x2, y=y, thickness=thickness,
                        feather=feather, sample_gap=sample_gap, mode=mode,
                        mask_strength=mask_strength)
                    if mask_result is not None:
                        smart["frames_with_mask"] += mask_result.stats.get("frames_with_mask", 0)
                        smart["frames_without_mask"] += mask_result.stats.get("frames_without_mask", 0)
                        smart["unreliable_frames"] += mask_result.stats.get("unreliable_frames", 0)
                        if mask_result.had_mask:
                            smart["total_mask_pixels"] += int(np.count_nonzero(mask_result.mask))
                            smart["coverages"].append(mask_result.coverage)
                    writer.write(cleaned)
                    done += 1
                    now = time.time()
                    if now - last_report >= 0.5:
                        last_report = now
                        ratio = done / max(1, expected)
                        scaled = report_lo + ratio * (report_hi - report_lo)
                        progress_callback({
                            "status": "processing",
                            "progress": round(scaled, 1),
                            "processed_frames": base_processed + done,
                            "total_frames": base_total,
                            "message": "Đang xoá subtitle...",
                        })
            finally:
                cap.release()
                writer.release()
            if done != expected:
                raise RemoveError(f"Chunk incomplete: {done}/{expected}")
            if not validate_chunk_file(chunk_path, expected, width, height):
                raise RemoveError("Chunk validation failed")
            return {"skipped": False, "processed": done, "smart": smart}
        except Exception as exc:
            last_error = exc
            try:
                chunk_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt >= 2:
                raise RemoveError(f"chunk_failed [{start_frame}-{end_frame}]: {exc}") from exc
    raise RemoveError(f"chunk_failed: {last_error}")


def process_video(
    *,
    input_path: Path,
    work_video_path: Path,
    output_path: Path,
    x1: int,
    x2: int,
    y: int,
    thickness: int,
    feather: int,
    sample_gap: int,
    mode: str,
    mask_strength: int = 50,
    progress_callback: ProgressCallback,
    job_dir: Path | None = None,
) -> dict:
    from .media import concat_videos
    meta = probe_video(input_path)
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    total_frames, duration = meta["total_frames"], meta["duration"]
    _clamp_region(width, height, x1=x1, x2=x2, y=y, thickness=thickness)
    if mode not in {"fast", "smooth", "smart", "cover"}:
        raise RemoveError("mode must be fast, smooth, smart, or cover")
    chunk_seconds = get_chunk_seconds()
    chunk_frames = max(1, int(round(fps * chunk_seconds)))
    # Single path
    if total_frames <= chunk_frames or (duration > 0 and duration <= chunk_seconds):
        result = _process_single(
            input_path=input_path, work_video_path=work_video_path, output_path=output_path,
            x1=x1, x2=x2, y=y, thickness=thickness, feather=feather,
            sample_gap=sample_gap, mode=mode, mask_strength=mask_strength,
            progress_callback=progress_callback)
        result["chunks"] = {"total_chunks": 1, "current_chunk": 1, "chunk_seconds": chunk_seconds}
        _attach_quality(result, input_path, output_path, x1=x1, x2=x2, y=y, thickness=thickness, mode=mode)
        return result
    # Chunk path
    base = job_dir or output_path.parent
    chunks_dir = base / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "chunks.json"
    specs = calc_chunks(total_frames, chunk_frames)
    # Load existing manifest for resume
    manifest = {f"{c['start_frame']}-{c['end_frame']}": c for c in specs}
    if manifest_path.exists():
        try:
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in saved:
                key = f"{entry.get('start_frame')}-{entry.get('end_frame')}"
                if key in manifest and entry.get("status") == "done":
                    chunk_p = chunks_dir / f"chunk-{entry.get('index', 0):04d}.mp4"
                    exp = entry["end_frame"] - entry["start_frame"] + 1
                    if validate_chunk_file(chunk_p, exp, width, height):
                        manifest[key]["status"] = "done"
        except Exception:
            pass
    started = time.time()
    agg_smart = {"frames_with_mask": 0, "frames_without_mask": 0, "unreliable_frames": 0,
                 "total_mask_pixels": 0, "average_mask_coverage": 0.0, "max_mask_coverage": 0.0}
    coverages: list[float] = []
    done_total = 0
    # Count already-done resume
    for spec in specs:
        key = f"{spec['start_frame']}-{spec['end_frame']}"
        chunk_p = chunks_dir / f"chunk-{spec['index']:04d}.mp4"
        exp = spec["end_frame"] - spec["start_frame"] + 1
        if manifest.get(key, {}).get("status") == "done" and validate_chunk_file(chunk_p, exp, width, height):
            done_total += exp
    for spec in specs:
        idx, sf, ef = spec["index"], spec["start_frame"], spec["end_frame"]
        exp = ef - sf + 1
        chunk_p = chunks_dir / f"chunk-{idx:04d}.mp4"
        lo = (done_total / max(1, total_frames)) * 90.0
        hi = ((done_total + exp) / max(1, total_frames)) * 90.0
        def _cb(d, _lo=lo, _hi=hi):
            progress_callback({
                **d,
                "current_chunk": idx + 1,
                "total_chunks": len(specs),
                "message": f"Chunk {idx+1}/{len(specs)} — Đang xoá subtitle...",
            })
        try:
            res = _process_frame_range(
                input_path=input_path, chunk_path=chunk_p, start_frame=sf, end_frame=ef,
                x1=x1, x2=x2, y=y, thickness=thickness, feather=feather,
                sample_gap=sample_gap, mode=mode, mask_strength=mask_strength,
                width=width, height=height, fps=fps, progress_callback=_cb,
                base_processed=done_total, base_total=total_frames,
                report_lo=lo, report_hi=hi)
        except Exception as exc:
            progress_callback({"status": "failed", "failed_chunk": idx, "error": str(exc)})
            raise
        done_total += exp
        if res.get("smart"):
            s = res["smart"]
            agg_smart["frames_with_mask"] += s["frames_with_mask"]
            agg_smart["frames_without_mask"] += s["frames_without_mask"]
            agg_smart["unreliable_frames"] += s["unreliable_frames"]
            agg_smart["total_mask_pixels"] += s["total_mask_pixels"]
            coverages.extend(s["coverages"])
            if s["coverages"]:
                agg_smart["max_mask_coverage"] = max(agg_smart["max_mask_coverage"], max(s["coverages"]))
        # Persist manifest
        try:
            entries = []
            for c in specs:
                cp = chunks_dir / f"chunk-{c['index']:04d}.mp4"
                e2 = c["end_frame"] - c["start_frame"] + 1
                st = "done" if (c["index"] <= idx and validate_chunk_file(cp, e2, width, height)) else "waiting"
                if c["index"] == idx + 1:
                    st = "waiting"
                entries.append({**c, "status": st, "path": str(cp)})
            manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except Exception:
            pass
        progress_callback({
            "status": "processing", "progress": round((done_total / max(1, total_frames)) * 90.0, 1),
            "processed_frames": done_total, "total_frames": total_frames,
            "current_chunk": idx + 1, "total_chunks": len(specs),
            "message": f"Chunk {idx+1}/{len(specs)} xong.",
        })
    if coverages:
        agg_smart["average_mask_coverage"] = round(float(np.mean(coverages)), 4)
    progress_callback({"status": "processing", "progress": 92.0, "processed_frames": total_frames,
                       "total_frames": total_frames, "message": "Đang ghép các chunk..."})
    chunk_files = [chunks_dir / f"chunk-{c['index']:04d}.mp4" for c in specs]
    concat_tmp = base / "concatenated-video.mp4"
    concat_videos(chunk_files, concat_tmp)
    progress_callback({"status": "processing", "progress": 96.0, "message": "Đang ghép audio gốc..."})
    mux_audio(processed_video=concat_tmp, source_video=input_path, output_path=output_path)
    # Boundary + duration validation (no duplicate/missing): check total frames of concat
    try:
        out_meta = probe_video(output_path)
    except Exception as exc:
        raise RemoveError(f"concat_failed: {exc}") from exc
    if abs(out_meta["total_frames"] - total_frames) > max(3, int(total_frames * 0.02)):
        raise RemoveError(f"concat_failed: frame mismatch {out_meta['total_frames']} vs {total_frames}")
    elapsed = max(0.0, time.time() - started)
    result = {"processed_frames": total_frames, "total_frames": total_frames,
              "elapsed_seconds": round(elapsed, 1), "width": width, "height": height,
              "fps": fps, "duration": meta["duration"],
              "smart_stats": agg_smart,
              "chunks": {"total_chunks": len(specs), "current_chunk": len(specs),
                         "chunk_seconds": chunk_seconds,
                         "chunk_frames": chunk_frames},
              "output_duration": out_meta["duration"]}
    _attach_quality(result, input_path, output_path, x1=x1, x2=x2, y=y, thickness=thickness, mode=mode)
    # Cleanup temp chunks after success (keep per TTL policy: delete chunk files + concat tmp)
    try:
        for cf in chunk_files:
            cf.unlink(missing_ok=True)
        concat_tmp.unlink(missing_ok=True)
        try:
            chunks_dir.rmdir()
        except Exception:
            pass
    except Exception:
        pass
    return result


def _attach_quality(result: dict, input_path: Path, output_path: Path, *, x1: int, x2: int, y: int, thickness: int, mode: str) -> None:
    try:
        samples = compute_roi_mae(input_path, output_path, x1=x1, x2=x2, y=y, thickness=thickness)
    except Exception:
        return
    if samples:
        result["quality"] = samples
        if mode in {"fast", "smooth", "cover"}:
            maes = [v["roi_mae"] for v in samples.values()]
            if maes and max(maes) < 1.0:
                result["warning_code"] = "processing_no_effect"
                result["warning_message"] = "Video output gần như không thay đổi trong vùng subtitle (processing_no_effect)."
