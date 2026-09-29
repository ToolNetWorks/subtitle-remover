from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

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
) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise RemoveError("Empty video frame")
    if mode not in {"fast", "smooth", "smart"}:
        raise RemoveError("mode must be fast, smooth, or smart")

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
        )

    if mode == "smooth":
        return _smooth_fill(
            frame,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            sample_gap=max(1, int(sample_gap)),
            feather=max(0, int(feather)),
        )

    return smart_remove(
        frame,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        mask_strength=max(0, min(100, int(mask_strength))),
    )


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

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            cleaned = remove_strip(
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

            progress_callback({
                "status": "processing",
                "progress": round(ratio * 95.0, 1),
                "processed_frames": processed,
                "total_frames": total_frames,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "message": "Đang xoá subtitle...",
            })
    finally:
        capture.release()
        writer.release()

    if processed <= 0:
        raise RemoveError("No frames were processed")

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
    return {
        "processed_frames": processed,
        "total_frames": total_frames,
        "elapsed_seconds": round(elapsed, 1),
        "width": width,
        "height": height,
        "fps": fps,
        "duration": meta["duration"],
    }
