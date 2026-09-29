#!/usr/bin/env python3
"""Compare FAST, SMOOTH, SMART modes with proper ground-truth metrics."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

BASE_URL = "http://127.0.0.1:9091"
TEST_VIDEO = "/tmp/subtitle-comparison.mp4"


def _gradient_frame(h, w):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            frame[y, x] = [
                int(30 + 40 * (y / h)),
                int(50 + 30 * (x / w)),
                int(80 + 50 * ((x + y) / (w + h))),
            ]
    return frame


def _noise_frame(h, w, strength=15):
    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    noise = np.random.randint(-strength, strength + 1, (h, w, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def _moving_gradient_frame(h, w, frame_idx, total_frames):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    shift = (frame_idx / total_frames) * 40
    for y in range(h):
        for x in range(w):
            frame[y, x] = [
                int(30 + 40 * ((y + shift) / h) % 255),
                int(50 + 30 * ((x + shift) / w) % 255),
                int(80 + 50 * (((x + y) + shift) / (w + h)) % 255),
            ]
    return frame


def _create_subtitle_mask(text_shape, x, y, font, font_scale, thickness, outline_px):
    mask = np.zeros(text_shape[:2], dtype=np.uint8)
    if outline_px > 0:
        cv2.putText(mask, "X", (x, y), font, font_scale, 255, thickness + outline_px * 2, cv2.LINE_AA)
    cv2.putText(mask, "X", (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    return mask


def create_test_video():
    """Create a test video with multiple scenes and burned-in subtitles."""
    if os.path.exists(TEST_VIDEO):
        os.remove(TEST_VIDEO)

    width, height = 640, 360
    fps = 30
    duration = 9  # 3 scenes x 3 seconds each
    total_frames = fps * duration

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    subtitle_y = height - 60
    subtitle_x = 60

    clean_frames = []
    subtitle_frames = []
    subtitle_masks = []

    scenes = [
        (0, 3 * fps, "gradient", _gradient_frame(height, width)),
        (3 * fps, 6 * fps, "noise", _noise_frame(height, width)),
        (6 * fps, 9 * fps, "moving", None),
    ]

    for i in range(total_frames):
        scene_frame = None
        scene_name = "gradient"
        for start, end, name, base in scenes:
            if start <= i < end:
                scene_name = name
                if name == "moving":
                    scene_frame = _moving_gradient_frame(height, width, i - start, 3 * fps)
                else:
                    scene_frame = base.copy()
                break

        if scene_frame is None:
            scene_frame = _gradient_frame(height, width)

        clean_frame = scene_frame.copy()
        subtitle_frame = scene_frame.copy()

        if i < 3 * fps:
            text = "First subtitle"
            outline_px = 2
        elif i < 6 * fps:
            text = "Second line test"
            outline_px = 3
        else:
            text = "Final subtitle 123"
            outline_px = 2

        cv2.putText(
            subtitle_frame,
            text,
            (subtitle_x, subtitle_y + 28),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        text_mask = _create_subtitle_mask(
            (height, width), subtitle_x, subtitle_y + 28, font, font_scale, thickness, outline_px
        )

        clean_frames.append(clean_frame)
        subtitle_frames.append(subtitle_frame)
        subtitle_masks.append(text_mask)

    subtitle_video = "/tmp/subtitle-burned.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(subtitle_video, fourcc, fps, (width, height))
    for frame in subtitle_frames:
        writer.write(frame)
    writer.release()

    cmd = [
        "ffmpeg", "-y",
        "-i", subtitle_video,
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=9",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        TEST_VIDEO,
    ]
    print(f"Creating test video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg stderr:", result.stderr)
        raise RuntimeError("Failed to create test video")

    np.save("/tmp/clean-frames.npy", np.array(clean_frames))
    np.save("/tmp/subtitle-masks.npy", np.array(subtitle_masks))
    print(f"Test video created: {TEST_VIDEO}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration}s")
    print(f"  Scenes: gradient, noise, moving")
    return clean_frames, subtitle_masks


def wait_for_status(job_id, expected_statuses, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE_URL}/api/jobs/{job_id}")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") in expected_statuses:
                return data
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} did not reach {expected_statuses} within {timeout}s")


def process_video(mode, mask_strength=50):
    session = requests.Session()

    upload_start = time.time()
    with open(TEST_VIDEO, "rb") as f:
        files = {"video": ("test.mp4", f, "video/mp4")}
        r = session.post(f"{BASE_URL}/api/jobs", files=files)
    upload_elapsed = time.time() - upload_start
    assert r.status_code == 200
    job = r.json()
    job_id = job["job_id"]

    data = wait_for_status(job_id, {"ready", "failed"})
    assert data["status"] == "ready"

    form = {
        "x1": 50,
        "x2": 590,
        "y": 300,
        "thickness": 48,
        "feather": 8,
        "sample_gap": 4,
        "mode": mode,
        "mask_strength": mask_strength,
    }
    r = session.post(f"{BASE_URL}/api/jobs/{job_id}/process", data=form)
    assert r.status_code == 200
    proc_data = r.json()
    process_started_at = proc_data.get("started_at", time.time())

    data = wait_for_status(job_id, {"done", "failed"}, timeout=300)
    assert data["status"] == "done"

    process_completed_at = data.get("completed_at", time.time())
    processing_elapsed = process_completed_at - process_started_at

    r = session.get(f"{BASE_URL}/api/jobs/{job_id}/result", stream=True)
    assert r.status_code == 200
    result_path = f"/tmp/subtitle-{mode}-result.mp4"
    with open(result_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    return result_path, processing_elapsed, upload_elapsed


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_metrics(result_path, clean_frames, subtitle_masks):
    """Calculate proper metrics using ground-truth subtitle masks."""
    cap = cv2.VideoCapture(result_path)
    if not cap.isOpened():
        return None

    subtitle_recovery_mae_values = []
    roi_background_mae_values = []
    whole_roi_mae_values = []
    roi_psnr_values = []
    frame_idx = 0

    roi_y = clean_frames[0].shape[0] - 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(clean_frames):
            clean = clean_frames[frame_idx]
            sub_mask = subtitle_masks[frame_idx]
            if frame.shape != clean.shape:
                frame = cv2.resize(frame, (clean.shape[1], clean.shape[0]))
                sub_mask = cv2.resize(sub_mask, (clean.shape[1], clean.shape[0]))

            roi_result = frame[roi_y:, :]
            roi_clean = clean[roi_y:, :]
            roi_mask = sub_mask[roi_y:, :]

            subtitle_pixels = roi_mask > 0
            bg_pixels = ~subtitle_pixels

            if np.count_nonzero(subtitle_pixels) > 0:
                subtitle_recovery_mae = np.mean(
                    np.abs(roi_result[subtitle_pixels].astype(float) - roi_clean[subtitle_pixels].astype(float))
                )
                subtitle_recovery_mae_values.append(subtitle_recovery_mae)

            if np.count_nonzero(bg_pixels) > 0:
                roi_bg_mae = np.mean(
                    np.abs(roi_result[bg_pixels].astype(float) - roi_clean[bg_pixels].astype(float))
                )
                roi_background_mae_values.append(roi_bg_mae)

            whole_roi_mae = np.mean(np.abs(roi_result.astype(float) - roi_clean.astype(float)))
            whole_roi_mae_values.append(whole_roi_mae)

            roi_psnr = calculate_psnr(roi_result, roi_clean)
            roi_psnr_values.append(roi_psnr)

        frame_idx += 1

    cap.release()

    if subtitle_recovery_mae_values:
        return {
            "subtitle_recovery_mae": float(np.mean(subtitle_recovery_mae_values)),
            "subtitle_recovery_mae_std": float(np.std(subtitle_recovery_mae_values)),
            "roi_background_mae": float(np.mean(roi_background_mae_values)) if roi_background_mae_values else None,
            "roi_background_mae_std": float(np.std(roi_background_mae_values)) if roi_background_mae_values else None,
            "whole_roi_mae": float(np.mean(whole_roi_mae_values)),
            "whole_roi_mae_std": float(np.std(whole_roi_mae_values)),
            "roi_psnr": float(np.mean(roi_psnr_values)),
            "roi_psnr_std": float(np.std(roi_psnr_values)),
        }
    return None


def main():
    print("=== FAST vs SMOOTH vs SMART comparison (ground truth) ===\n")

    print("Creating test video with subtitles...")
    clean_frames, subtitle_masks = create_test_video()
    total_frames = len(clean_frames)

    modes = ["fast", "smooth", "smart"]
    results = {}

    for mode in modes:
        print(f"\nProcessing with {mode.upper()} mode...")
        result_path, processing_elapsed, upload_elapsed = process_video(mode)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=r_frame_rate", "-of", "json", result_path],
            capture_output=True, text=True,
        )
        video_info = json.loads(probe.stdout)
        stream = video_info["streams"][0]
        rate = stream.get("r_frame_rate", "30/1")
        num, den = rate.split("/")
        output_fps = float(num) / float(den)

        processing_fps = total_frames / processing_elapsed if processing_elapsed > 0 else 0
        realtime_factor = processing_fps / output_fps if output_fps > 0 else 0

        metrics = calculate_metrics(result_path, clean_frames, subtitle_masks)

        results[mode] = {
            "path": result_path,
            "processing_elapsed": processing_elapsed,
            "output_fps": output_fps,
            "processing_fps": processing_fps,
            "realtime_factor": realtime_factor,
            "metrics": metrics,
        }

        print(f"  Processing time: {processing_elapsed:.2f}s")
        print(f"  Processing FPS: {processing_fps:.1f}")
        print(f"  Realtime factor: {realtime_factor:.2f}x")
        if metrics:
            print(f"  Subtitle recovery MAE: {metrics['subtitle_recovery_mae']:.2f}")
            print(f"  ROI background MAE: {metrics['roi_background_mae']:.2f}")
            print(f"  Whole ROI MAE: {metrics['whole_roi_mae']:.2f}")
            print(f"  ROI PSNR: {metrics['roi_psnr']:.2f}")

    print("\n=== SUMMARY ===")
    print(f"{'Mode':<10} {'Proc FPS':<12} {'RT Factor':<12} {'Subtitle MAE':<15} {'ROI BG MAE':<12} {'Whole ROI MAE':<15} {'PSNR':<12}")
    print("-" * 90)
    for mode, data in results.items():
        m = data["metrics"]
        sub_mae = f"{m['subtitle_recovery_mae']:.2f}" if m else "N/A"
        bg_mae = f"{m['roi_background_mae']:.2f}" if m and m["roi_background_mae"] is not None else "N/A"
        whole_mae = f"{m['whole_roi_mae']:.2f}" if m else "N/A"
        psnr = f"{m['roi_psnr']:.2f}" if m else "N/A"
        print(f"{mode.upper():<10} {data['processing_fps']:<12.1f} {data['realtime_factor']:<12.2f} {sub_mae:<15} {bg_mae:<12} {whole_mae:<15} {psnr:<12}")

    if all(r["metrics"] for r in results.values()):
        print("\n=== ROI Background Preservation (lower is better) ===")
        for mode in modes:
            bg = results[mode]["metrics"]["roi_background_mae"]
            print(f"{mode.upper()}: {bg:.2f}" if bg is not None else f"{mode.upper()}: N/A")

        print("\n=== Subtitle Recovery (lower is better) ===")
        for mode in modes:
            sub = results[mode]["metrics"]["subtitle_recovery_mae"]
            print(f"{mode.upper()}: {sub:.2f}")

    print("\n=== Comparison test completed ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== Comparison test FAILED: {exc} ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)
