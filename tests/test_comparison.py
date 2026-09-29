#!/usr/bin/env python3
"""Compare FAST, SMOOTH, SMART modes with correct quality metrics."""

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


def create_test_video():
    """Create a test video with clean background and burned-in subtitles."""
    if os.path.exists(TEST_VIDEO):
        os.remove(TEST_VIDEO)

    width, height = 640, 360
    fps = 30
    duration = 6  # seconds
    total_frames = fps * duration

    # Subtitles that change over time
    subtitles = [
        (0, 2 * fps, "First subtitle"),
        (2 * fps, 4 * fps, "Second line test"),
        (4 * fps, 6 * fps, "Final subtitle 123"),
    ]

    clean_frames = []
    subtitle_frames = []

    for i in range(total_frames):
        # Create gradient background (clean frame)
        clean_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                clean_frame[y, x] = [
                    int(30 + 40 * (y / height)),
                    int(50 + 30 * (x / width)),
                    int(80 + 50 * ((x + y) / (width + height))),
                ]

        # Create subtitle frame (copy of clean)
        subtitle_frame = clean_frame.copy()

        # Add subtitle at bottom (NO black box behind)
        subtitle_y = height - 60
        current_text = "Test"
        for start, end, text in subtitles:
            if start <= i < end:
                current_text = text
                break

        # Draw text with outline
        cv2.putText(
            subtitle_frame,
            current_text,
            (60, subtitle_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        clean_frames.append(clean_frame.copy())
        subtitle_frames.append(subtitle_frame)

    # Write subtitle video with ffmpeg
    subtitle_video = "/tmp/subtitle-burned.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(subtitle_video, fourcc, fps, (width, height))
    for frame in subtitle_frames:
        writer.write(frame)
    writer.release()

    # Add audio and re-encode
    cmd = [
        "ffmpeg",
        "-y",
        "-i", subtitle_video,
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=6",
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

    # Save clean frames for MAE calculation
    np.save("/tmp/clean-frames.npy", np.array(clean_frames))
    print(f"Test video created: {TEST_VIDEO}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration}s")
    print(f"  Subtitles: {len(subtitles)} different texts")
    return clean_frames


def wait_for_status(job_id, expected_statuses, timeout=120):
    """Poll job status until it reaches one of expected_statuses."""
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE_URL}/api/jobs/{job_id}")
        if r.status_code == 200:
            data = r.json()
            status = data.get("status")
            if status in expected_statuses:
                return data
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} did not reach {expected_statuses} within {timeout}s")


def process_video(mode, mask_strength=50):
    """Upload and process video with given mode. Returns timing info."""
    session = requests.Session()

    # Upload
    upload_start = time.time()
    with open(TEST_VIDEO, "rb") as f:
        files = {"video": ("test.mp4", f, "video/mp4")}
        r = session.post(f"{BASE_URL}/api/jobs", files=files)
    upload_elapsed = time.time() - upload_start
    assert r.status_code == 200
    job = r.json()
    job_id = job["job_id"]

    # Wait for ready
    data = wait_for_status(job_id, {"ready", "failed"})
    assert data["status"] == "ready"

    # Process
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

    # Wait for done
    data = wait_for_status(job_id, {"done", "failed"}, timeout=300)
    assert data["status"] == "done"

    process_completed_at = data.get("completed_at", time.time())
    processing_elapsed = process_completed_at - process_started_at

    # Download result
    r = session.get(f"{BASE_URL}/api/jobs/{job_id}/result", stream=True)
    assert r.status_code == 200
    result_path = f"/tmp/subtitle-{mode}-result.mp4"
    with open(result_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    return result_path, processing_elapsed, upload_elapsed


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_metrics(result_path, clean_frames):
    """Calculate MAE, PSNR, and background preservation metrics."""
    cap = cv2.VideoCapture(result_path)
    if not cap.isOpened():
        return None

    roi_mae_values = []
    roi_psnr_values = []
    bg_mae_values = []
    frame_idx = 0

    # ROI: bottom 100 pixels where subtitles are
    roi_y = clean_frames[0].shape[0] - 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(clean_frames):
            clean = clean_frames[frame_idx]
            if frame.shape != clean.shape:
                frame = cv2.resize(frame, (clean.shape[1], clean.shape[0]))

            # ROI metrics (subtitle area)
            roi_result = frame[roi_y:, :]
            roi_clean = clean[roi_y:, :]

            roi_mae = np.mean(np.abs(roi_result.astype(float) - roi_clean.astype(float)))
            roi_psnr = calculate_psnr(roi_result, roi_clean)
            roi_mae_values.append(roi_mae)
            roi_psnr_values.append(roi_psnr)

            # Background preservation (area outside ROI)
            bg_result = frame[:roi_y, :]
            bg_clean = clean[:roi_y, :]
            bg_mae = np.mean(np.abs(bg_result.astype(float) - bg_clean.astype(float)))
            bg_mae_values.append(bg_mae)

        frame_idx += 1

    cap.release()

    if roi_mae_values:
        return {
            "roi_mae": np.mean(roi_mae_values),
            "roi_mae_std": np.std(roi_mae_values),
            "roi_psnr": np.mean(roi_psnr_values),
            "roi_psnr_std": np.std(roi_psnr_values),
            "bg_mae": np.mean(bg_mae_values),
            "bg_mae_std": np.std(bg_mae_values),
        }
    return None


def main():
    print("=== FAST vs SMOOTH vs SMART comparison (corrected) ===\n")

    # Create test video
    print("Creating test video with subtitles...")
    clean_frames = create_test_video()
    total_frames = len(clean_frames)

    modes = ["fast", "smooth", "smart"]
    results = {}

    for mode in modes:
        print(f"\nProcessing with {mode.upper()} mode...")
        result_path, processing_elapsed, upload_elapsed = process_video(mode)

        # Get video info
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-of", "json", result_path],
            capture_output=True, text=True,
        )
        video_info = json.loads(probe.stdout)
        stream = video_info["streams"][0]
        output_width = int(stream["width"])
        output_height = int(stream["height"])
        
        rate = stream.get("r_frame_rate", "30/1")
        num, den = rate.split("/")
        output_fps = float(num) / float(den)

        # Calculate processing FPS
        processing_fps = total_frames / processing_elapsed if processing_elapsed > 0 else 0
        realtime_factor = processing_fps / output_fps if output_fps > 0 else 0

        # Calculate quality metrics
        metrics = calculate_metrics(result_path, clean_frames)

        results[mode] = {
            "path": result_path,
            "processing_elapsed": processing_elapsed,
            "upload_elapsed": upload_elapsed,
            "output_fps": output_fps,
            "processing_fps": processing_fps,
            "realtime_factor": realtime_factor,
            "metrics": metrics,
        }

        print(f"  Processing time: {processing_elapsed:.2f}s")
        print(f"  Output FPS: {output_fps:.1f}")
        print(f"  Processing FPS: {processing_fps:.1f}")
        print(f"  Realtime factor: {realtime_factor:.2f}x")
        if metrics:
            print(f"  ROI MAE: {metrics['roi_mae']:.2f} ± {metrics['roi_mae_std']:.2f}")
            print(f"  ROI PSNR: {metrics['roi_psnr']:.2f} ± {metrics['roi_psnr_std']:.2f}")
            print(f"  Background MAE: {metrics['bg_mae']:.2f} ± {metrics['bg_mae_std']:.2f}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Mode':<10} {'Proc FPS':<12} {'RT Factor':<12} {'ROI MAE':<12} {'ROI PSNR':<12} {'BG MAE':<12}")
    print("-" * 70)
    for mode, data in results.items():
        m = data["metrics"]
        mae_str = f"{m['roi_mae']:.2f}" if m else "N/A"
        psnr_str = f"{m['roi_psnr']:.2f}" if m else "N/A"
        bg_str = f"{m['bg_mae']:.2f}" if m else "N/A"
        print(f"{mode.upper():<10} {data['processing_fps']:<12.1f} {data['realtime_factor']:<12.2f} {mae_str:<12} {psnr_str:<12} {bg_str:<12}")

    # Detailed comparison
    if all(r["metrics"] for r in results.values()):
        smart_mae = results["smart"]["metrics"]["roi_mae"]
        fast_mae = results["fast"]["metrics"]["roi_mae"]
        smooth_mae = results["smooth"]["metrics"]["roi_mae"]

        print(f"\n=== ROI MAE Comparison ===")
        print(f"FAST:   {fast_mae:.2f}")
        print(f"SMOOTH: {smooth_mae:.2f}")
        print(f"SMART:  {smart_mae:.2f}")

        print(f"\n=== Background Preservation (MAE) ===")
        print(f"FAST:   {results['fast']['metrics']['bg_mae']:.2f}")
        print(f"SMOOTH: {results['smooth']['metrics']['bg_mae']:.2f}")
        print(f"SMART:  {results['smart']['metrics']['bg_mae']:.2f}")

        best = min(modes, key=lambda m: results[m]["metrics"]["roi_mae"])
        print(f"\nBest ROI quality: {best.upper()} (lowest MAE)")

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
