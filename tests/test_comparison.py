#!/usr/bin/env python3
"""Compare FAST, SMOOTH, SMART modes with quality metrics."""

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

    # Create a clean gradient background video
    width, height = 640, 360
    fps = 30
    duration = 3  # seconds
    total_frames = fps * duration

    # Create clean frames
    clean_frames = []
    for i in range(total_frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(30 + 40 * (y / height)),
                    int(50 + 30 * (x / width)),
                    int(80 + 50 * ((x + y) / (width + height))),
                ]

        # Add subtitle at bottom
        subtitle_y = height - 60
        cv2.rectangle(frame, (50, subtitle_y), (width - 50, subtitle_y + 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "Test Subtitle",
            (60, subtitle_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        clean_frames.append(frame)

    # Write video with ffmpeg
    clean_video = "/tmp/clean-bg.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clean_video, fourcc, fps, (width, height))
    for frame in clean_frames:
        writer.write(frame)
    writer.release()

    # Add audio and re-encode
    cmd = [
        "ffmpeg",
        "-y",
        "-i", clean_video,
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=3",
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
    """Upload and process video with given mode."""
    session = requests.Session()

    # Upload
    with open(TEST_VIDEO, "rb") as f:
        files = {"video": ("test.mp4", f, "video/mp4")}
        r = session.post(f"{BASE_URL}/api/jobs", files=files)
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

    # Wait for done
    data = wait_for_status(job_id, {"done", "failed"}, timeout=300)
    assert data["status"] == "done"

    # Download result
    r = session.get(f"{BASE_URL}/api/jobs/{job_id}/result", stream=True)
    assert r.status_code == 200
    result_path = f"/tmp/subtitle-{mode}-result.mp4"
    with open(result_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    return result_path


def calculate_mae(result_path, clean_frames):
    """Calculate MAE between result video and clean frames."""
    cap = cv2.VideoCapture(result_path)
    if not cap.isOpened():
        return None

    mae_values = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(clean_frames):
            # Resize if needed
            clean = clean_frames[frame_idx]
            if frame.shape != clean.shape:
                frame = cv2.resize(frame, (clean.shape[1], clean.shape[0]))

            # Calculate MAE only in subtitle ROI (bottom area)
            roi_y = clean.shape[0] - 100
            roi_result = frame[roi_y:, :]
            roi_clean = clean[roi_y:, :]

            mae = np.mean(np.abs(roi_result.astype(float) - roi_clean.astype(float)))
            mae_values.append(mae)

        frame_idx += 1

    cap.release()

    if mae_values:
        return np.mean(mae_values), np.std(mae_values)
    return None, None


def main():
    print("=== FAST vs SMOOTH vs SMART comparison ===\n")

    # Create test video
    print("Creating test video with subtitles...")
    clean_frames = create_test_video()

    modes = ["fast", "smooth", "smart"]
    results = {}

    for mode in modes:
        print(f"\nProcessing with {mode.upper()} mode...")
        start_time = time.time()
        result_path = process_video(mode)
        elapsed = time.time() - start_time

        # Get video info
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", result_path],
            capture_output=True, text=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])

        # Calculate FPS
        cap = cv2.VideoCapture(result_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Calculate MAE
        mae, mae_std = calculate_mae(result_path, clean_frames)

        results[mode] = {
            "path": result_path,
            "elapsed": elapsed,
            "duration": duration,
            "fps": fps,
            "mae": mae,
            "mae_std": mae_std,
        }

        print(f"  Done in {elapsed:.1f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  MAE: {mae:.2f} ± {mae_std:.2f}" if mae else "  MAE: N/A")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Mode':<10} {'Time (s)':<12} {'FPS':<10} {'MAE':<15}")
    print("-" * 50)
    for mode, data in results.items():
        mae_str = f"{data['mae']:.2f}" if data["mae"] else "N/A"
        print(f"{mode.upper():<10} {data['elapsed']:<12.1f} {data['fps']:<10.1f} {mae_str:<15}")

    # Verify SMART has lowest MAE (or at least reasonable)
    if results["smart"]["mae"] and results["fast"]["mae"]:
        smart_mae = results["smart"]["mae"]
        fast_mae = results["fast"]["mae"]
        smooth_mae = results["smooth"]["mae"]

        print(f"\nSMART MAE: {smart_mae:.2f}")
        print(f"FAST MAE: {fast_mae:.2f}")
        print(f"SMOOTH MAE: {smooth_mae:.2f}")

        if smart_mae < fast_mae:
            print("✓ SMART has lower MAE than FAST (better quality)")
        else:
            print("✗ SMART has higher MAE than FAST")

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
