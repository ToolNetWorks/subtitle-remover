#!/usr/bin/env python3
"""NOOP baseline test to measure codec error floor."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests

BASE_URL = "http://127.0.0.1:9091"


def create_test_video():
    """Create a simple test video."""
    test_video = "/tmp/noop-test.mp4"
    if os.path.exists(test_video):
        os.remove(test_video)

    width, height = 640, 360
    fps = 30
    duration = 3

    frames = []
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(30 + 40 * (y / height)),
                    int(50 + 30 * (x / width)),
                    int(80 + 50 * ((x + y) / (width + height))),
                ]
        frames.append(frame)

    clean_video = "/tmp/noop-clean.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clean_video, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    cmd = [
        "ffmpeg", "-y",
        "-i", clean_video,
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=3",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        test_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to create test video")

    np.save("/tmp/noop-clean-frames.npy", np.array(frames))
    return test_video


def calculate_metrics(result_path, clean_frames):
    """Calculate MAE and PSNR for NOOP baseline."""
    cap = cv2.VideoCapture(result_path)
    if not cap.isOpened():
        return None

    mae_values = []
    psnr_values = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(clean_frames):
            clean = clean_frames[frame_idx]
            if frame.shape != clean.shape:
                frame = cv2.resize(frame, (clean.shape[1], clean.shape[0]))

            mae = np.mean(np.abs(frame.astype(float) - clean.astype(float)))
            mae_values.append(mae)

            mse = np.mean((frame.astype(float) - clean.astype(float)) ** 2)
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                psnr_values.append(psnr)

        frame_idx += 1

    cap.release()

    if mae_values:
        return {
            "mae": np.mean(mae_values),
            "psnr": np.mean(psnr_values) if psnr_values else 0,
        }
    return None


def main():
    print("=== NOOP Baseline Test ===\n")
    print("Creating test video...")
    test_video = create_test_video()

    session = requests.Session()
    with open(test_video, "rb") as f:
        files = {"video": ("test.mp4", f, "video/mp4")}
        r = session.post(f"{BASE_URL}/api/jobs", files=files)
    assert r.status_code == 200
    job = r.json()
    job_id = job["job_id"]

    for _ in range(30):
        r = session.get(f"{BASE_URL}/api/jobs/{job_id}")
        if r.json().get("status") == "ready":
            break
        import time
        time.sleep(1)

    form = {
        "x1": 50,
        "x2": 590,
        "y": 300,
        "thickness": 48,
        "feather": 8,
        "sample_gap": 4,
        "mode": "fast",
        "mask_strength": 50,
    }
    r = session.post(f"{BASE_URL}/api/jobs/{job_id}/process", data=form)
    assert r.status_code == 200

    import time
    for _ in range(60):
        r = session.get(f"{BASE_URL}/api/jobs/{job_id}")
        if r.json().get("status") == "done":
            break
        time.sleep(1)

    r = session.get(f"{BASE_URL}/api/jobs/{job_id}/result", stream=True)
    assert r.status_code == 200
    result_path = "/tmp/noop-result.mp4"
    with open(result_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    clean_frames = np.load("/tmp/noop-clean-frames.npy")
    metrics = calculate_metrics(result_path, clean_frames)

    if metrics:
        print(f"NOOP baseline MAE: {metrics['mae']:.2f}")
        print(f"NOOP baseline PSNR: {metrics['psnr']:.2f}")
        print("\nThis represents the codec error floor.")
        print("Any remover MAE should be compared against this baseline.")

    print("\n=== NOOP test completed ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== NOOP test FAILED: {exc} ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)
