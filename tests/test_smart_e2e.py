#!/usr/bin/env python3
"""End-to-end test for SMART mode."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:9091"
TEST_VIDEO = "/tmp/subtitle-test.mp4"


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


def main():
    print("=== SMART mode e2e test ===")

    session = requests.Session()

    # 1. Upload
    print("\n1. Uploading video with subtitles...")
    with open(TEST_VIDEO, "rb") as f:
        files = {"video": ("subtitle-test.mp4", f, "video/mp4")}
        r = session.post(f"{BASE_URL}/api/jobs", files=files)
    assert r.status_code == 200, f"Upload failed: {r.status_code} {r.text}"
    job = r.json()
    job_id = job["job_id"]
    print(f"   Job created: {job_id}")
    assert job["status"] == "uploaded"

    # 2. Wait for ready
    print("   Waiting for ready...")
    data = wait_for_status(job_id, {"ready", "failed"})
    assert data["status"] == "ready", f"Job failed: {data.get('error')}"
    print(f"   Job ready: {data['width']}x{data['height']}")

    # 3. Preview SMART
    print("2. Requesting SMART preview...")
    preview_form = {
        "x1": 50,
        "x2": 590,
        "y": 330,
        "thickness": 48,
        "feather": 8,
        "sample_gap": 4,
        "mode": "smart",
        "mask_strength": 50,
    }
    r = session.post(f"{BASE_URL}/api/jobs/{job_id}/preview", data=preview_form)
    assert r.status_code == 200, f"Preview failed: {r.status_code} {r.text}"
    assert r.headers.get("content-type") == "image/jpeg"
    print(f"   Preview OK, size={len(r.content)} bytes")

    # 4. Preview mask
    print("3. Requesting mask preview...")
    r = session.post(f"{BASE_URL}/api/jobs/{job_id}/preview-mask", data=preview_form)
    assert r.status_code == 200, f"Mask preview failed: {r.status_code} {r.text}"
    assert r.headers.get("content-type") == "image/jpeg"
    print(f"   Mask preview OK, size={len(r.content)} bytes")

    # 5. Process SMART
    print("4. Starting SMART process...")
    r = session.post(f"{BASE_URL}/api/jobs/{job_id}/process", data=preview_form)
    assert r.status_code == 200, f"Process start failed: {r.status_code} {r.text}"
    proc = r.json()
    assert proc["status"] == "queued", f"Unexpected status: {proc['status']}"
    print(f"   Process queued")

    # 6. Wait for done
    print("   Waiting for done...")
    data = wait_for_status(job_id, {"done", "failed"}, timeout=300)
    assert data["status"] == "done", f"Processing failed: {data.get('error')}"
    print(f"   Done: {data.get('processed_frames')}/{data.get('total_frames')} frames")

    # 7. Result
    print("5. Downloading result...")
    r = session.get(f"{BASE_URL}/api/jobs/{job_id}/result", stream=True)
    assert r.status_code == 200, f"Result failed: {r.status_code} {r.text}"
    result_path = "/tmp/subtitle-smart-result.mp4"
    with open(result_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"   Result saved: {result_path} ({os.path.getsize(result_path)} bytes)")

    # 8. Verify audio exists
    print("6. Verifying audio...")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_name",
            "-of", "json",
            result_path,
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, f"ffprobe failed: {probe.stderr}"
    audio_info = json.loads(probe.stdout)
    audio_streams = audio_info.get("streams", [])
    assert len(audio_streams) > 0, "No audio stream in result"
    print(f"   Audio stream found: {audio_streams[0].get('codec_name')}")

    print("\n=== SMART mode e2e test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== SMART mode e2e test FAILED: {exc} ===")
        sys.exit(1)
