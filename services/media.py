from __future__ import annotations

import json
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result
    raise MediaError(result.stderr.strip() or "Media command failed")


def probe_video(path: Path) -> dict:
    if not path.exists():
        raise MediaError(f"Video not found: {path}")

    result = _run([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ])
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise MediaError("No video stream found")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    rate = str(stream.get("r_frame_rate") or "0/1")
    numerator, denominator = rate.split("/", 1)
    denominator_value = float(denominator or 1)
    fps = float(numerator or 0) / denominator_value if denominator_value else 0.0

    duration = float(
        stream.get("duration")
        or (payload.get("format") or {}).get("duration")
        or 0.0
    )

    nb_frames_raw = stream.get("nb_frames")
    total_frames = int(nb_frames_raw) if str(nb_frames_raw).isdigit() else 0
    if total_frames <= 0 and duration > 0 and fps > 0:
        total_frames = max(1, int(round(duration * fps)))

    if width <= 0 or height <= 0 or fps <= 0:
        raise MediaError("Invalid video metadata")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "total_frames": total_frames,
    }


def extract_frame(
    input_path: Path,
    output_path: Path,
    at_seconds: float,
) -> None:
    at_seconds = max(0.0, float(at_seconds))
    _run([
        "ffmpeg",
        "-y",
        "-ss", f"{at_seconds:.3f}",
        "-i", str(input_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ])


def mux_audio(
    processed_video: Path,
    source_video: Path,
    output_path: Path,
) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(processed_video),
            "-i", str(source_video),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return

    # Some source audio codecs cannot be copied into MP4.
    _run([
        "ffmpeg",
        "-y",
        "-i", str(processed_video),
        "-i", str(source_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ])
