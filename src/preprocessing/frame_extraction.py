"""Reusable ffmpeg frame extraction.

This is the single source of truth for turning a raw video file into the
2-FPS / short-side-256 JPEG frames the model was trained on. Both the training
preprocessing entrypoint (scripts/extract_frames.py, which drives the selected
300 videos) and the inference entrypoint (scripts/infer_video.py, which handles
arbitrary YouTube / held-out YouCook2 videos) call into this module so the pixel
pipeline is byte-identical across train and test.

Output layout (per video):
    <output_dir>/frame_000000.jpg, frame_000001.jpg, ...
    <output_dir>/timestamps.json   # {frame_name: {frame_index, timestamp_sec}}
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# Defaults match the training preprocessing spec (README §3.1).
DEFAULT_FPS = 2
DEFAULT_IMAGE_QUALITY = 2       # ffmpeg -q:v, 1(best)~31(worst)
DEFAULT_RESIZE_SHORT = 256      # short-side resize target (px)
DEFAULT_TIMEOUT = 300           # seconds per video


def get_video_fps(video_path: str | Path) -> float | None:
    """Return the source video's frame rate via ffprobe, or None on failure."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fps_str = result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            return float(fps_str)
    except Exception:
        pass
    return None


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: int = DEFAULT_FPS,
    image_quality: int = DEFAULT_IMAGE_QUALITY,
    resize_short: int = DEFAULT_RESIZE_SHORT,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[bool, int | str]:
    """Extract frames at `fps`, short-side resized to `resize_short`.

    Writes frame_%06d.jpg + timestamps.json into `output_dir`. On failure the
    (possibly partial) output_dir is removed.

    Returns (True, frame_count) on success, or (False, error_message) on failure.
    """
    video_path = str(video_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # vf filter chain:
    #   1) fps=N            -> N frames per second
    #   2) scale            -> short side to `resize_short`, aspect kept, even dims
    #   3) showinfo         -> per-frame pts_time emitted on stderr (parsed below)
    vf_filter = (
        f"fps={fps},"
        f"scale='if(gt(iw,ih),-2,{resize_short})':'if(gt(iw,ih),{resize_short},-2)',"
        f"showinfo"
    )

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", vf_filter,
        "-q:v", str(image_quality),
        "-start_number", "0",
        os.path.join(output_dir, "frame_%06d.jpg"),
        "-y",
        "-loglevel", "info",   # showinfo requires >= info
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        frame_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".jpg"))
        frame_count = len(frame_files)
        if frame_count == 0:
            shutil.rmtree(output_dir, ignore_errors=True)
            return False, (result.stderr[:200] if result.stderr else "no frames extracted")

        # Parse pts_time from showinfo output.
        timestamps: list[float] = []
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                try:
                    pts_part = line.split("pts_time:")[1]
                    timestamps.append(float(pts_part.split()[0]))
                except (IndexError, ValueError):
                    continue

        # Fall back to a uniform grid if showinfo under-reported.
        if len(timestamps) < frame_count:
            timestamps = [i / fps for i in range(frame_count)]

        timestamp_data = {
            f"frame_{i:06d}.jpg": {"frame_index": i, "timestamp_sec": round(ts, 4)}
            for i, ts in enumerate(timestamps[:frame_count])
        }
        with open(os.path.join(output_dir, "timestamps.json"), "w", encoding="utf-8") as f:
            json.dump(timestamp_data, f, indent=2)

        return True, frame_count

    except subprocess.TimeoutExpired:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, "timeout"
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, str(exc)
