"""Raw-video acquisition and frame-tensor loading for inference.

- download_video: fetch a single arbitrary URL (or a YouCook2 video_url) with
  yt-dlp, mirroring the format selection used by scripts/data/download_videos.py.
- load_frame_tensors: read an extracted frame directory (frame_*.jpg +
  timestamps.json) and apply the eval transform, returning a sequence aligned by
  frame_index.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
from PIL import Image

# Same format ladder as scripts/data/download_videos.py (<=480p mp4, audio merged).
_YT_FORMAT = (
    "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/"
    "best[height<=480][ext=mp4]/best[height<=480]"
)


def download_video(
    url: str,
    out_dir: str | Path,
    *,
    video_id: str | None = None,
    cookies: str | Path | None = None,
    timeout: int = 600,
) -> Path:
    """Download `url` into out_dir/<video_id or yt-id>.mp4 and return its path.

    Skips the download if the target file already exists. Raises RuntimeError on
    failure (yt-dlp missing, video unavailable, etc.).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # When video_id is known, force a deterministic filename; otherwise let
    # yt-dlp name by the YouTube id.
    out_template = str(out_dir / (f"{video_id}.%(ext)s" if video_id else "%(id)s.%(ext)s"))

    if video_id:
        existing = out_dir / f"{video_id}.mp4"
        if existing.exists():
            return existing

    cmd = ["yt-dlp"]
    if cookies and Path(cookies).is_file() and Path(cookies).stat().st_size > 0:
        cmd += ["--cookies", str(cookies)]
    cmd += [
        "-o", out_template,
        "--format", _YT_FORMAT,
        "--merge-output-format", "mp4",
        "--no-overwrites",
        "--retries", "3",
        "--socket-timeout", "30",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp not found. Install it: pip install yt-dlp") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"download timed out after {timeout}s: {url}") from exc

    # Locate the produced mp4.
    if video_id:
        produced = out_dir / f"{video_id}.mp4"
        if produced.exists():
            return produced
    else:
        # Newest mp4 in out_dir is our download.
        mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s:
            return mp4s[0]

    raise RuntimeError(
        f"yt-dlp produced no mp4 for {url}.\n{(result.stderr or '')[:500]}"
    )


def load_frame_tensors(frames_dir: str | Path, transform) -> dict:
    """Load an extracted frame directory into model-ready tensors.

    Returns:
        {
          "frame_names":  list[str]    # sorted by frame_index
          "frame_indices": list[int]
          "timestamps":   list[float]  # seconds
          "images":       Tensor (N, 3, 224, 224)   # transform applied
        }
    """
    frames_dir = Path(frames_dir)
    ts_path = frames_dir / "timestamps.json"
    if not ts_path.exists():
        raise FileNotFoundError(f"timestamps.json missing in {frames_dir}")

    with ts_path.open(encoding="utf-8") as f:
        ts = json.load(f)

    # Order strictly by frame_index (the autoregressive sequence order).
    items = sorted(ts.items(), key=lambda kv: kv[1]["frame_index"])

    frame_names: list[str] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    images: list[torch.Tensor] = []
    for name, meta in items:
        img_path = frames_dir / name
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            images.append(transform(im.convert("RGB")))
        frame_names.append(name)
        frame_indices.append(int(meta["frame_index"]))
        timestamps.append(float(meta["timestamp_sec"]))

    if not images:
        raise RuntimeError(f"no frames loaded from {frames_dir}")

    return {
        "frame_names": frame_names,
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "images": torch.stack(images, dim=0),
    }
