"""Preprocessing utilities (kept import-light: no torch/torchvision at package import).

frame_extraction is the single source of truth for the ffmpeg 2fps/short-256
frame pipeline shared by scripts/extract_frames.py (training) and
scripts/infer_video.py (inference).
"""
