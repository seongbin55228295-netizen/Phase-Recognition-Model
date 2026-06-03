"""Frame-level label generation logic (segment merge + Soft Boundary).

Single source of truth for turning the auto/reviewed segment annotations into the
per-frame training labels that scripts/generate_frame_labels.py writes out. Kept
import-light (pandas only) so the merge/partition/boundary-weighting logic can be
unit-tested without torch/torchvision. The thin CLI wires project paths, builds
the manifest, prints stats, and writes the per-video CSVs.

Pipeline:
    merge_segments (auto + reviewed -> one segment table; partition checked)
        -> per video: load_timestamps + expand_video
           (map each frame to its covering segment; Soft Boundary weights at edges)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REVIEWED_WEIGHT = 1.0
KEY = ["video_id", "segment_id"]

OUTPUT_COLUMNS = [
    "frame_index", "frame_name", "timestamp_sec",
    "label", "source", "segment_id",
    "sample_weight", "is_boundary",
]


# --- segment 통합 ---

def normalize_acts(acts: pd.DataFrame) -> pd.DataFrame:
    df = acts.rename(columns={
        "predicted_label": "label",
        "recommended_sample_weight": "sample_weight",
    })
    df["source"] = "auto_" + df["label_quality"].astype(str)
    return df.drop(columns=["label_quality"])


def normalize_reviewed(reviewed: pd.DataFrame) -> pd.DataFrame:
    df = reviewed.rename(columns={"predicted_label": "label"})
    df["source"] = "reviewed"
    df["sample_weight"] = REVIEWED_WEIGHT
    return df


def assert_partition(acts: pd.DataFrame, reviewed: pd.DataFrame) -> None:
    ak = set(map(tuple, acts[KEY].itertuples(index=False, name=None)))
    rk = set(map(tuple, reviewed[KEY].itertuples(index=False, name=None)))
    overlap = ak & rk
    if overlap:
        raise RuntimeError(
            f"action_annotations and reviewed_annotations overlap on {len(overlap)} keys. "
            "Ensure scripts/generate_annotation_labels.py produced a clean acts/review partition "
            "and that reviewed_annotations.csv contains only rows from review_queue.csv."
        )


def merge_segments(acts_path: Path, reviewed_path: Path) -> pd.DataFrame:
    acts = pd.read_csv(acts_path, encoding="utf-8-sig")
    reviewed = pd.read_csv(reviewed_path, encoding="utf-8-sig")
    assert_partition(acts, reviewed)
    merged = pd.concat(
        [normalize_acts(acts), normalize_reviewed(reviewed)],
        ignore_index=True,
    )
    return merged.sort_values(["subset", "video_id", "segment_id"]).reset_index(drop=True)


# --- 프레임 단위 확장 ---

def load_timestamps(video_dir: Path) -> pd.DataFrame:
    ts_path = Path(video_dir) / "timestamps.json"
    with ts_path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = [
        {"frame_index": v["frame_index"], "frame_name": k, "timestamp_sec": v["timestamp_sec"]}
        for k, v in data.items()
    ]
    return pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)


def expand_video(video_id: str, segments: pd.DataFrame, frames: pd.DataFrame,
                 boundary_width: int, boundary_factor: float) -> pd.DataFrame:
    """Map each frame to a covering segment; emit a per-frame label row if covered."""
    output_rows = []
    segments = segments.sort_values("segment_id").reset_index(drop=True)

    for _, seg in segments.iterrows():
        start = float(seg["segment_start"])
        end = float(seg["segment_end"])
        in_segment = frames[(frames["timestamp_sec"] >= start)
                            & (frames["timestamp_sec"] < end)].reset_index(drop=True)
        n = len(in_segment)
        if n == 0:
            continue
        seg_weight = float(seg["sample_weight"])
        for i, frame in in_segment.iterrows():
            distance_from_edge = min(i, n - 1 - i)
            is_boundary = bool(distance_from_edge < boundary_width)
            weight = seg_weight * (boundary_factor if is_boundary else 1.0)
            output_rows.append({
                "frame_index": int(frame["frame_index"]),
                "frame_name": frame["frame_name"],
                "timestamp_sec": float(frame["timestamp_sec"]),
                "label": seg["label"],
                "source": seg["source"],
                "segment_id": int(seg["segment_id"]),
                "sample_weight": round(weight, 6),
                "is_boundary": is_boundary,
            })

    if not output_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(output_rows).sort_values("frame_index").reset_index(drop=True)
    dup = out["frame_index"].duplicated().sum()
    if dup:
        print(f"  WARN {video_id}: {dup} frames matched multiple segments (overlapping). "
              "Earlier segment_id wins.", file=sys.stderr)
        out = out.drop_duplicates("frame_index", keep="first").reset_index(drop=True)
    return out[OUTPUT_COLUMNS]
