"""
자동 분류와 수동 검수 결과를 메모리상에서 통합한 뒤,
선별 영상의 프레임 단위 라벨을 Soft Boundary 가중치와 함께 생성한다.

[입력]
  processed/action_annotations.csv   (자동, good/weak 가중치 보유)
  processed/reviewed_annotations.csv (사람 확정, 가중치 1.0 일괄 부여)
  data/frames/<video_id>/timestamps.json (영상별 프레임 timestamp)

[출력]
  processed/frame_labels/<video_id>.csv (영상당 1개, 학습 직접 입력)
    컬럼: frame_index, frame_name, timestamp_sec,
          label, source, segment_id, sample_weight, is_boundary
  processed/frame_labels/_manifest.json
    영상별 메타데이터 (subset, recipe_type, n_segments, n_frames_emitted 등)
    학습 코드가 train/val 분리·필터링에 사용한다.

[가중치 정책]
  auto_good → 1.0
  auto_weak → 0.5
  reviewed  → 1.0
  Soft Boundary: 각 segment 양 끝 ±boundary_width 프레임에 boundary_factor 곱

[사용법]
  python scripts/generate_frame_labels.py
  python scripts/generate_frame_labels.py --boundary-width 2 --boundary-factor 0.5
  python scripts/generate_frame_labels.py --dry-run
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTS_PATH = ROOT / "processed" / "action_annotations.csv"
REVIEWED_PATH = ROOT / "processed" / "reviewed_annotations.csv"
ANNOTATIONS_JSON = ROOT / "YouCookII" / "annotations" / "youcookii_annotations_trainval.json"
FRAMES_ROOT = ROOT / "data" / "frames"
OUTPUT_ROOT = ROOT / "processed" / "frame_labels"
MANIFEST_PATH = OUTPUT_ROOT / "_manifest.json"

REVIEWED_WEIGHT = 1.0
KEY = ["video_id", "segment_id"]

OUTPUT_COLUMNS = [
    "frame_index", "frame_name", "timestamp_sec",
    "label", "source", "segment_id",
    "sample_weight", "is_boundary",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", default=str(ACTS_PATH))
    parser.add_argument("--reviewed", default=str(REVIEWED_PATH))
    parser.add_argument("--annotations-json", default=str(ANNOTATIONS_JSON),
                        help="Used to look up subset/recipe_type for the manifest.")
    parser.add_argument("--frames-root", default=str(FRAMES_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--boundary-width", type=int, default=2,
                        help="Frames at each segment edge treated as boundary.")
    parser.add_argument("--boundary-factor", type=float, default=0.5,
                        help="Weight multiplier applied to boundary frames.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
    ts_path = video_dir / "timestamps.json"
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


# --- 메인 ---

def main():
    args = parse_args()
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)

    merged = merge_segments(Path(args.acts), Path(args.reviewed))

    # video 단위 메타 lookup (manifest 작성용)
    with Path(args.annotations_json).open(encoding="utf-8") as f:
        db = json.load(f)["database"]

    available_videos = sorted([p.name for p in frames_root.iterdir() if p.is_dir()])
    labeled_vids = set(merged["video_id"])
    target_videos = [v for v in available_videos if v in labeled_vids]
    skipped_no_labels = [v for v in available_videos if v not in labeled_vids]
    if skipped_no_labels:
        print(f"WARN: {len(skipped_no_labels)} frame folders have no labels and are skipped.")

    if not args.dry_run:
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    manifest_videos = {}
    total_frames_emitted = 0
    total_boundary = 0
    total_segments_processed = 0

    for vid in target_videos:
        video_dir = frames_root / vid
        ts_path = video_dir / "timestamps.json"
        if not ts_path.exists():
            print(f"  SKIP {vid}: timestamps.json missing.")
            continue

        frames = load_timestamps(video_dir)
        segments = merged[merged["video_id"] == vid]
        out = expand_video(vid, segments, frames, args.boundary_width, args.boundary_factor)

        n_frames_total = len(frames)
        n_emitted = len(out)
        n_boundary = int(out["is_boundary"].sum()) if not out.empty else 0
        total_frames_emitted += n_emitted
        total_boundary += n_boundary
        total_segments_processed += len(segments)

        db_entry = db.get(vid, {})
        manifest_videos[vid] = {
            "subset": db_entry.get("subset"),
            "recipe_type": int(db_entry["recipe_type"]) if "recipe_type" in db_entry else None,
            "n_segments": int(len(segments)),
            "n_frames_total": n_frames_total,
            "n_frames_emitted": n_emitted,
            "n_boundary_frames": n_boundary,
        }

        if not args.dry_run:
            out.to_csv(output_root / f"{vid}.csv", index=False, encoding="utf-8-sig")

    # 통계 출력
    print()
    print(f"merged segment rows    : {len(merged)} (acts + reviewed)")
    print(f"videos with frames     : {len(available_videos)}")
    print(f"videos processed       : {len(manifest_videos)}")
    print(f"segments processed     : {total_segments_processed}")
    print(f"frames emitted (total) : {total_frames_emitted}")
    if total_frames_emitted:
        print(f"boundary frames        : {total_boundary} ({100*total_boundary/total_frames_emitted:.1f}%)")

    subset_counts = {}
    for meta in manifest_videos.values():
        subset_counts[meta["subset"]] = subset_counts.get(meta["subset"], 0) + 1
    print(f"subset distribution    : {subset_counts}")

    manifest = {
        "schema_version": 1,
        "description": (
            "Per-video metadata for frame_labels/*.csv. "
            "Training code reads this to split train/val and weight per-video losses."
        ),
        "boundary_width": args.boundary_width,
        "boundary_factor": args.boundary_factor,
        "summary": {
            "total_videos": len(manifest_videos),
            "total_segments": total_segments_processed,
            "total_frames_emitted": total_frames_emitted,
            "total_boundary_frames": total_boundary,
            "subset_distribution": subset_counts,
        },
        "videos": manifest_videos,
    }

    if args.dry_run:
        print("\ndry-run: not writing outputs.")
        return

    with (output_root / "_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote: {output_root}/  ({len(manifest_videos)} csv files + _manifest.json)")


if __name__ == "__main__":
    main()
