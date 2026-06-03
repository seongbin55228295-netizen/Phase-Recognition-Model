"""
자동 분류와 수동 검수 결과를 메모리상에서 통합한 뒤,
선별 영상의 프레임 단위 라벨을 Soft Boundary 가중치와 함께 생성한다.

이 파일은 얇은 CLI 래퍼다 — 프로젝트 경로 배선, manifest 작성, 통계 출력,
CSV 쓰기만 담당하고, segment 병합·프레임 확장·Soft Boundary 로직은
src/preprocessing/frame_labeling.py 에 있다.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# segment 병합·프레임 확장·Soft Boundary 로직 단일 출처:
# src/preprocessing/frame_labeling.py
from src.preprocessing.frame_labeling import (
    expand_video,
    load_timestamps,
    merge_segments,
)

ACTS_PATH = ROOT / "processed" / "action_annotations.csv"
REVIEWED_PATH = ROOT / "processed" / "reviewed_annotations.csv"
ANNOTATIONS_JSON = ROOT / "YouCookII" / "annotations" / "youcookii_annotations_trainval.json"
FRAMES_ROOT = ROOT / "data" / "frames"
OUTPUT_ROOT = ROOT / "processed" / "frame_labels"
MANIFEST_PATH = OUTPUT_ROOT / "_manifest.json"


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
