"""
선별된 영상에서 2 FPS 프레임 추출
- ffmpeg로 초당 2프레임 추출
- 프레임별 타임스탬프 기록
- 해상도 통일 (짧은 변 기준 256px 리사이즈)
- 실패 시 불완전 폴더 정리

대상 영상은 configs/selected_video_ids.json 의 300개 ID 를 기준으로 하며,
카테고리(레시피명) 메타데이터는 YouCookII/annotations/*.json 과 label_foodtype.csv
에서 즉석 lookup 한다.

사용법: python scripts/extract_frames.py
필요: ffmpeg 설치, configs/selected_video_ids.json, 다운로드된 영상
"""
import csv
import json
import os
import subprocess
import shutil
from pathlib import Path

# === 설정 ===
ROOT = Path(__file__).resolve().parents[1]
SELECTED_IDS_CONFIG = ROOT / "configs" / "selected_video_ids.json"
ANNOTATIONS_JSON = ROOT / "YouCookII" / "annotations" / "youcookii_annotations_trainval.json"
FOODTYPE_PATH = ROOT / "YouCookII" / "label_foodtype.csv"
VIDEO_DIR = str(ROOT / "data" / "videos")
FRAMES_DIR = str(ROOT / "data" / "frames")
STATS_FILE = str(ROOT / "reports" / "logs" / "frame_extraction_stats.json")
FPS = 2
IMAGE_QUALITY = 2       # 1(최고)~31(최저)
RESIZE_SHORT = 256      # 짧은 변 기준 리사이즈 (px)


def load_selected_videos():
    """selected_video_ids + JSON + label_foodtype 로 처리 대상 영상 메타 리스트 생성."""
    with SELECTED_IDS_CONFIG.open(encoding="utf-8") as f:
        selected_ids = json.load(f)["video_ids"]
    with ANNOTATIONS_JSON.open(encoding="utf-8") as f:
        db = json.load(f)["database"]
    name_by_type = {}
    with FOODTYPE_PATH.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                name_by_type[int(row[0])] = row[1]
    videos = []
    for vid in selected_ids:
        entry = db.get(vid)
        if entry is None:
            videos.append({"video_id": vid, "category": "unknown"})
            continue
        rtype = int(entry["recipe_type"])
        videos.append({
            "video_id": vid,
            "category": name_by_type.get(rtype, f"recipe_{rtype}"),
        })
    return videos


def find_video_file(video_id):
    """data/videos/ 평탄 구조(download_videos.py 산출물)에서 영상 파일 탐색."""
    for ext in (".mp4", ".mkv", ".webm"):
        path = os.path.join(VIDEO_DIR, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def get_video_fps(video_path):
    """원본 영상의 FPS 확인"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fps_str = result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            return float(fps_str)
    except:
        pass
    return None


def extract_frames(video_path, output_dir):
    """
    ffmpeg로 2 FPS 프레임 추출
    - fps 필터로 정확한 시간 간격 추출
    - 짧은 변 기준 256px 리사이즈 (비율 유지)
    - showinfo 필터로 프레임별 타임스탬프 기록
    """
    os.makedirs(output_dir, exist_ok=True)

    # vf 필터 구성:
    # 1) fps=2: 초당 2프레임 추출
    # 2) scale: 짧은 변을 256px로 리사이즈 (비율 유지, 2의 배수로 맞춤)
    # 3) showinfo: 각 프레임의 타임스탬프를 stderr에 출력
    vf_filter = (
        f"fps={FPS},"
        f"scale='if(gt(iw,ih),-2,{RESIZE_SHORT})':'if(gt(iw,ih),{RESIZE_SHORT},-2)',"
        f"showinfo"
    )

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", vf_filter,
        "-q:v", str(IMAGE_QUALITY),
        "-start_number", "0",
        os.path.join(output_dir, "frame_%06d.jpg"),
        "-y",
        "-loglevel", "info"   # showinfo 출력을 위해 info 레벨 필요
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # 추출된 프레임 수 확인
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        frame_count = len(frame_files)

        if frame_count == 0:
            # 실패 시 빈 폴더 삭제
            shutil.rmtree(output_dir, ignore_errors=True)
            return False, result.stderr[:200] if result.stderr else "프레임 추출 실패"

        # showinfo에서 타임스탬프 파싱
        timestamps = []
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                try:
                    pts_part = line.split("pts_time:")[1]
                    pts_time = float(pts_part.split()[0])
                    timestamps.append(pts_time)
                except (IndexError, ValueError):
                    continue

        # 타임스탬프가 부족하면 계산으로 보완
        if len(timestamps) < frame_count:
            timestamps = [i / FPS for i in range(frame_count)]

        # 타임스탬프 저장
        timestamp_data = {}
        for i, ts in enumerate(timestamps[:frame_count]):
            frame_name = f"frame_{i:06d}.jpg"
            timestamp_data[frame_name] = {
                "frame_index": i,
                "timestamp_sec": round(ts, 4),
            }

        with open(os.path.join(output_dir, "timestamps.json"), "w") as f:
            json.dump(timestamp_data, f, indent=2)

        return True, frame_count

    except subprocess.TimeoutExpired:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, "타임아웃"
    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, str(e)


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    all_videos = load_selected_videos()
    total = len(all_videos)
    success = 0
    failed = []
    skipped = 0
    total_frames = 0

    print(f"총 {total}개 영상에서 {FPS} FPS 프레임 추출 시작")
    print(f"리사이즈: 짧은 변 {RESIZE_SHORT}px")
    print("=" * 55)

    for i, vid in enumerate(all_videos, 1):
        video_id = vid["video_id"]
        output_dir = os.path.join(FRAMES_DIR, video_id)

        # 이미 추출 완료된 경우 건너뛰기 (timestamps.json 존재 여부로 판단)
        ts_file = os.path.join(output_dir, "timestamps.json")
        if os.path.exists(ts_file):
            frame_count = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
            total_frames += frame_count
            skipped += 1
            success += 1
            continue

        # 불완전한 이전 시도 정리
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        # 영상 파일 찾기
        video_path = find_video_file(video_id)
        if video_path is None:
            failed.append({"video_id": video_id, "category": vid["category"], "reason": "영상 파일 없음"})
            print(f"[{i}/{total}] {video_id} - 영상 파일 없음")
            continue

        # 원본 FPS 확인
        original_fps = get_video_fps(video_path)
        fps_info = f", 원본 {original_fps:.1f}fps" if original_fps else ""

        # 프레임 추출
        print(f"[{i}/{total}] {video_id} ({vid['category']}{fps_info}) ...", end=" ", flush=True)
        ok, result = extract_frames(video_path, output_dir)

        if ok:
            success += 1
            total_frames += result
            print(f"성공 ({result}프레임)")
        else:
            failed.append({"video_id": video_id, "category": vid["category"], "reason": str(result)})
            print(f"실패: {result}")

    # 결과 출력
    print()
    print("=" * 55)
    print(f"프레임 추출 완료")
    print(f"  성공: {success}개 (건너뜀: {skipped}개 포함)")
    print(f"  실패: {len(failed)}개")
    print(f"  총 프레임 수: {total_frames:,}개")
    print(f"  저장 위치: {FRAMES_DIR}/")

    if failed:
        print(f"\n실패 목록:")
        for item in failed:
            print(f"  {item['video_id']} ({item['category']}): {item['reason']}")

    # 통계 저장
    stats = {
        "fps": FPS,
        "resize_short_side": RESIZE_SHORT,
        "total_videos": total,
        "success": success,
        "failed_count": len(failed),
        "total_frames": total_frames,
        "failed_list": failed,
    }
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
