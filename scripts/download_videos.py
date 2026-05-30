"""
YouCook2 영상 다운로드 스크립트 (Windows 호환)
사전 설치: pip install yt-dlp / 시스템에 ffmpeg

다운로드 대상 URL 은 YouCookII/annotations/youcookii_annotations_trainval.json 의
각 항목 video_url 필드에서 직접 추출한다 (별도 매니페스트 파일 불필요).

YouTube 봇 차단 우회를 위해 cookies.txt(Netscape 포맷)를 사용할 수 있다.
- 우선순위: --cookies CLI 인자 > 환경변수 YT_COOKIES_FILE > secrets/cookies.txt 기본 경로
- 파일이 없거나 비어 있으면 쿠키 없이 진행한다 (대량 다운로드 시 차단 가능).
- cookies.txt 는 본인의 YouTube 세션 자격증명이므로 절대 커밋하지 말 것
  ([.gitignore](../.gitignore) 에 secrets/ 와 cookies.txt 등록됨).
  생성 방법은 README 의 "2(a) 영상 다운로드" 절을 참고.

사용법:
  python scripts/download_videos.py
  python scripts/download_videos.py --cookies path/to/cookies.txt
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# === 설정 ===
ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_JSON = ROOT / "YouCookII" / "annotations" / "youcookii_annotations_trainval.json"
OUTPUT_DIR = str(ROOT / "data" / "videos")                       # 영상 저장 폴더
LOG_FILE = str(ROOT / "reports" / "logs" / "download_log.txt")   # 로그 파일
FAILED_FILE = str(ROOT / "processed" / "failed_downloads.json")  # 실패 목록 (JSON)
DEFAULT_COOKIES_PATH = ROOT / "secrets" / "cookies.txt"          # 기본 쿠키 경로 (.gitignore)


def resolve_cookies(cli_path: str | None) -> str | None:
    """쿠키 파일 경로를 결정한다. 존재하지 않으면 None 반환."""
    candidates = [
        cli_path,
        os.environ.get("YT_COOKIES_FILE"),
        str(DEFAULT_COOKIES_PATH),
    ]
    for path in candidates:
        if path and Path(path).is_file() and Path(path).stat().st_size > 0:
            return path
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cookies", default=None,
                        help="Path to a Netscape-format cookies.txt for YouTube auth bypass.")
    args = parser.parse_args()

    cookies_path = resolve_cookies(args.cookies)

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # URL 목록을 YouCookII 원본 JSON 에서 직접 추출
    with ANNOTATIONS_JSON.open(encoding="utf-8") as f:
        db = json.load(f)["database"]
    urls = [entry["video_url"] for entry in db.values() if entry.get("video_url")]

    total = len(urls)
    print(f"총 {total}개 영상 다운로드 시작")
    print(f"저장 폴더: {OUTPUT_DIR}/")
    if cookies_path:
        print(f"쿠키 사용: {cookies_path}")
    else:
        print("쿠키 미사용 (대량 다운로드 시 봇 차단 가능). README 참고하여 cookies.txt 준비 권장.")
    print("=" * 50)

    # 이미 다운로드된 파일 확인
    existing = set()
    if os.path.isdir(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            if fname.endswith(".mp4"):
                vid_id = fname.replace(".mp4", "")
                existing.add(vid_id)
    print(f"이미 다운로드된 영상: {len(existing)}개 (건너뜀)")
    print("=" * 50)

    success = len(existing)
    failed = []
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log = open(LOG_FILE, "a", encoding="utf-8")

    for i, url in enumerate(urls, 1):
        # URL에서 YouTube ID 추출
        vid_id = url.split("v=")[-1].split("&")[0]

        # 이미 있으면 건너뛰기
        if vid_id in existing:
            continue

        print(f"[{i}/{total}] 다운로드 중: {vid_id} ...", end=" ", flush=True)

        # yt-dlp 명령 구성
        output_path = os.path.join(OUTPUT_DIR, "%(id)s.%(ext)s")
        cmd = ["yt-dlp"]
        if cookies_path:
            cmd += ["--cookies", cookies_path]
        cmd += [
            "-o", output_path,
            "--format", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]",
            "--merge-output-format", "mp4",
            "--no-overwrites",
            "--retries", "3",
            "--socket-timeout", "30",
            url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )

            # 다운로드 성공 여부 확인
            expected_file = os.path.join(OUTPUT_DIR, f"{vid_id}.mp4")
            if os.path.exists(expected_file):
                success += 1
                print(f"성공 ({success}/{total})")
                log.write(f"[SUCCESS] {vid_id}\n")
            else:
                failed.append(vid_id)
                print(f"실패")
                log.write(f"[FAILED] {vid_id}: {result.stderr[:200]}\n")

        except subprocess.TimeoutExpired:
            failed.append(vid_id)
            print(f"타임아웃")
            log.write(f"[TIMEOUT] {vid_id}\n")

        except Exception as e:
            failed.append(vid_id)
            print(f"에러: {e}")
            log.write(f"[ERROR] {vid_id}: {str(e)}\n")

        # YouTube 차단 방지용 대기
        time.sleep(1)

    log.close()

    # 실패 목록 저장 (JSON)
    payload = {
        "schema_version": 1,
        "description": "video_ids of YouCook2 videos that failed to download via scripts/download_videos.py. Used by downstream preprocessing scripts to exclude these videos from the available annotation set.",
        "count": len(failed),
        "failed_video_ids": failed,
    }
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 최종 결과 출력
    print()
    print("=" * 50)
    print(f"다운로드 완료")
    print(f"  성공: {success}개")
    print(f"  실패: {len(failed)}개")
    print(f"  실패 목록: {FAILED_FILE}")
    print(f"  로그: {LOG_FILE}")


if __name__ == "__main__":
    main()
