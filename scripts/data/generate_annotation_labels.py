"""
YouCook2 annotation 을 프로토타입 임베딩으로 자동 분류한 뒤,
검수 규칙·임계값에 따라 자동 학습용/수동 검수용으로 분기한다.

이 파일은 얇은 CLI 래퍼다 — 프로젝트 경로 배선, 임베딩 모델 생성, CSV 쓰기만
담당하고, 분류·검수 분기 로직은 src/preprocessing/annotation_labeling.py 에 있다.

[입력]
  data/external/YouCookII/annotations/youcookii_annotations_trainval.json  ← 원본 annotation
  data/processed/failed_downloads.json                       ← 다운로드 실패 영상 제외 필터
  configs/action_class_prototypes.json                       ← 8개 클래스 프로토타입 문장

[출력]
  data/processed/action_annotations.csv
    컬럼: video_id, subset, recipe_type, segment_id,
          segment_start, segment_end, sentence,
          predicted_label, label_quality, recommended_sample_weight
  data/processed/review_queue.csv
    컬럼: video_id, subset, recipe_type, segment_id,
          segment_start, segment_end, sentence,
          predicted_label, review_reason, suggested_issue

[규칙·임계값]
  src/preprocessing/annotation_labeling.py 의 REVIEW_RULES, AUTO_QUALITY_RULES
  상수로 관리한다. 재조정 시 해당 상수를 직접 편집한다.

[사용법]
  python scripts/data/generate_annotation_labels.py
  python scripts/data/generate_annotation_labels.py --sample-size 200  # 빠른 검증
  python scripts/data/generate_annotation_labels.py --dry-run
"""
import argparse
import csv
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 분류·검수 분기 로직 단일 출처: src/preprocessing/annotation_labeling.py
from src.preprocessing.annotation_labeling import (
    ACTS_COLUMNS,
    AUTO_QUALITY_RULES,
    REVIEW_COLUMNS,
    REVIEW_RULES,
    build_class_vectors,
    load_annotation_rows,
    load_prototype_config,
    score_rows,
    select_rows,
    split_rows,
)

PROTOTYPES_PATH = ROOT / "configs" / "action_class_prototypes.json"
ANNOTATIONS_JSON = ROOT / "data" / "external" / "YouCookII" / "annotations" / "youcookii_annotations_trainval.json"
FAILED_DOWNLOADS = ROOT / "data" / "processed" / "failed_downloads.json"
ACTS_OUT = ROOT / "data" / "processed" / "action_annotations.csv"
REVIEW_OUT = ROOT / "data" / "processed" / "review_queue.csv"

DEFAULT_SAMPLE_SIZE = "all"
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", default=DEFAULT_SAMPLE_SIZE,
                        help="Number of annotation rows to sample, or 'all' for the full input.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed used when --sample-size is a number.")
    parser.add_argument("--acts-out", default=str(ACTS_OUT))
    parser.add_argument("--review-out", default=str(REVIEW_OUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary, do not write outputs.")
    return parser.parse_args()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    prototype_config = load_prototype_config(PROTOTYPES_PATH)
    labels = prototype_config["labels"]
    model_name = prototype_config["embedding_model"]
    prototype_sentences = prototype_config["prototype_sentences"]

    annotation_rows = load_annotation_rows(ANNOTATIONS_JSON, FAILED_DOWNLOADS)
    selected = select_rows(annotation_rows, args.sample_size, args.seed)

    model = SentenceTransformer(model_name)
    class_vectors = build_class_vectors(model, labels, prototype_sentences)
    scored = score_rows(model, selected, labels, class_vectors)

    acts_rows, review_rows = split_rows(scored, REVIEW_RULES, AUTO_QUALITY_RULES)

    good_count = sum(1 for r in acts_rows if r["label_quality"] == "good")
    weak_count = sum(1 for r in acts_rows if r["label_quality"] == "weak")

    print(f"annotation_file       : {ANNOTATIONS_JSON}")
    print(f"failed_downloads      : {FAILED_DOWNLOADS}")
    print(f"embedding_model       : {model_name}")
    print(f"class_count           : {len(labels)}")
    print(f"input rows            : {len(annotation_rows)}")
    print(f"scored rows           : {len(scored)}")
    print(f"action_annotations    : {len(acts_rows)} (good={good_count}, weak={weak_count})")
    print(f"review_queue          : {len(review_rows)}")
    print(f"check: acts + review  = {len(acts_rows) + len(review_rows)} (must equal {len(scored)})")

    if args.dry_run:
        print("dry-run: not writing outputs.")
        return

    write_csv(args.acts_out, acts_rows, ACTS_COLUMNS)
    write_csv(args.review_out, review_rows, REVIEW_COLUMNS)
    print(f"wrote: {args.acts_out}")
    print(f"wrote: {args.review_out}")

    if len(acts_rows) + len(review_rows) != len(scored):
        print("WARNING: acts + review != scored. Check logic.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
