"""
YouCook2 annotation 을 프로토타입 임베딩으로 자동 분류한 뒤,
내장된 검수 규칙·임계값에 따라 자동 학습용/수동 검수용으로 분기한다.

[입력]
  YouCookII/annotations/youcookii_annotations_trainval.json  ← 원본 annotation
  processed/failed_downloads.json                            ← 다운로드 실패 영상 제외 필터
  configs/action_class_prototypes.json                       ← 8개 클래스 프로토타입 문장

[출력]
  processed/action_annotations.csv
    컬럼: video_id, subset, recipe_type, segment_id,
          segment_start, segment_end, sentence,
          predicted_label, label_quality, recommended_sample_weight
  processed/review_queue.csv
    컬럼: video_id, subset, recipe_type, segment_id,
          segment_start, segment_end, sentence,
          predicted_label, review_reason, suggested_issue

[규칙·임계값]
  본 스크립트 내부의 REVIEW_RULES, AUTO_QUALITY_RULES 상수로 관리한다.
  재조정 시 해당 상수를 직접 편집한다.

[사용법]
  python scripts/generate_annotation_labels.py
  python scripts/generate_annotation_labels.py --sample-size 200  # 빠른 검증
  python scripts/generate_annotation_labels.py --dry-run
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPES_PATH = ROOT / "configs" / "action_class_prototypes.json"
ANNOTATIONS_JSON = ROOT / "YouCookII" / "annotations" / "youcookii_annotations_trainval.json"
FAILED_DOWNLOADS = ROOT / "processed" / "failed_downloads.json"
ACTS_OUT = ROOT / "processed" / "action_annotations.csv"
REVIEW_OUT = ROOT / "processed" / "review_queue.csv"

DEFAULT_SAMPLE_SIZE = "all"
DEFAULT_SEED = 42

# === 검수 분기 규칙 ===
REVIEW_RULES = {
    "top1_threshold": {
        "value": 0.35,
        "reason_token": "top1_below_0.35",
        "message": "top similarity score is too low",
    },
    "margin_threshold": {
        "value": 0.01,
        "reason_token": "margin_below_0.01",
        "message": "top label is barely separated from second label",
    },
    "sensitive_labels": {
        "labels": ["Plate", "Idle", "Bake"],
        "margin_threshold": 0.05,
        "reason_token": "low_confidence_sensitive_label",
        "message_template": "{label} prediction is low-confidence and needs manual confirmation",
    },
    "generic_context_verbs": {
        "verbs": ["add", "place", "put", "pour"],
        "margin_threshold": 0.05,
        "reason_token": "generic_context_verb_very_low_margin",
        "message": "generic add/place/put/pour-style wording with very low margin",
    },
    "multi_action_connectors": {
        "patterns": [" and then ", " then ", " while ", " until ", " before ", " after "],
        "reason_token": "multi_action_connector",
        "message": "connector suggests a multi-step sentence",
    },
    "multiple_action_groups": {
        "min_distinct_groups": 2,
        "keywords": {
            "Prep": ["wash", "rinse", "peel", "crack", "measure", "drain", "soak", "thaw", "defrost", "prepare"],
            "Cut": ["chop", "slice", "dice", "cut", "mince", "julienne", "shred", "grate", "halve", "quarter"],
            "Mix": ["mix", "stir", "whisk", "fold", "combine", "blend", "toss", "beat", "swirl"],
            "Cook-Heat": ["cook", "fry", "saute", "sauté", "boil", "simmer", "steam", "grill", "heat", "brown", "sear"],
            "Bake": ["bake", "broil", "toast", "roast"],
            "Season": ["season", "sprinkle", "salt", "pepper", "drizzle", "glaze", "marinate"],
            "Plate": ["plate", "serve", "transfer", "arrange", "garnish", "dish"],
        },
        "reason_token": "multiple_action_groups",
        "message_template": "strong action groups detected: {groups}",
    },
}

AUTO_QUALITY_RULES = {
    "good": {
        "top1_min": 0.50,
        "margin_min": 0.05,
        "sample_weight": 1.0,
    },
    "weak": {
        "sample_weight": 0.5,
    },
}

ACTS_COLUMNS = [
    "video_id", "subset", "recipe_type", "segment_id",
    "segment_start", "segment_end", "sentence",
    "predicted_label", "label_quality", "recommended_sample_weight",
]
REVIEW_COLUMNS = [
    "video_id", "subset", "recipe_type", "segment_id",
    "segment_start", "segment_end", "sentence",
    "predicted_label", "review_reason", "suggested_issue",
]


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


def load_prototype_config(path):
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    labels = config["labels"]
    prototype_sentences = config["prototype_sentences"]
    missing_labels = [lbl for lbl in labels if lbl not in prototype_sentences]
    if missing_labels:
        raise ValueError(f"Missing prototype sentences for labels: {missing_labels}")
    return config


def load_annotation_rows(json_path, failed_path):
    """Load YouCook2 annotations as flat per-segment rows, filtered to downloaded videos."""
    with json_path.open(encoding="utf-8") as f:
        db = json.load(f)["database"]
    failed_ids = set()
    if failed_path.exists():
        with failed_path.open(encoding="utf-8") as f:
            failed_ids = set(json.load(f).get("failed_video_ids", []))

    rows = []
    for video_id, entry in db.items():
        if video_id in failed_ids:
            continue
        subset = entry["subset"]
        recipe_type = int(entry["recipe_type"])
        for ann in entry["annotations"]:
            sentence = ann.get("sentence", "")
            if not sentence:
                continue
            start, end = ann["segment"]
            rows.append({
                "video_id": video_id,
                "subset": subset,
                "recipe_type": recipe_type,
                "segment_id": ann["id"],
                "segment_start": start,
                "segment_end": end,
                "sentence": sentence,
            })
    return rows


def select_rows(rows, sample_size, seed):
    if sample_size.lower() == "all":
        return rows
    try:
        size = int(sample_size)
    except ValueError as exc:
        raise ValueError("--sample-size must be an integer or 'all'.") from exc
    if size < 1:
        raise ValueError("--sample-size must be positive, or use 'all'.")
    rng = random.Random(seed)
    return rng.sample(rows, min(size, len(rows)))


def mean_normalized_vector(vectors):
    matrix = np.asarray(vectors)
    mean_vector = matrix.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    return mean_vector / norm if norm else mean_vector


def build_class_vectors(model, labels, prototype_sentences):
    class_vectors = {}
    for label in labels:
        embeddings = model.encode(
            prototype_sentences[label],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        class_vectors[label] = mean_normalized_vector(embeddings)
    return class_vectors


def score_rows(model, rows, labels, class_vectors):
    """Attach predicted_label, top1_score, second_label, margin to each row in-place."""
    sentences = [row["sentence"] for row in rows]
    sentence_embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    class_matrix = np.stack([class_vectors[lbl] for lbl in labels])

    for row, emb in zip(rows, sentence_embeddings):
        scores = class_matrix @ emb
        order = np.argsort(scores)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1])
        row["predicted_label"] = labels[top_idx]
        row["top1_score"] = float(scores[top_idx])
        row["second_label"] = labels[second_idx]
        row["margin"] = float(scores[top_idx] - scores[second_idx])
    return rows


def evaluate_review_rules(row, rules):
    """Return (reasons, issues) for review routing. Empty lists => row passes to auto track."""
    reasons, issues = [], []
    sentence = str(row["sentence"]).lower()
    margin = float(row["margin"])
    top1 = float(row["top1_score"])
    label = row["predicted_label"]

    top1_rule = rules["top1_threshold"]
    if top1 < top1_rule["value"]:
        reasons.append(top1_rule["reason_token"])
        issues.append(top1_rule["message"])

    margin_rule = rules["margin_threshold"]
    if margin < margin_rule["value"]:
        reasons.append(margin_rule["reason_token"])
        issues.append(margin_rule["message"])

    sens = rules["sensitive_labels"]
    if label in sens["labels"] and margin < sens["margin_threshold"]:
        reasons.append(sens["reason_token"])
        issues.append(sens["message_template"].format(label=label))

    groups_rule = rules["multiple_action_groups"]
    detected = []
    for group_label, keywords in groups_rule["keywords"].items():
        if any(f" {kw} " in f" {sentence} " for kw in keywords):
            detected.append(group_label)
    if len(detected) >= groups_rule["min_distinct_groups"]:
        reasons.append(groups_rule["reason_token"])
        issues.append(groups_rule["message_template"].format(groups="|".join(detected)))

    conn_rule = rules["multi_action_connectors"]
    if any(p in f" {sentence} " for p in conn_rule["patterns"]):
        reasons.append(conn_rule["reason_token"])
        issues.append(conn_rule["message"])

    generic = rules["generic_context_verbs"]
    if margin < generic["margin_threshold"]:
        starts_with_generic = any(sentence.startswith(v + " ") for v in generic["verbs"])
        if starts_with_generic:
            reasons.append(generic["reason_token"])
            issues.append(generic["message"])

    return reasons, issues


def classify_quality(row, good_rule):
    if (float(row["top1_score"]) >= good_rule["top1_min"]
            and float(row["margin"]) >= good_rule["margin_min"]):
        return "good", good_rule["sample_weight"]
    return None, None


def split_rows(scored_rows, review_rules, auto_rules):
    good_rule = auto_rules["good"]
    weak_weight = auto_rules["weak"]["sample_weight"]

    acts_rows, review_rows = [], []
    for row in scored_rows:
        reasons, issues = evaluate_review_rules(row, review_rules)
        base = {col: row[col] for col in [
            "video_id", "subset", "recipe_type", "segment_id",
            "segment_start", "segment_end", "sentence", "predicted_label",
        ]}
        if reasons:
            review_rows.append({
                **base,
                "review_reason": ";".join(reasons),
                "suggested_issue": "; ".join(issues),
            })
        else:
            quality, weight = classify_quality(row, good_rule)
            if quality is None:
                quality, weight = "weak", weak_weight
            acts_rows.append({
                **base,
                "label_quality": quality,
                "recommended_sample_weight": weight,
            })
    return acts_rows, review_rows


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
