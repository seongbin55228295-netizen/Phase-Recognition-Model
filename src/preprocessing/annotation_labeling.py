"""Prototype-embedding annotation labeling logic.

Single source of truth for turning raw YouCook2 annotation sentences into the
8-class auto/review split that scripts/data/generate_annotation_labels.py writes out.

Kept import-light: only numpy is imported here. The embedding model is *injected*
(``build_class_vectors``/``score_rows`` take a ``model`` argument) rather than
imported, so the routing/quality rules can be unit-tested without
sentence-transformers. The thin CLI (scripts/data/generate_annotation_labels.py)
wires project paths, constructs the SentenceTransformer, and writes CSVs.

Pipeline:
    load_annotation_rows -> select_rows
        -> build_class_vectors + score_rows
        -> split_rows  (each scored row -> auto-train track or manual-review queue)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


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


def load_prototype_config(path):
    path = Path(path)
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
    json_path = Path(json_path)
    failed_path = Path(failed_path)
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
    if str(sample_size).lower() == "all":
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
