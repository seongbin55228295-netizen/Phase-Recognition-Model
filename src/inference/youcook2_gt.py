"""Per-frame pseudo ground-truth for a held-out YouCook2 video.

For quantitative evaluation we need 8-class frame labels for a test video. Those
are produced exactly the way the *training* labels were, minus the human-review
step (which only existed for the selected 300):

  1. Each YouCook2 annotation sentence is mapped to one of the 8 classes by
     cosine similarity to the prototype class vectors  (== generate_annotation_labels.py).
  2. Segments are expanded to per-frame labels via the extracted timestamps,
     with Soft Boundary weighting at segment edges  (== generate_frame_labels.py).

⚠️ These are *pseudo-labels* (auto top-1, no human confirmation). Metrics computed
against them measure agreement with the automatic labeling pipeline, not against a
gold human standard. Report accordingly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Mirrors AUTO_QUALITY_RULES in src/preprocessing/annotation_labeling.py.
GOOD_TOP1_MIN = 0.50
GOOD_MARGIN_MIN = 0.05
WEIGHT_GOOD = 1.0
WEIGHT_WEAK = 0.5

GT_COLUMNS = [
    "frame_index", "frame_name", "timestamp_sec",
    "gt_label", "gt_quality", "gt_weight", "is_boundary", "segment_id",
]


def load_entry(video_id: str, annotations_json: str | Path) -> dict | None:
    """Return the YouCook2 database entry for video_id, or None if absent."""
    with open(annotations_json, encoding="utf-8") as f:
        db = json.load(f)["database"]
    return db.get(video_id)


def _mean_normalized(vectors: np.ndarray) -> np.ndarray:
    mean_vector = np.asarray(vectors).mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    return mean_vector / norm if norm else mean_vector


def _score_segments(sentences: list[str], prototypes_path: str | Path) -> list[dict]:
    """Top-1 class + margin for each sentence (lazy sentence-transformers import)."""
    from sentence_transformers import SentenceTransformer  # heavy; only on this path

    with open(prototypes_path, encoding="utf-8") as f:
        cfg = json.load(f)
    labels = cfg["labels"]
    model = SentenceTransformer(cfg["embedding_model"])

    class_matrix = np.stack([
        _mean_normalized(model.encode(
            cfg["prototype_sentences"][lbl],
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        ))
        for lbl in labels
    ])
    sent_emb = model.encode(
        sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )

    out = []
    for emb in sent_emb:
        scores = class_matrix @ emb
        order = np.argsort(scores)[::-1]
        top, second = int(order[0]), int(order[1])
        out.append({
            "predicted_label": labels[top],
            "top1_score": float(scores[top]),
            "margin": float(scores[top] - scores[second]),
        })
    return out


def _segment_weight(top1: float, margin: float) -> tuple[str, float]:
    if top1 >= GOOD_TOP1_MIN and margin >= GOOD_MARGIN_MIN:
        return "good", WEIGHT_GOOD
    return "weak", WEIGHT_WEAK


def build_youcook2_gt(
    video_id: str,
    frames: pd.DataFrame,
    *,
    annotations_json: str | Path,
    prototypes_path: str | Path,
    boundary_width: int = 2,
    boundary_factor: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Build per-frame pseudo-GT for `video_id`.

    Args:
        frames: DataFrame with columns frame_index, frame_name, timestamp_sec
                (one row per extracted frame, the inference sequence).
    Returns:
        (gt_df, meta). gt_df has GT_COLUMNS, one row per *covered* frame (frames in
        annotation gaps are omitted, matching training). meta carries subset /
        recipe_type / n_segments for reporting.
    """
    entry = load_entry(video_id, annotations_json)
    if entry is None:
        raise KeyError(f"{video_id} not found in {annotations_json}")

    anns = [a for a in entry.get("annotations", []) if a.get("sentence")]
    if not anns:
        raise ValueError(f"{video_id} has no usable annotation sentences")

    scored = _score_segments([a["sentence"] for a in anns], prototypes_path)

    # Build per-segment records, sorted by segment id (earlier id wins on overlap).
    segments = []
    for ann, sc in zip(anns, scored):
        quality, weight = _segment_weight(sc["top1_score"], sc["margin"])
        start, end = ann["segment"]
        segments.append({
            "segment_id": int(ann["id"]),
            "segment_start": float(start),
            "segment_end": float(end),
            "label": sc["predicted_label"],
            "quality": quality,
            "weight": weight,
        })
    segments.sort(key=lambda s: s["segment_id"])

    frames = frames.sort_values("frame_index").reset_index(drop=True)
    rows = []
    for seg in segments:
        in_seg = frames[(frames["timestamp_sec"] >= seg["segment_start"])
                        & (frames["timestamp_sec"] < seg["segment_end"])].reset_index(drop=True)
        n = len(in_seg)
        if n == 0:
            continue
        for i, frame in in_seg.iterrows():
            distance_from_edge = min(i, n - 1 - i)
            is_boundary = bool(distance_from_edge < boundary_width)
            weight = seg["weight"] * (boundary_factor if is_boundary else 1.0)
            rows.append({
                "frame_index": int(frame["frame_index"]),
                "frame_name": frame["frame_name"],
                "timestamp_sec": float(frame["timestamp_sec"]),
                "gt_label": seg["label"],
                "gt_quality": seg["quality"],
                "gt_weight": round(weight, 6),
                "is_boundary": is_boundary,
                "segment_id": seg["segment_id"],
            })

    if rows:
        gt_df = (pd.DataFrame(rows)
                 .sort_values("frame_index")
                 .drop_duplicates("frame_index", keep="first")   # earlier segment_id wins
                 .reset_index(drop=True))[GT_COLUMNS]
    else:
        gt_df = pd.DataFrame(columns=GT_COLUMNS)

    meta = {
        "video_id": video_id,
        "subset": entry.get("subset"),
        "recipe_type": int(entry["recipe_type"]) if "recipe_type" in entry else None,
        "n_segments": len(segments),
        "n_frames_total": int(len(frames)),
        "n_frames_covered": int(len(gt_df)),
    }
    return gt_df, meta
