"""Unit tests for the preprocessing labeling logic extracted into src/preprocessing/.

Covers the domain-critical functions that used to live inside the scripts and
were therefore untestable:
  - src/preprocessing/annotation_labeling.py : prototype review routing + auto/review split
  - src/preprocessing/frame_labeling.py      : segment merge/partition + Soft Boundary weights

Needs only numpy + pandas (no torch / torchvision / sentence-transformers): the
embedding model is injected, so none of the tested functions construct one.

Run:
    python tests/test_preprocessing_labeling.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.annotation_labeling import (  # noqa: E402
    AUTO_QUALITY_RULES,
    REVIEW_RULES,
    classify_quality,
    evaluate_review_rules,
    load_annotation_rows,
    mean_normalized_vector,
    select_rows,
    split_rows,
)
from src.preprocessing.frame_labeling import (  # noqa: E402
    assert_partition,
    expand_video,
    load_timestamps,
    merge_segments,
    normalize_acts,
    normalize_reviewed,
)


# --------------------------------------------------------------------------- #
# annotation_labeling
# --------------------------------------------------------------------------- #

def _scored_row(sentence, top1, margin, label, **over):
    """A fully-populated scored row (all keys split_rows / review rules need)."""
    row = {
        "video_id": "vidA", "subset": "training", "recipe_type": 1,
        "segment_id": 0, "segment_start": 0.0, "segment_end": 5.0,
        "sentence": sentence, "predicted_label": label,
        "top1_score": top1, "margin": margin, "second_label": "Idle",
    }
    row.update(over)
    return row


def test_select_rows_all_returns_input():
    rows = [{"i": i} for i in range(5)]
    assert select_rows(rows, "all", 42) is rows
    assert select_rows(rows, "ALL", 42) is rows  # case-insensitive


def test_select_rows_sample_is_deterministic_and_capped():
    rows = [{"i": i} for i in range(10)]
    a = select_rows(rows, 4, seed=7)
    b = select_rows(rows, 4, seed=7)
    assert len(a) == 4 and a == b               # deterministic for a fixed seed
    assert len(select_rows(rows, 999, seed=7)) == 10  # capped at len(rows)


def test_mean_normalized_vector_is_unit_norm():
    v = mean_normalized_vector([[3.0, 0.0], [0.0, 0.0]])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    # all-zero input stays zero (no divide-by-zero)
    z = mean_normalized_vector([[0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(z, 0.0)


def test_review_clean_row_passes():
    reasons, issues = evaluate_review_rules(
        _scored_row("slice the onion", top1=0.6, margin=0.2, label="Cut"), REVIEW_RULES)
    assert reasons == [] and issues == []


def test_review_low_top1():
    reasons, _ = evaluate_review_rules(
        _scored_row("boil the pasta", top1=0.30, margin=0.2, label="Cook-Heat"), REVIEW_RULES)
    assert reasons == ["top1_below_0.35"]


def test_review_low_margin():
    reasons, _ = evaluate_review_rules(
        _scored_row("boil the pasta", top1=0.6, margin=0.005, label="Cook-Heat"), REVIEW_RULES)
    assert reasons == ["margin_below_0.01"]


def test_review_sensitive_label_low_confidence():
    # margin in (0.01, 0.05): margin rule skipped, sensitive-label rule fires.
    reasons, _ = evaluate_review_rules(
        _scored_row("plate the dish", top1=0.6, margin=0.03, label="Plate"), REVIEW_RULES)
    assert reasons == ["low_confidence_sensitive_label"]


def test_review_multiple_action_groups():
    reasons, _ = evaluate_review_rules(
        _scored_row("chop the onion and mix the batter", top1=0.6, margin=0.2, label="Cut"),
        REVIEW_RULES)
    assert reasons == ["multiple_action_groups"]


def test_review_generic_context_verb():
    # starts with "add" + margin < 0.05 (but >= 0.01) -> generic verb rule only.
    reasons, _ = evaluate_review_rules(
        _scored_row("add the salt to the pan", top1=0.6, margin=0.03, label="Season"),
        REVIEW_RULES)
    assert reasons == ["generic_context_verb_very_low_margin"]


def test_classify_quality_boundaries():
    good = AUTO_QUALITY_RULES["good"]
    assert classify_quality({"top1_score": 0.6, "margin": 0.1}, good) == ("good", 1.0)
    assert classify_quality({"top1_score": 0.49, "margin": 0.1}, good) == (None, None)  # top1 too low
    assert classify_quality({"top1_score": 0.6, "margin": 0.04}, good) == (None, None)  # margin too low


def test_split_rows_partition_and_weights():
    good = _scored_row("slice the onion", top1=0.6, margin=0.2, label="Cut", segment_id=1)
    weak = _scored_row("boil the pasta", top1=0.6, margin=0.03, label="Cook-Heat", segment_id=2)
    review = _scored_row("boil the pasta", top1=0.2, margin=0.2, label="Cook-Heat", segment_id=3)

    acts_rows, review_rows = split_rows([good, weak, review], REVIEW_RULES, AUTO_QUALITY_RULES)

    # every input lands in exactly one track
    assert len(acts_rows) + len(review_rows) == 3
    assert len(acts_rows) == 2 and len(review_rows) == 1

    by_seg = {r["segment_id"]: r for r in acts_rows}
    assert by_seg[1]["label_quality"] == "good" and by_seg[1]["recommended_sample_weight"] == 1.0
    assert by_seg[2]["label_quality"] == "weak" and by_seg[2]["recommended_sample_weight"] == 0.5
    assert review_rows[0]["segment_id"] == 3 and "review_reason" in review_rows[0]


def test_load_annotation_rows_filters_failed_and_flattens():
    db = {
        "database": {
            "vidGood": {"subset": "training", "recipe_type": 5, "annotations": [
                {"id": 0, "segment": [0, 3], "sentence": "chop the onion"},
                {"id": 1, "segment": [3, 6], "sentence": ""},          # empty -> skipped
            ]},
            "vidFail": {"subset": "training", "recipe_type": 5, "annotations": [
                {"id": 0, "segment": [0, 3], "sentence": "mix it"},     # whole video filtered out
            ]},
        }
    }
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ann.json").write_text(json.dumps(db), encoding="utf-8")
        (d / "failed.json").write_text(json.dumps({"failed_video_ids": ["vidFail"]}), encoding="utf-8")
        rows = load_annotation_rows(d / "ann.json", d / "failed.json")

    assert len(rows) == 1
    assert rows[0]["video_id"] == "vidGood" and rows[0]["segment_id"] == 0
    assert rows[0]["recipe_type"] == 5 and rows[0]["sentence"] == "chop the onion"


# --------------------------------------------------------------------------- #
# frame_labeling
# --------------------------------------------------------------------------- #

def _frames(timestamps):
    return pd.DataFrame({
        "frame_index": list(range(len(timestamps))),
        "frame_name": [f"frame_{i:06d}.jpg" for i in range(len(timestamps))],
        "timestamp_sec": [float(t) for t in timestamps],
    })


def _one_segment(start, end, label="Cut", source="auto_good", weight=1.0, seg_id=0):
    return pd.DataFrame([{
        "video_id": "vidA", "segment_id": seg_id,
        "segment_start": start, "segment_end": end,
        "label": label, "source": source, "sample_weight": weight,
    }])


def test_normalize_acts_maps_columns_and_source():
    acts = pd.DataFrame([{
        "video_id": "v", "segment_id": 0, "subset": "training",
        "predicted_label": "Cut", "recommended_sample_weight": 0.5, "label_quality": "weak",
    }])
    out = normalize_acts(acts).iloc[0]
    assert out["label"] == "Cut" and out["sample_weight"] == 0.5
    assert out["source"] == "auto_weak"
    assert "label_quality" not in out.index


def test_normalize_reviewed_forces_weight_one():
    reviewed = pd.DataFrame([{"video_id": "v", "segment_id": 0, "predicted_label": "Mix"}])
    out = normalize_reviewed(reviewed).iloc[0]
    assert out["label"] == "Mix" and out["source"] == "reviewed" and out["sample_weight"] == 1.0


def test_assert_partition_detects_overlap():
    acts = pd.DataFrame([{"video_id": "v", "segment_id": 0}, {"video_id": "v", "segment_id": 1}])
    disjoint = pd.DataFrame([{"video_id": "v", "segment_id": 2}])
    overlap = pd.DataFrame([{"video_id": "v", "segment_id": 1}])
    assert_partition(acts, disjoint)  # no raise
    try:
        assert_partition(acts, overlap)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on overlapping (video_id, segment_id)")


def test_expand_video_soft_boundary_weights():
    # 7 frames fully inside one segment; boundary = 2 frames at each edge, factor 0.5.
    frames = _frames(range(7))                 # ts 0..6
    seg = _one_segment(0, 7, weight=1.0)
    out = expand_video("vidA", seg, frames, boundary_width=2, boundary_factor=0.5)

    assert len(out) == 7
    boundary_idx = set(out.loc[out["is_boundary"], "frame_index"])
    assert boundary_idx == {0, 1, 5, 6}        # first 2 + last 2
    w = dict(zip(out["frame_index"], out["sample_weight"]))
    assert all(w[i] == 0.5 for i in (0, 1, 5, 6))
    assert all(w[i] == 1.0 for i in (2, 3, 4))


def test_expand_video_boundary_scales_with_segment_weight():
    # auto_weak segment (0.5) -> boundary frames get 0.5 * 0.5 = 0.25.
    frames = _frames(range(7))
    seg = _one_segment(0, 7, source="auto_weak", weight=0.5)
    out = expand_video("vidA", seg, frames, boundary_width=2, boundary_factor=0.5)
    w = dict(zip(out["frame_index"], out["sample_weight"]))
    assert w[0] == 0.25 and w[3] == 0.5


def test_expand_video_excludes_gap_and_end_frames():
    # segment [0, 7): frames at ts 7 and 8 (gap/end) must NOT be emitted.
    frames = _frames(range(9))                 # ts 0..8
    seg = _one_segment(0, 7, weight=1.0)
    out = expand_video("vidA", seg, frames, boundary_width=2, boundary_factor=0.5)
    assert list(out["frame_index"]) == list(range(7))   # 7 excluded (>= end), 8 is a gap


def test_expand_video_dedups_overlapping_segments():
    # two overlapping segments share the frame at ts 1; output must be deduped.
    frames = _frames([0, 1, 2])
    segs = pd.concat([_one_segment(0, 2, seg_id=0), _one_segment(1, 3, seg_id=1)],
                     ignore_index=True)
    out = expand_video("vidA", segs, frames, boundary_width=0, boundary_factor=0.5)
    assert out["frame_index"].is_unique
    assert set(out["frame_index"]) == {0, 1, 2}


def test_load_timestamps_sorted():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "timestamps.json").write_text(json.dumps({
            "frame_000001.jpg": {"frame_index": 1, "timestamp_sec": 0.5},
            "frame_000000.jpg": {"frame_index": 0, "timestamp_sec": 0.0},
        }), encoding="utf-8")
        df = load_timestamps(d)
    assert list(df["frame_index"]) == [0, 1]            # sorted ascending
    assert list(df.columns) == ["frame_index", "frame_name", "timestamp_sec"]


def test_merge_segments_combines_and_sorts():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        acts = pd.DataFrame([{
            "video_id": "v2", "subset": "validation", "segment_id": 0,
            "segment_start": 0, "segment_end": 2, "sentence": "x",
            "predicted_label": "Cut", "recommended_sample_weight": 1.0, "label_quality": "good",
        }])
        reviewed = pd.DataFrame([{
            "video_id": "v1", "subset": "training", "segment_id": 0,
            "segment_start": 0, "segment_end": 2, "sentence": "y", "predicted_label": "Mix",
        }])
        acts.to_csv(d / "acts.csv", index=False, encoding="utf-8-sig")
        reviewed.to_csv(d / "reviewed.csv", index=False, encoding="utf-8-sig")
        merged = merge_segments(d / "acts.csv", d / "reviewed.csv")

    assert len(merged) == 2
    assert set(merged["source"]) == {"auto_good", "reviewed"}
    # sorted by subset -> training ("v1") before validation ("v2")
    assert list(merged["video_id"]) == ["v1", "v2"]


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} preprocessing-labeling tests passed.")


if __name__ == "__main__":
    _run_all()
