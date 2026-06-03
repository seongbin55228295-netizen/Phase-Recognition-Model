"""Evaluation metrics for phase recognition.

Frame-level: frame_accuracy (optionally sample-weight weighted), macro_f1.
Segment-level: segment_iou (per-class frame mIoU), edit_score (normalized
segment-sequence Levenshtein, the action-segmentation "Edit" score).

All functions take plain ``list[int]`` sequences and have no sklearn dependency.
Segment-level metrics expect a single video's contiguous frame sequence; the
evaluator computes them per video and averages across videos.
"""
from __future__ import annotations

from collections import Counter


def frame_accuracy(preds: list[int], labels: list[int], weights: list[float] | None = None) -> float:
    if not preds:
        return 0.0
    if weights is None:
        correct = sum(int(p == y) for p, y in zip(preds, labels))
        return correct / len(preds)
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    correct_w = sum(w for p, y, w in zip(preds, labels, weights) if p == y)
    return correct_w / total_w


def macro_f1(preds: list[int], labels: list[int]) -> float:
    """Macro-averaged F1 over classes that appear in `labels` (no sklearn dep)."""
    if not preds:
        return 0.0
    classes = sorted(set(labels) | set(preds))
    f1s = []
    pred_c = Counter(preds)
    true_c = Counter(labels)
    tp_c: Counter[int] = Counter()
    for p, y in zip(preds, labels):
        if p == y:
            tp_c[p] += 1
    for c in classes:
        tp = tp_c[c]
        fp = pred_c[c] - tp
        fn = true_c[c] - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0


def segment_iou(preds: list[int], labels: list[int]) -> float:
    """Mean per-class frame IoU (Jaccard) over a single video's frame sequence.

    For each class c appearing in either preds or labels:
        IoU_c = |{i: preds[i]==c and labels[i]==c}| / |{i: preds[i]==c or labels[i]==c}|
    Averaged over those classes. Hallucinated classes (predicted, never in GT)
    contribute IoU_c = 0, so they are penalized.
    """
    if not preds:
        return 0.0
    classes = set(labels) | set(preds)
    ious: list[float] = []
    for c in classes:
        inter = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        union = sum(1 for p, y in zip(preds, labels) if p == c or y == c)
        if union == 0:
            continue
        ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


def _run_length_segments(seq: list[int]) -> list[int]:
    """Collapse consecutive identical labels into a segment-label sequence.

    e.g. [0, 0, 1, 1, 1, 2] -> [0, 1, 2]
    """
    out: list[int] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _levenshtein(a: list[int], b: list[int]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def edit_score(preds: list[int], labels: list[int]) -> float:
    """Normalized segment-level edit score in [0, 100] (higher is better).

    Both frame sequences are run-length compressed into segment-label sequences,
    then the Levenshtein distance between them is normalized by the longer
    sequence: score = (1 - d / max(len_pred_seg, len_gt_seg)) * 100. Penalizes
    over-segmentation and ordering errors (Lea et al., action segmentation).
    """
    p_seg = _run_length_segments(preds)
    y_seg = _run_length_segments(labels)
    denom = max(len(p_seg), len(y_seg))
    if denom == 0:
        return 100.0
    d = _levenshtein(p_seg, y_seg)
    return (1.0 - d / denom) * 100.0
