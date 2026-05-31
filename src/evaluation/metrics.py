"""Frame-level metrics.

Segment IoU and Edit Distance will be added once basic training is verified.
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
