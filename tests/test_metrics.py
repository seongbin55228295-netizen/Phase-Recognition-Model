"""Unit tests for evaluation metrics.

Run with either:
    python -m pytest tests/test_metrics.py
    python tests/test_metrics.py        # plain-assert fallback (no pytest needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # noqa: E402
    edit_score,
    frame_accuracy,
    macro_f1,
    segment_iou,
)


def test_frame_accuracy_perfect_and_weighted():
    assert frame_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert frame_accuracy([1, 1], [1, 2]) == 0.5
    # weighted: second (wrong) position carries 3x weight -> 1/(1+3)
    assert frame_accuracy([1, 1], [1, 2], weights=[1.0, 3.0]) == 0.25
    assert frame_accuracy([], []) == 0.0


def test_macro_f1_perfect_and_single_class_collapse():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == 1.0
    # predicting one class for a two-class balanced target:
    # class0 F1 = 2*1*0.5/(1.5)=0.666..., class1 F1 = 0 -> mean 0.333...
    f1 = macro_f1([0, 0, 0, 0], [0, 0, 1, 1])
    assert abs(f1 - (2 / 3) / 2) < 1e-9


def test_segment_iou_perfect_disjoint_and_hallucination():
    assert segment_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    # fully swapped two classes -> every class IoU 0
    assert segment_iou([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0
    # [0,0] vs [0,1]: class0 inter=1 union=2 ->0.5 ; class1 inter=0 union=1 ->0 ; mean 0.25
    assert abs(segment_iou([0, 0], [0, 1]) - 0.25) < 1e-9
    # hallucinated class 2 (never in GT) is penalized: drags mean down
    iou = segment_iou([0, 2], [0, 0])
    # class0: inter=1 union=2 ->0.5 ; class2: inter=0 union=1 ->0 ; mean 0.25
    assert abs(iou - 0.25) < 1e-9
    assert segment_iou([], []) == 0.0


def test_edit_score_perfect_reversed_and_oversegmentation():
    # identical segment structure -> 100
    assert edit_score([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]) == 100.0
    # run-length collapse ignores within-segment frame counts
    assert edit_score([0, 1, 2], [0, 0, 1, 1, 1, 2]) == 100.0
    # reversed three-segment order: [0,1,2] vs [2,1,0], distance 2 over len 3
    assert abs(edit_score([0, 1, 2], [2, 1, 0]) - (1 - 2 / 3) * 100) < 1e-9
    # over-segmentation: predicting an extra flicker segment
    # pred segs [0,1,0] vs gt [0]: distance 2 over max-len 3 -> (1-2/3)*100
    assert abs(edit_score([0, 1, 0], [0, 0, 0]) - (1 - 2 / 3) * 100) < 1e-9
    assert edit_score([], []) == 100.0


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} metric tests passed.")


if __name__ == "__main__":
    _run_all()
