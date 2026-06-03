"""Logic tests for scripts/evaluate.py that need no torchvision / GPU / checkpoints.

We stub ``torchvision`` (only its ``transforms`` name is touched at import time by
src.data.dataset) and drive the Evaluator with a tiny deterministic stub model +
stub tokenizer. This exercises the highest-risk plumbing — per-timestep tensor
slicing in the FR rollout, chunking in the batched TF path, GT-history windowing,
and metric aggregation — without loading ResNet/DistilBERT.

Run:
    python tests/test_evaluate_logic.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- stub torchvision so `from torchvision import transforms` succeeds at import ---
if "torchvision" not in sys.modules:
    tv = types.ModuleType("torchvision")
    tv_t = types.ModuleType("torchvision.transforms")
    tv.transforms = tv_t
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tv_t

from src.data.labels import IDX_TO_LABEL  # noqa: E402
from src.evaluation.metrics import edit_score, frame_accuracy, macro_f1, segment_iou  # noqa: E402

import importlib.util  # noqa: E402

# import scripts/evaluate.py as a module (it is not a package)
_spec = importlib.util.spec_from_file_location("evaluate_mod", ROOT / "scripts" / "evaluate.py")
evaluate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_mod)
Evaluator = evaluate_mod.Evaluator
aggregate = evaluate_mod.aggregate


class StubModel:
    """Predicts class (frame_id % 8), where frame_id is encoded in img_tokens[:,0,0].

    Ignores history entirely, so FR and TF must yield the same prediction stream —
    which is exactly what lets us assert an exact expected sequence.
    """

    def encode_image(self, imgs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def forward_with_cached_image(self, img_tokens, input_ids, attention_mask):
        frame_ids = img_tokens[:, 0, 0].round().long()
        B = frame_ids.shape[0]
        logits = torch.full((B, 8), -10.0)
        for i in range(B):
            logits[i, int(frame_ids[i]) % 8] = 10.0
        return logits


class StubTokenizer:
    def __call__(self, histories, **kwargs):
        n = len(histories)
        return {
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }


def _make_evaluator(chunk=5):
    return Evaluator(
        model=StubModel(), tokenizer=StubTokenizer(),
        history_length=3, max_text_len=16, device="cpu", use_amp=False, chunk=chunk,
    )


def _img_tokens(n: int):
    """(n, 1, 2) with frame index in channel 0."""
    t = torch.zeros(n, 1, 2)
    t[:, 0, 0] = torch.arange(n, dtype=torch.float32)
    return t


def test_free_running_rollout_indexing():
    ev = _make_evaluator(chunk=5)
    n = 13
    preds = ev._predict_free_running(_img_tokens(n))
    assert preds == [i % 8 for i in range(n)]


def test_batched_known_history_chunking():
    ev = _make_evaluator(chunk=5)  # 13 frames -> chunks of 5,5,3
    n = 13
    preds = ev._predict_known_history(_img_tokens(n), ["[START]"] * n)
    assert preds == [i % 8 for i in range(n)]


def test_tf_history_windowing():
    ev = _make_evaluator()
    # labels 0,1,2,3 -> Prep,Cut,Mix,Cook-Heat ; k=3
    hist = ev.tf_histories([0, 1, 2, 3])
    assert hist[0] == "[START]"
    assert hist[1] == f"[t-1: {IDX_TO_LABEL[0]}]"
    assert hist[2] == f"[t-2: {IDX_TO_LABEL[0]}] [t-1: {IDX_TO_LABEL[1]}]"
    assert hist[3] == f"[t-3: {IDX_TO_LABEL[0]}] [t-2: {IDX_TO_LABEL[1]}] [t-1: {IDX_TO_LABEL[2]}]"


def test_aggregate_matches_direct_metrics():
    v1 = {"video_id": "a", "preds": [0, 0, 1, 1], "labels": [0, 1, 1, 1], "weights": [1.0, 1.0, 0.5, 1.0]}
    v2 = {"video_id": "b", "preds": [2, 2, 2], "labels": [2, 2, 3], "weights": [1.0, 1.0, 1.0]}
    agg = aggregate([v1, v2])

    all_p = v1["preds"] + v2["preds"]
    all_y = v1["labels"] + v2["labels"]
    all_w = v1["weights"] + v2["weights"]
    assert abs(agg["frame_accuracy"] - frame_accuracy(all_p, all_y)) < 1e-12
    assert abs(agg["frame_accuracy_weighted"] - frame_accuracy(all_p, all_y, all_w)) < 1e-12
    assert abs(agg["macro_f1"] - macro_f1(all_p, all_y)) < 1e-12
    # segment metrics are per-video averaged
    exp_iou = (segment_iou(v1["preds"], v1["labels"]) + segment_iou(v2["preds"], v2["labels"])) / 2
    exp_edit = (edit_score(v1["preds"], v1["labels"]) + edit_score(v2["preds"], v2["labels"])) / 2
    assert abs(agg["segment_iou"] - exp_iou) < 1e-12
    assert abs(agg["edit_score"] - exp_edit) < 1e-12


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} evaluate-logic tests passed.")


if __name__ == "__main__":
    _run_all()
