"""Deterministic full-video evaluation entrypoint.

Rebuilds a trained variant from its YAML config, loads ``best.pt``, and runs a
deterministic autoregressive pass over the *entire* emitted-frame sequence of
every video in the chosen split (default: validation, 84 videos). Unlike the
in-training validation (one random 64-frame window per video), this covers each
video end to end so that segment-level metrics are meaningful and reproducible.

Each checkpoint is scored under BOTH inference regimes with the same weights:
  - FR (Free-Running): history built from the model's own prior predictions.
  - TF (Teacher-Forcing/oracle): history built from ground-truth labels.
The Exposure-Bias signature is ``tf_fr_gap = frame_acc_TF - frame_acc_FR``.

TF (and the image-only model, which ignores text) needs no sequential rollout —
all histories are known up front, so those passes are batched. FR for the
fusion model must roll out one timestep at a time.

Usage:
    python scripts/evaluate.py --config experiments/baseline.yaml
    python scripts/evaluate.py --config experiments/baseline.yaml --limit 3   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.labels import IDX_TO_LABEL, LABEL_TO_IDX
from src.evaluation.metrics import edit_score, frame_accuracy, macro_f1, segment_iou
from src.training.history import build_history_string

# NOTE: torchvision/transformers-backed imports (the model + image transform) are
# deferred into build_model()/main() so this module — and its rollout/aggregation
# logic — can be imported and unit-tested in environments without torchvision.

try:  # torch>=2.4 exposes the generic device-aware autocast
    from torch import autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to experiment YAML config")
    p.add_argument(
        "--split",
        default="validation",
        choices=["validation", "training", "test"],
        help="Manifest subset to evaluate. 'test' is an alias for 'validation' "
        "(the dataset has no separate test split).",
    )
    p.add_argument("--output-dir", default="reports/metrics", help="Where to write <variant>.json")
    p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N videos (0 = all). Smoke testing.")
    p.add_argument("--chunk", type=int, default=128, help="Frames per forward chunk (image encode + batched TF/image-only).")
    return p.parse_args()


def build_model(model_cfg: dict):
    from src.models import ImageOnlyModel, PhaseRecognitionModel  # heavy (torchvision)

    model_type = model_cfg.get("type", "phase_recognition")
    if model_type == "phase_recognition":
        model = PhaseRecognitionModel(
            num_classes=model_cfg["num_classes"],
            hidden_dim=model_cfg["hidden_dim"],
            fusion=model_cfg["fusion"],
            text_model_name=model_cfg["text_model_name"],
            image_pretrained=model_cfg["image_pretrained"],
            fusion_kwargs=model_cfg.get("fusion_kwargs"),
            classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        )
        text_aware = True
    elif model_type == "image_only":
        model = ImageOnlyModel(
            num_classes=model_cfg["num_classes"],
            hidden_dim=model_cfg["hidden_dim"],
            image_pretrained=model_cfg["image_pretrained"],
            classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        )
        text_aware = False
    else:
        raise ValueError(f"Unknown model.type: {model_type!r}")
    return model, text_aware


class Evaluator:
    def __init__(self, model, tokenizer, history_length, max_text_len, device, use_amp, chunk):
        self.model = model
        self.tokenizer = tokenizer
        self.k = history_length
        self.max_text_len = max_text_len
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")
        self.chunk = chunk
        self.amp_device = "cuda" if device.startswith("cuda") else "cpu"
        self.amp_dtype = torch.float16

    def _tokenize(self, histories: list[str]):
        enc = self.tokenizer(
            histories, padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def _encode_images(self, frame_paths: list[Path], transform) -> torch.Tensor:
        """Load + encode every frame of a video into image tokens (N, T, D)."""
        tokens: list[torch.Tensor] = []
        for s in range(0, len(frame_paths), self.chunk):
            batch_imgs = []
            for fp in frame_paths[s:s + self.chunk]:
                with Image.open(fp) as im:
                    batch_imgs.append(transform(im.convert("RGB")))
            imgs = torch.stack(batch_imgs, dim=0).to(self.device)
            with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
                tok = self.model.encode_image(imgs)
            tokens.append(tok.float())
        return torch.cat(tokens, dim=0)  # (N, T, D)

    @torch.no_grad()
    def _predict_known_history(self, img_tokens: torch.Tensor, histories: list[str]) -> list[int]:
        """Batched forward when all histories are known up front (TF / image-only)."""
        N = img_tokens.shape[0]
        preds: list[int] = []
        for s in range(0, N, self.chunk):
            chunk_hist = histories[s:s + self.chunk]
            input_ids, attn = self._tokenize(chunk_hist)
            with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model.forward_with_cached_image(
                    img_tokens[s:s + self.chunk], input_ids, attn
                )
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
        return preds

    @torch.no_grad()
    def _predict_free_running(self, img_tokens: torch.Tensor) -> list[int]:
        """Sequential rollout: history fed from the model's own prior predictions."""
        N = img_tokens.shape[0]
        prior: list[str] = []
        preds: list[int] = []
        for t in range(N):
            hist = build_history_string(prior[-self.k:] if self.k > 0 else [], self.k)
            input_ids, attn = self._tokenize([hist])
            with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model.forward_with_cached_image(
                    img_tokens[t:t + 1], input_ids, attn
                )
            pred = int(logits.argmax(dim=-1).item())
            preds.append(pred)
            prior.append(IDX_TO_LABEL[pred])
        return preds

    def tf_histories(self, gt_labels: list[int]) -> list[str]:
        hist: list[str] = []
        for t in range(len(gt_labels)):
            prior = [IDX_TO_LABEL[gt_labels[i]] for i in range(max(0, t - self.k), t)]
            hist.append(build_history_string(prior, self.k))
        return hist


def aggregate(per_video: list[dict]) -> dict:
    """Frame metrics over all frames pooled; segment metrics averaged per video."""
    all_preds = [p for v in per_video for p in v["preds"]]
    all_labels = [y for v in per_video for y in v["labels"]]
    all_weights = [w for v in per_video for w in v["weights"]]
    ious = [segment_iou(v["preds"], v["labels"]) for v in per_video]
    edits = [edit_score(v["preds"], v["labels"]) for v in per_video]
    n = max(len(per_video), 1)
    return {
        "frame_accuracy": frame_accuracy(all_preds, all_labels),
        "frame_accuracy_weighted": frame_accuracy(all_preds, all_labels, all_weights),
        "macro_f1": macro_f1(all_preds, all_labels),
        "segment_iou": sum(ious) / n,
        "edit_score": sum(edits) / n,
    }


def main() -> None:
    import yaml

    from src.data.dataset import build_eval_transform  # heavy (torchvision)
    from src.models import build_tokenizer

    for stream in (sys.stdout, sys.stderr):  # Windows cp949 console safety
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    variant = cfg.get("name", Path(args.config).stem)
    subset = "validation" if args.split == "test" else args.split
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    manifest_path = ROOT / data_cfg["manifest_path"]
    frames_root = ROOT / data_cfg["frames_root"]
    labels_root = ROOT / data_cfg["labels_root"]

    model, text_aware = build_model(cfg["model"])
    ckpt_path = ROOT / cfg["checkpoint"]["dir"] / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    tokenizer = build_tokenizer(cfg["model"].get("text_model_name", "distilbert-base-uncased"))
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        history_length=train_cfg["history_length"],
        max_text_len=train_cfg["max_text_len"],
        device=device,
        use_amp=train_cfg.get("use_amp", True),
        chunk=args.chunk,
    )
    transform = build_eval_transform()

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    video_ids = [vid for vid, m in manifest["videos"].items() if m["subset"] == subset]
    video_ids.sort()
    if args.limit > 0:
        video_ids = video_ids[: args.limit]

    print(f"[{variant}] device={device} split={subset} videos={len(video_ids)} "
          f"k={evaluator.k} text_aware={text_aware}")

    fr_videos: list[dict] = []
    tf_videos: list[dict] = []
    total_frames = 0

    for i, vid in enumerate(video_ids, 1):
        df = pd.read_csv(labels_root / f"{vid}.csv")
        labels = [LABEL_TO_IDX[x] for x in df["label"].tolist()]
        weights = [float(w) for w in df["sample_weight"].tolist()]
        frame_paths = [frames_root / vid / name for name in df["frame_name"].tolist()]
        total_frames += len(labels)

        img_tokens = evaluator._encode_images(frame_paths, transform)

        tf_hist = evaluator.tf_histories(labels)
        tf_preds = evaluator._predict_known_history(img_tokens, tf_hist)
        if text_aware:
            fr_preds = evaluator._predict_free_running(img_tokens)
        else:
            fr_preds = tf_preds  # image-only ignores history -> FR == TF

        fr_videos.append({"video_id": vid, "preds": fr_preds, "labels": labels, "weights": weights})
        tf_videos.append({"video_id": vid, "preds": tf_preds, "labels": labels, "weights": weights})
        print(f"  [{i}/{len(video_ids)}] {vid}  frames={len(labels)}")

    fr = aggregate(fr_videos)
    tf = aggregate(tf_videos)
    result = {
        "variant": variant,
        "config": str(Path(args.config).as_posix()),
        "checkpoint": str(ckpt_path.relative_to(ROOT).as_posix()),
        "split": subset,
        "n_videos": len(video_ids),
        "n_frames": total_frames,
        "history_length": evaluator.k,
        "text_aware": text_aware,
        "fr": fr,
        "tf": tf,
        "tf_fr_gap": tf["frame_accuracy"] - fr["frame_accuracy"],
    }

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k in ("fr", "tf", "tf_fr_gap")}, indent=2))
    print(f"-> wrote {out_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
