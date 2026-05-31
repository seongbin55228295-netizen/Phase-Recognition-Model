"""Training entrypoint.

Usage:
    python scripts/train.py --config experiments/baseline.yaml

------------------------------------------------------------------------------
Design decisions (single source of truth: the YAML config; this docstring only
documents the *reasoning* behind the defaults in experiments/baseline.yaml).
------------------------------------------------------------------------------
Sample unit
    Fixed 64-frame window from one video's emitted-frame CSV (data/frames/<vid>/).
    Pros: deterministic memory footprint, easy history-length comparison.
    Videos with < 64 emitted frames (3 of 300) are tail-padded with zero tensors
    and valid_mask=False so those positions contribute 0 to the loss.

Autoregressive history
    Format: "[t-k: Label] [t-(k-1): Label] ... [t-1: Label]"  (oldest first).
    At t=0 the special token "[START]" is used; partial-prefix windows emit
    only as many "[t-i: Label]" tokens as exist.
    Default history length k = 3 (configurable).
    History is built from label *strings* — no tensor flows from prior preds
    into the current forward graph, so BPTT is naturally disabled.

Window forward path
    Teacher-Forcing (p == 0): all (B * 64) (image, history) pairs are batched
    into one forward call. Fast, high GPU utilization.
    Scheduled Sampling (p > 0): per-timestep sequential forward. Each prior
    position independently uses the model's detached argmax with probability p,
    otherwise the ground-truth label. Slower but required because prior preds
    depend on prior forwards.

Scheduled Sampling schedule
    Linear ramp p: 0.0 -> 0.5 across the first 10 epochs, held at 0.5 afterwards.

Optimizer
    AdamW with two parameter groups (Differential LR):
      - encoder_lr = 1e-5  (ResNet-50 backbone + DistilBERT body)
      - head_lr    = 1e-4  (image_encoder.proj, text_encoder.proj, fusion, classifier)
    weight_decay = 1e-4, grad_clip = 1.0, mixed-precision (AMP) on CUDA.

Loss
    Cross-entropy weighted by `sample_weight * valid_mask` per position.
    Soft Boundary (segment edges ±2 frames at weight 0.5) is already encoded in
    sample_weight by preprocessing; the trainer just multiplies it in.

Evaluation
    Validation runs in Free-Running mode: histories built solely from the
    model's own prior predictions, never from ground-truth labels. Reported
    metrics: frame_accuracy (sample_weight-weighted) + macro_f1.

Target hardware
    Single GPU, VRAM >= 16 GB. With AMP on, batch_size=8 + window=64 fits
    comfortably; lower batch_size if running on 8-12 GB.
------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders
from src.models import ImageOnlyModel, PhaseRecognitionModel, build_tokenizer
from src.training import LinearScheduledSampling, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to experiment YAML config")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: PhaseRecognitionModel, cfg: dict) -> torch.optim.Optimizer:
    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("image_encoder.backbone") or name.startswith("text_encoder.bert"):
            encoder_params.append(p)
        else:
            head_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(cfg["encoder_lr"])},
            {"params": head_params, "lr": float(cfg["head_lr"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg = cfg["data"]
    train_loader, val_loader = build_dataloaders(
        manifest_path=ROOT / data_cfg["manifest_path"],
        frames_root=ROOT / data_cfg["frames_root"],
        labels_root=ROOT / data_cfg["labels_root"],
        window_size=data_cfg["window_size"],
        batch_size=data_cfg["batch_size"],
        num_windows_per_video=data_cfg["num_windows_per_video"],
        num_workers=data_cfg["num_workers"],
    )

    model_cfg = cfg["model"]
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
        tokenizer = build_tokenizer(model_cfg["text_model_name"])
    elif model_type == "image_only":
        model = ImageOnlyModel(
            num_classes=model_cfg["num_classes"],
            hidden_dim=model_cfg["hidden_dim"],
            image_pretrained=model_cfg["image_pretrained"],
            classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        )
        # Still need a tokenizer for the Trainer interface, but tokens get ignored.
        tokenizer = build_tokenizer(model_cfg.get("text_model_name", "distilbert-base-uncased"))
    else:
        raise ValueError(f"Unknown model.type: {model_type!r}")

    train_cfg = cfg["training"]
    optimizer = build_optimizer(model, train_cfg["optimizer"])
    ss_cfg = train_cfg["scheduled_sampling"]
    ss = LinearScheduledSampling(
        p_start=ss_cfg["p_start"], p_end=ss_cfg["p_end"], ramp_epochs=ss_cfg["ramp_epochs"]
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        ss_schedule=ss,
        history_length=train_cfg["history_length"],
        max_text_len=train_cfg["max_text_len"],
        device=device,
        use_amp=train_cfg["use_amp"],
        grad_clip=train_cfg["grad_clip"],
    )

    ckpt_dir = ROOT / cfg["checkpoint"]["dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_mode = cfg.get("eval_mode", "fr")  # 'fr' (Free-Running) or 'tf' (Teacher-Forcing oracle)
    if eval_mode not in ("fr", "tf"):
        raise ValueError(f"eval_mode must be 'fr' or 'tf', got {eval_mode!r}")

    history = []
    best_acc = -1.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.evaluate(use_teacher_forcing=(eval_mode == "tf"))
        record = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(json.dumps(record))

        if epoch % cfg["checkpoint"]["save_every"] == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["frame_accuracy"] > best_acc:
            best_acc = val_metrics["frame_accuracy"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
