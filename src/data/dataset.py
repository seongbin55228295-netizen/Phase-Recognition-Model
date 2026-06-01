"""Frame-window Dataset for the Phase Recognition pipeline.

Each __getitem__ returns one fixed-size window of consecutive emitted frames
from a single video, with per-frame labels, sample weights, and a valid mask.

The trainer is responsible for:
  - building the autoregressive history string at each timestep within a window
  - mixing teacher-forcing vs. model predictions (Scheduled Sampling)
  - skipping loss at positions where valid_mask is False (padding tail)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .labels import LABEL_TO_IDX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PhaseRecognitionDataset(Dataset):
    """Window-based dataset.

    Args:
        manifest_path: processed/frame_labels/_manifest.json
        frames_root:   data/frames/ (each video -> <vid>/frame_NNNNNN.jpg)
        labels_root:   processed/frame_labels/ (each video -> <vid>.csv)
        subset:        'training' or 'validation' (read from manifest)
        window_size:   number of consecutive emitted frames per sample
        num_windows_per_video: epoch-length multiplier (1 window/video by default)
        transform:     torchvision transform; defaults to eval transform
    """

    def __init__(
        self,
        manifest_path: str | Path,
        frames_root: str | Path,
        labels_root: str | Path,
        subset: str,
        window_size: int = 64,
        transform: Optional[transforms.Compose] = None,
        num_windows_per_video: int = 1,
    ) -> None:
        if subset not in ("training", "validation"):
            raise ValueError(f"subset must be 'training' or 'validation', got {subset!r}")

        self.frames_root = Path(frames_root)
        self.labels_root = Path(labels_root)
        self.window_size = window_size
        self.transform = transform or build_eval_transform()
        self.num_windows_per_video = num_windows_per_video

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        self.video_ids: list[str] = [
            vid for vid, meta in manifest["videos"].items() if meta["subset"] == subset
        ]

        # CSVs are tiny in aggregate (~90k rows / 300 files) - eager-load once.
        self._rows: dict[str, pd.DataFrame] = {
            vid: pd.read_csv(self.labels_root / f"{vid}.csv") for vid in self.video_ids
        }

    def __len__(self) -> int:
        return len(self.video_ids) * self.num_windows_per_video

    def __getitem__(self, idx: int) -> dict:
        vid = self.video_ids[idx % len(self.video_ids)]
        df = self._rows[vid]
        n_rows = len(df)
        W = self.window_size

        if n_rows >= W:
            start = random.randint(0, n_rows - W)
            window = df.iloc[start:start + W]
            valid_len = W
        else:
            window = df
            valid_len = n_rows

        images: list[torch.Tensor] = []
        labels = torch.zeros(W, dtype=torch.long)
        weights = torch.zeros(W, dtype=torch.float32)
        is_boundary = torch.zeros(W, dtype=torch.bool)
        valid_mask = torch.zeros(W, dtype=torch.bool)

        for i, (_, row) in enumerate(window.iterrows()):
            img_path = self.frames_root / vid / row["frame_name"]
            with Image.open(img_path) as im:
                images.append(self.transform(im.convert("RGB")))
            labels[i] = LABEL_TO_IDX[row["label"]]
            weights[i] = float(row["sample_weight"])
            is_boundary[i] = bool(row["is_boundary"])
            valid_mask[i] = True

        for _ in range(valid_len, W):
            images.append(torch.zeros(3, 224, 224))

        return {
            "video_id": vid,
            "images": torch.stack(images, dim=0),       # (W, 3, 224, 224)
            "labels": labels,                            # (W,)
            "sample_weights": weights,                   # (W,)
            "is_boundary": is_boundary,                  # (W,)
            "valid_mask": valid_mask,                    # (W,)
        }


def build_dataloaders(
    manifest_path: str | Path,
    frames_root: str | Path,
    labels_root: str | Path,
    window_size: int = 64,
    batch_size: int = 8,
    num_windows_per_video: int = 1,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = PhaseRecognitionDataset(
        manifest_path, frames_root, labels_root,
        subset="training",
        window_size=window_size,
        transform=build_train_transform(),
        num_windows_per_video=num_windows_per_video,
    )
    val_ds = PhaseRecognitionDataset(
        manifest_path, frames_root, labels_root,
        subset="validation",
        window_size=window_size,
        transform=build_eval_transform(),
        num_windows_per_video=1,
    )
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
