"""ResNet-50 image encoder.

Outputs 49 spatial tokens of size `hidden_dim` per image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # -> (B, 2048, 7, 7)
        self.proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 49, hidden_dim)."""
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat.flatten(2).transpose(1, 2)
