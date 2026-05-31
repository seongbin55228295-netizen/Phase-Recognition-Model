"""MLP classification head: 512 -> 256 -> 8."""
from __future__ import annotations

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
