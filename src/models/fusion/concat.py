"""Concat fusion: pool each modality, concatenate, then MLP -> 512d.

Comparison baseline against Co-attention (simplest plausible fusion that still
uses both modalities).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        **_unused,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        img_tokens: torch.Tensor,           # (B, 49, dim)
        text_tokens: torch.Tensor,          # (B, L, dim)
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img_pooled = img_tokens.mean(dim=1)
        if text_attention_mask is not None:
            mask = text_attention_mask.unsqueeze(-1).float()
            text_pooled = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            text_pooled = text_tokens.mean(dim=1)
        return self.net(torch.cat([img_pooled, text_pooled], dim=-1))
