"""Co-attention fusion: 2-layer bidirectional cross-attention between img and text tokens.

Output: pooled 512d fused vector per sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CoAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.img_ln1 = nn.LayerNorm(dim)
        self.img_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )
        self.img_ln2 = nn.LayerNorm(dim)

        self.text_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_ln1 = nn.LayerNorm(dim)
        self.text_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )
        self.text_ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        img: torch.Tensor,
        text: torch.Tensor,
        text_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_attn, _ = self.img_attn(
            query=img, key=text, value=text,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        img = self.img_ln1(img + img_attn)
        img = self.img_ln2(img + self.img_ffn(img))

        text_attn, _ = self.text_attn(
            query=text, key=img, value=img,
            need_weights=False,
        )
        text = self.text_ln1(text + text_attn)
        text = self.text_ln2(text + self.text_ffn(text))

        return img, text


class CoAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CoAttentionBlock(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.proj = nn.Linear(dim * 2, dim)

    def forward(
        self,
        img_tokens: torch.Tensor,           # (B, 49, dim)
        text_tokens: torch.Tensor,          # (B, L, dim)
        text_attention_mask: torch.Tensor | None = None,  # (B, L), 1=real 0=pad
    ) -> torch.Tensor:
        # MultiheadAttention expects True at padded positions.
        text_kpm = (text_attention_mask == 0) if text_attention_mask is not None else None

        for layer in self.layers:
            img_tokens, text_tokens = layer(img_tokens, text_tokens, text_kpm)

        img_pooled = img_tokens.mean(dim=1)
        if text_attention_mask is not None:
            mask = text_attention_mask.unsqueeze(-1).float()
            text_pooled = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            text_pooled = text_tokens.mean(dim=1)

        return self.proj(torch.cat([img_pooled, text_pooled], dim=-1))
