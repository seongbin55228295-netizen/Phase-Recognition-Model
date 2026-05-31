"""DistilBERT text encoder for the autoregressive history string.

The trainer tokenizes history strings each step; this module only consumes
input_ids / attention_mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel


def build_tokenizer(model_name: str = "distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)


class TextEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        model_name: str = "distilbert-base-uncased",
    ) -> None:
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask: (B, L) -> (B, L, hidden_dim)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state)
