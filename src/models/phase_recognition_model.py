"""End-to-end model: ImageEncoder + TextEncoder -> Fusion -> Classifier."""
from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import Classifier
from .encoders import ImageEncoder, TextEncoder
from .fusion import build_fusion


class PhaseRecognitionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        hidden_dim: int = 512,
        fusion: str = "co_attention",
        text_model_name: str = "distilbert-base-uncased",
        image_pretrained: bool = True,
        fusion_kwargs: dict | None = None,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim, pretrained=image_pretrained)
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim, model_name=text_model_name)
        self.fusion = build_fusion(fusion, dim=hidden_dim, **(fusion_kwargs or {}))
        self.classifier = Classifier(in_dim=hidden_dim, num_classes=num_classes,
                                      dropout=classifier_dropout)

    def forward(
        self,
        image: torch.Tensor,             # (B, 3, 224, 224)
        input_ids: torch.Tensor,         # (B, L)
        attention_mask: torch.Tensor,    # (B, L)
    ) -> torch.Tensor:
        img_tokens = self.image_encoder(image)
        text_tokens = self.text_encoder(input_ids, attention_mask)
        z = self.fusion(img_tokens, text_tokens, attention_mask)
        return self.classifier(z)
