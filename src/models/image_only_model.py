"""Image-only baseline (modality ablation variant).

No text encoder, no fusion. ResNet-50 features are mean-pooled across the 49
spatial tokens and fed straight into the MLP head.

Keeps the same forward signature as PhaseRecognitionModel so the Trainer can
call it interchangeably (text inputs are accepted and ignored).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import Classifier
from .encoders import ImageEncoder


class ImageOnlyModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        hidden_dim: int = 512,
        image_pretrained: bool = True,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim, pretrained=image_pretrained)
        self.classifier = Classifier(
            in_dim=hidden_dim, num_classes=num_classes, dropout=classifier_dropout
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def forward_with_cached_image(
        self,
        img_tokens: torch.Tensor,
        input_ids: torch.Tensor | None = None,       # ignored
        attention_mask: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        pooled = img_tokens.mean(dim=1)
        return self.classifier(pooled)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor | None = None,       # ignored
        attention_mask: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        img_tokens = self.encode_image(image)
        return self.forward_with_cached_image(img_tokens, input_ids, attention_mask)
