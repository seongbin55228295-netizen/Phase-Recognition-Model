"""Model loading + whole-video Free-Running autoregressive inference.

The Free-Running loop here mirrors Trainer.evaluate (src/training/trainer.py):
histories are built solely from the model's own prior predictions, never from
ground-truth labels. The only difference is scope — instead of resetting history
every 64-frame window, inference treats the entire video as one continuous
sequence (the history is just a length-k string, so there is no memory limit).

Works for both PhaseRecognitionModel and ImageOnlyModel: both expose
`encode_image` and `forward_with_cached_image`, and the image-only model simply
ignores the tokenized history.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ..data.labels import IDX_TO_LABEL
from ..models import ImageOnlyModel, PhaseRecognitionModel, build_tokenizer
from ..training.history import build_history_string


def build_model_from_config(model_cfg: dict) -> torch.nn.Module:
    """Construct a model from a YAML `model:` block (mirrors scripts/train.py)."""
    model_type = model_cfg.get("type", "phase_recognition")
    if model_type == "phase_recognition":
        return PhaseRecognitionModel(
            num_classes=model_cfg["num_classes"],
            hidden_dim=model_cfg["hidden_dim"],
            fusion=model_cfg["fusion"],
            text_model_name=model_cfg["text_model_name"],
            image_pretrained=model_cfg["image_pretrained"],
            fusion_kwargs=model_cfg.get("fusion_kwargs"),
            classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        )
    if model_type == "image_only":
        return ImageOnlyModel(
            num_classes=model_cfg["num_classes"],
            hidden_dim=model_cfg["hidden_dim"],
            image_pretrained=model_cfg["image_pretrained"],
            classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        )
    raise ValueError(f"Unknown model.type: {model_type!r}")


class Predictor:
    """Holds a loaded model + tokenizer and runs whole-video Free-Running inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        *,
        history_length: int = 3,
        max_text_len: int = 64,
        device: str = "cpu",
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.history_length = history_length
        self.max_text_len = max_text_len
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        use_amp: bool = True,
    ) -> "Predictor":
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg["model"]
        model = build_model_from_config(model_cfg)

        state = torch.load(checkpoint_path, map_location=device)
        # best.pt is a bare state_dict; tolerate a wrapped {"state_dict": ...} too.
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        model.load_state_dict(state)

        tokenizer = build_tokenizer(model_cfg.get("text_model_name", "distilbert-base-uncased"))
        train_cfg = cfg.get("training", {})
        return cls(
            model,
            tokenizer,
            history_length=train_cfg.get("history_length", 3),
            max_text_len=train_cfg.get("max_text_len", 64),
            device=device,
            use_amp=use_amp,
        )

    def _autocast(self):
        """AMP context on CUDA only; a no-op elsewhere (never references CUDA on CPU)."""
        if self.use_amp:
            return torch.autocast(device_type="cuda", enabled=True)
        return contextlib.nullcontext()

    def _tokenize(self, histories: list[str]):
        enc = self.tokenizer(
            histories,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor, batch_size: int) -> torch.Tensor:
        """(N, 3, 224, 224) -> (N, T, D) image tokens, kept on CPU to bound VRAM."""
        token_chunks: list[torch.Tensor] = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i:i + batch_size].to(self.device)
            with self._autocast():
                tokens = self.model.encode_image(chunk)
            token_chunks.append(tokens.float().cpu())
        return torch.cat(token_chunks, dim=0)

    @torch.no_grad()
    def run(self, images: torch.Tensor, *, image_batch_size: int = 64) -> dict:
        """Free-Running inference over the whole sequence.

        Args:
            images: (N, 3, 224, 224) eval-transformed frames in temporal order.
        Returns:
            {"pred_indices": list[int], "pred_labels": list[str],
             "pred_probs": list[float],          # softmax prob of the chosen class
             "probs": Tensor (N, num_classes)}   # full distribution per frame
        """
        self.model.eval()
        n = images.shape[0]
        img_tokens_all = self._encode_images(images, image_batch_size)  # (N, T, D) on CPU

        prior_pred_labels: list[str] = []
        pred_indices: list[int] = []
        pred_probs: list[float] = []
        prob_rows: list[torch.Tensor] = []

        for t in range(n):
            k = min(self.history_length, t)
            prior = prior_pred_labels[-k:] if k > 0 else []
            history = build_history_string(prior, self.history_length)
            input_ids, attn = self._tokenize([history])

            img_t = img_tokens_all[t:t + 1].to(self.device)
            with self._autocast():
                logits = self.model.forward_with_cached_image(img_t, input_ids, attn)
            probs = F.softmax(logits.float(), dim=-1)[0]            # (num_classes,)
            pred = int(torch.argmax(probs).item())

            pred_indices.append(pred)
            pred_probs.append(float(probs[pred].item()))
            prob_rows.append(probs.cpu())
            prior_pred_labels.append(IDX_TO_LABEL[pred])

        return {
            "pred_indices": pred_indices,
            "pred_labels": [IDX_TO_LABEL[i] for i in pred_indices],
            "pred_probs": pred_probs,
            "probs": torch.stack(prob_rows, dim=0),
        }
