"""Trainer for the Phase Recognition model.

Two paths:
  - Teacher-Forcing batch-flatten: when SS probability p == 0, all 64*B (image, history)
    pairs are forwarded in one batched call. Fast.
  - Scheduled Sampling sequential: when p > 0, forward timestep-by-timestep, mixing prior
    ground-truth labels with the model's own (detached) prior predictions in each history.

Histories are token strings — no gradient flows through prior predictions (BPTT disabled).
Soft Boundary weights from the per-frame CSV are applied to the per-position CE losses.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.labels import IDX_TO_LABEL
from .history import build_history_string
from .scheduled_sampling import LinearScheduledSampling


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        ss_schedule: Optional[LinearScheduledSampling] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
        history_length: int = 3,
        max_text_len: int = 64,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.ss_schedule = ss_schedule or LinearScheduledSampling(0.0, 0.0, 0)
        self.lr_scheduler = lr_scheduler
        self.history_length = history_length
        self.max_text_len = max_text_len
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")
        self.grad_clip = grad_clip
        self.scaler = GradScaler(enabled=self.use_amp)

    def _tokenize(self, histories: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            histories,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @staticmethod
    def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits, labels, reduction="none")
        denom = weights.sum().clamp(min=1e-6)
        return (loss * weights).sum() / denom

    def _step_teacher_forcing(self, batch: dict) -> tuple[torch.Tensor, int]:
        """Single batched forward across all (B, W) positions."""
        images = batch["images"].to(self.device)              # (B, W, 3, 224, 224)
        labels = batch["labels"].to(self.device)              # (B, W)
        weights = batch["sample_weights"].to(self.device)     # (B, W)
        valid = batch["valid_mask"].to(self.device).float()   # (B, W)
        B, W = labels.shape

        # Build history strings from ground-truth labels.
        histories: list[str] = []
        labels_cpu = labels.cpu().tolist()
        for b in range(B):
            row = labels_cpu[b]
            for t in range(W):
                prior = [IDX_TO_LABEL[row[i]] for i in range(max(0, t - self.history_length), t)]
                histories.append(build_history_string(prior, self.history_length))

        input_ids, attn_mask = self._tokenize(histories)
        images_flat = images.reshape(B * W, *images.shape[2:])
        labels_flat = labels.reshape(B * W)
        weights_flat = (weights * valid).reshape(B * W)

        with autocast(enabled=self.use_amp):
            logits = self.model(images_flat, input_ids, attn_mask)  # (B*W, num_classes)
            loss = self._weighted_ce(logits, labels_flat, weights_flat)

        return loss, int(valid.sum().item())

    def _step_scheduled_sampling(self, batch: dict, p: float) -> tuple[torch.Tensor, int]:
        """Sequential per-timestep forward with SS mixing.

        Per-position SS sampling: when building history at time t, each prior position
        independently uses the model's prior prediction with probability p, otherwise the
        ground-truth label.
        """
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)
        weights = batch["sample_weights"].to(self.device)
        valid = batch["valid_mask"].to(self.device).float()
        B, W = labels.shape

        labels_cpu = labels.cpu().tolist()
        prior_pred_labels: list[list[str]] = [[] for _ in range(B)]

        total_loss = torch.zeros((), device=self.device)
        n_valid_positions = 0

        for t in range(W):
            histories = []
            for b in range(B):
                k_actual = min(self.history_length, t)
                prior = []
                for i in range(t - k_actual, t):
                    if random.random() < p:
                        prior.append(prior_pred_labels[b][i])
                    else:
                        prior.append(IDX_TO_LABEL[labels_cpu[b][i]])
                histories.append(build_history_string(prior, self.history_length))

            input_ids, attn_mask = self._tokenize(histories)
            images_t = images[:, t]
            labels_t = labels[:, t]
            weights_t = (weights[:, t] * valid[:, t])

            with autocast(enabled=self.use_amp):
                logits_t = self.model(images_t, input_ids, attn_mask)
                loss_t = self._weighted_ce(logits_t, labels_t, weights_t)
            total_loss = total_loss + loss_t

            preds_t = logits_t.detach().argmax(dim=-1).cpu().tolist()
            for b in range(B):
                prior_pred_labels[b].append(IDX_TO_LABEL[preds_t[b]])

            n_valid_positions += int(valid[:, t].sum().item())

        return total_loss / W, n_valid_positions

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        p = self.ss_schedule.p_at(epoch)
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch} (p={p:.2f})")
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            if p <= 0:
                loss, n = self._step_teacher_forcing(batch)
            else:
                loss, n = self._step_scheduled_sampling(batch, p)

            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * n
            running_n += n
            pbar.set_postfix(loss=f"{running_loss / max(running_n, 1):.4f}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return {"loss": running_loss / max(running_n, 1), "ss_p": p}

    @torch.no_grad()
    def evaluate(self, use_teacher_forcing: bool = False) -> dict:
        """Validation pass.

        use_teacher_forcing=False (default): Free-Running — histories built from the
            model's own prior predictions.
        use_teacher_forcing=True: Oracle — histories built from ground-truth labels
            (Teacher-Forcing evaluation, upper-bound reference).
        """
        from ..evaluation.metrics import frame_accuracy, macro_f1

        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_weights: list[float] = []

        for batch in tqdm(self.val_loader, desc="eval"):
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)
            weights = batch["sample_weights"].to(self.device)
            valid = batch["valid_mask"].to(self.device).float()
            B, W = labels.shape
            labels_cpu = labels.cpu().tolist()
            prior_pred_labels: list[list[str]] = [[] for _ in range(B)]

            for t in range(W):
                histories = []
                for b in range(B):
                    k_actual = min(self.history_length, t)
                    if use_teacher_forcing:
                        prior = [IDX_TO_LABEL[labels_cpu[b][i]] for i in range(t - k_actual, t)]
                    else:
                        prior = prior_pred_labels[b][-k_actual:] if k_actual > 0 else []
                    histories.append(build_history_string(prior, self.history_length))
                input_ids, attn_mask = self._tokenize(histories)
                with autocast(enabled=self.use_amp):
                    logits_t = self.model(images[:, t], input_ids, attn_mask)
                preds_t = logits_t.argmax(dim=-1).cpu().tolist()
                labels_t = labels[:, t].cpu().tolist()
                weights_t = (weights[:, t] * valid[:, t]).cpu().tolist()
                for b in range(B):
                    if weights_t[b] > 0:
                        all_preds.append(preds_t[b])
                        all_labels.append(labels_t[b])
                        all_weights.append(weights_t[b])
                    prior_pred_labels[b].append(IDX_TO_LABEL[preds_t[b]])

        return {
            "frame_accuracy": frame_accuracy(all_preds, all_labels, all_weights),
            "macro_f1": macro_f1(all_preds, all_labels),
            "n_positions": len(all_preds),
        }
