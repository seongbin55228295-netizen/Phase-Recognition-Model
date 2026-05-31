"""Autoregressive history string builder.

Format: "[t-k: Label] [t-(k-1): Label] ... [t-1: Label]"  (oldest first)
At t=0 (no prior labels) -> "[START]".
For partial history (t < k), only as many tokens as available are emitted.
"""
from __future__ import annotations


START_TOKEN = "[START]"


def build_history_string(prior_labels: list[str], history_length: int) -> str:
    if not prior_labels:
        return START_TOKEN
    actual = prior_labels[-history_length:]
    n = len(actual)
    return " ".join(f"[t-{n - i}: {label}]" for i, label in enumerate(actual))
