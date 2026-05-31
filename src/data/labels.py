"""8 phase classes, loaded from configs/action_class_prototypes.json (single source of truth)."""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROTOTYPES_PATH = _REPO_ROOT / "configs" / "action_class_prototypes.json"


def _load_class_names() -> list[str]:
    with open(_PROTOTYPES_PATH, encoding="utf-8") as f:
        return list(json.load(f)["labels"])


CLASS_NAMES: list[str] = _load_class_names()
NUM_CLASSES: int = len(CLASS_NAMES)
LABEL_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
