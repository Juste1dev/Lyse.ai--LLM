from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        return self.total / max(1, self.count)



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def choose_autocast_dtype(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32



def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()



def resolve_precision(requested: str) -> str:
    requested = requested.lower()
    if requested == "bf16" and not supports_bf16():
        return "fp16"
    return requested



def latest_checkpoint_path(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("step-*.pt"))
    return candidates[-1] if candidates else None



def save_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")



def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"



def count_tokens(batch: torch.Tensor) -> int:
    return int(batch.numel())



def now_ts() -> float:
    return time.time()
