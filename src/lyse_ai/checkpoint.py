from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .utils import ensure_dir, latest_checkpoint_path



def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    payload: Dict[str, Any],
) -> Path:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    path = checkpoint_dir / f"step-{step:08d}.pt"
    torch.save(payload, path)
    return path



def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)



def load_latest_checkpoint(checkpoint_dir: str | Path, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    latest = latest_checkpoint_path(checkpoint_dir)
    if latest is None:
        return None
    return load_checkpoint(latest, map_location=map_location)
