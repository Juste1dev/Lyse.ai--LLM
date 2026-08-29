from __future__ import annotations

import bisect
import glob
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PackedShardsDataset(Dataset):
    def __init__(self, shard_paths: Sequence[str]):
        if not shard_paths:
            raise ValueError("Aucun shard trouvé.")
        self.shard_paths = [str(p) for p in shard_paths]
        self.shards = [np.load(path, mmap_mode="r") for path in self.shard_paths]
        self.lengths = [int(arr.shape[0]) for arr in self.shards]
        self.cumsum = []
        total = 0
        for length in self.lengths:
            total += length
            self.cumsum.append(total)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self.cumsum, idx)
        start = 0 if shard_idx == 0 else self.cumsum[shard_idx - 1]
        local_idx = idx - start
        row = self.shards[shard_idx][local_idx]
        x = torch.tensor(row[:-1], dtype=torch.long)
        y = torch.tensor(row[1:], dtype=torch.long)
        return x, y



def discover_shards(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))



def create_pretrain_dataloader(
    shard_pattern: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    shard_paths = discover_shards(shard_pattern)
    dataset = PackedShardsDataset(shard_paths)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
