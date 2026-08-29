from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import SentencePieceTokenizer


ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


class ConversationSFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: SentencePieceTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        if not self.items:
            raise ValueError(f"Fichier SFT vide: {path}")

    def __len__(self) -> int:
        return len(self.items)

    def _encode_messages(self, messages: Sequence[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        ids = [self.tokenizer.bos_id]
        labels = [-100]
        for msg in messages:
            role = msg.get("role", "user").strip().lower()
            content = (msg.get("content") or "").strip()
            prefix = ROLE_TOKENS.get(role, "<|user|>") + "\n"
            prefix_ids = self.tokenizer.encode(prefix, add_bos=False, add_eos=False)
            content_ids = self.tokenizer.encode(content + "\n", add_bos=False, add_eos=False)
            all_ids = prefix_ids + content_ids
            ids.extend(all_ids)
            if role == "assistant":
                labels.extend(all_ids)
            else:
                labels.extend([-100] * len(all_ids))
        ids.append(self.tokenizer.eos_id)
        labels.append(self.tokenizer.eos_id)
        ids = ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]
        return ids, labels

    def __getitem__(self, idx: int):
        row = self.items[idx]
        if "messages" in row:
            ids, labels = self._encode_messages(row["messages"])
        else:
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            ids, labels = self._encode_messages(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            )
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)



def sft_collate_fn(batch, pad_id: int):
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, lbls) in enumerate(batch):
        input_ids[i, : len(ids)] = ids
        labels[i, : len(lbls)] = lbls
    return input_ids, labels



def create_sft_dataloader(
    path: str,
    tokenizer: SentencePieceTokenizer,
    batch_size: int,
    shuffle: bool,
    max_seq_len: int,
    num_workers: int = 0,
):
    dataset = ConversationSFTDataset(path, tokenizer=tokenizer, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer.pad_id),
        drop_last=True,
    )
