from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


@dataclass
class TokenizerInfo:
    model_path: str
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else self.sp.piece_to_id("<bos>")
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else self.sp.piece_to_id("<eos>")
        self.unk_id = self.sp.unk_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        clean = [int(i) for i in ids if int(i) >= 0]
        return self.sp.decode(clean)

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    def info(self) -> TokenizerInfo:
        return TokenizerInfo(
            model_path=self.model_path,
            vocab_size=self.vocab_size,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            unk_id=self.unk_id,
        )
