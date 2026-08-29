from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .model import LyseTransformer
from .tokenizer import SentencePieceTokenizer


@torch.no_grad()
def generate_text(
    model: LyseTransformer,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    device: str = "cuda",
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.cfg.max_seq_len :]
        out = model(x_cond)
        logits = out["logits"][:, -1, :]

        if repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                logits[:, token_id] /= repetition_penalty

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = probs.cumsum(dim=-1)
                sorted_mask = cumulative > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                sorted_logits[sorted_mask] = -float("inf")
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id], dim=1)
        if int(next_id.item()) == tokenizer.eos_id:
            break

    text = tokenizer.decode(x[0].tolist())
    return text



def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str = "cuda",
):
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig(**payload["model_config"])
    model = LyseTransformer(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, payload
