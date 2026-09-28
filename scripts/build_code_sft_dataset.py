from __future__ import annotations

# Version "code" de build_sft_dataset.py.
# Chez l'original, chaque message passe dans normalize_text() qui vire
# l'indentation — ça va pour du chat, beaucoup moins pour une réponse qui
# contient une fonction Python. Ici on conserve le formatage.
# Le format de sortie est le même ({"messages": [...]} en jsonl), donc
# train_sft.py marche tel quel, sans modification.

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# réponses plus courtes que ça = probablement du vide, on jette
MIN_RESPONSE_CHARS = 8


def clean_code_text(text: str) -> str:
    """Nettoyage doux : retours chariot, espaces en fin de ligne. Le reste ne bouge pas."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip("\n")


def _pick(row: dict, keys) -> str:
    for key in keys:
        value = row.get(key, "")
        if isinstance(value, str) and value.strip():
            return clean_code_text(value)
    return ""


def iter_source_conversations(source: dict):
    dataset = source["dataset"]
    config = source.get("dataset_config") or None
    split = source.get("split", "train")
    prompt_keys = source.get("prompt_keys", ["prompt", "instruction", "problem", "question"])
    input_keys = source.get("input_keys", ["input", "context"])
    response_keys = source.get("response_keys", ["response", "output", "solution", "answer"])
    system_prompt = clean_code_text(source.get("system_prompt", ""))
    max_conversations = int(source.get("max_conversations", 0) or 0)

    ds = load_dataset(dataset, config, split=split, streaming=True)
    emitted = 0
    for row in ds:
        prompt = _pick(row, prompt_keys)
        extra_input = _pick(row, input_keys)
        response = _pick(row, response_keys)
        if extra_input:
            # style Alpaca : la consigne, puis l'entrée en dessous
            prompt = f"{prompt}\n\n{extra_input}".strip() if prompt else extra_input
        if not prompt or len(response) < MIN_RESPONSE_CHARS:
            continue
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        yield messages
        emitted += 1
        if max_conversations and emitted >= max_conversations:
            break


def serialize_messages(messages) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Construit le dataset SFT code (jsonl messages, formatage conservé)")
    parser.add_argument("--mixture", type=str, required=True, help="json des datasets, ex data/mixtures/code_sft.json")
    parser.add_argument("--train-output", type=str, default="data/sft_code/train.jsonl")
    parser.add_argument("--val-output", type=str, default="data/sft_code/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.mixture, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sources = payload.get("sources", payload)

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = []
    counts: Counter = Counter()

    for idx, source in enumerate(sources):
        if not source.get("enabled", True):
            print(f"[INFO] source désactivée : {source.get('name', f'source_{idx}')}")
            continue
        name = str(source.get("name") or source.get("dataset") or f"source_{idx}")
        try:
            for messages in iter_source_conversations(source):
                key = serialize_messages(messages)
                if key in seen:
                    continue
                seen.add(key)
                kept.append((name, messages))
                counts[name] += 1
                if args.max_total and len(kept) >= args.max_total:
                    break
        except Exception as exc:
            print(f"[WARN] source ignorée : {name} ({exc})")
        if args.max_total and len(kept) >= args.max_total:
            break

    rng.shuffle(kept)

    train_count = 0
    val_count = 0
    split_counts = {"train": Counter(), "val": Counter()}
    with train_output.open("w", encoding="utf-8") as f_train, val_output.open("w", encoding="utf-8") as f_val:
        for source_name, messages in kept:
            target = f_val if rng.random() < args.val_ratio else f_train
            split_name = "val" if target is f_val else "train"
            target.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            split_counts[split_name][source_name] += 1
            if split_name == "train":
                train_count += 1
            else:
                val_count += 1

    manifest = {
        "mixture": str(Path(args.mixture).resolve()),
        "train_output": str(train_output.resolve()),
        "val_output": str(val_output.resolve()),
        "train_count": train_count,
        "val_count": val_count,
        "total_kept": len(kept),
        "unique_conversations": len(seen),
        "counts_by_source": dict(counts),
        "split_counts": {
            "train": dict(split_counts["train"]),
            "val": dict(split_counts["val"]),
        },
    }

    manifest_path = train_output.parent / "manifest_sft_code.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
