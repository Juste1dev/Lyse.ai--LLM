from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyse_ai.dataset_mixtures import iter_named_conversation_sources, normalize_messages, is_valid_sft_conversation



def serialize_messages(messages):
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)



def main():
    parser = argparse.ArgumentParser(description="Construit un dataset SFT multi-sources au format messages JSONL")
    parser.add_argument("--sources-config", type=str, required=True)
    parser.add_argument("--train-output", type=str, default="data/sft/train.jsonl")
    parser.add_argument("--val-output", type=str, default="data/sft/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.sources_config, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sources = payload.get("sources", payload)

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = []
    counts = Counter()

    for source_name, messages in iter_named_conversation_sources(sources):
        messages = normalize_messages(messages)
        if not is_valid_sft_conversation(messages):
            continue
        key = serialize_messages(messages)
        if key in seen:
            continue
        seen.add(key)
        kept.append((source_name, messages))
        counts[source_name] += 1
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
        "sources_config": str(Path(args.sources_config).resolve()),
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

    manifest_path = train_output.parent / "manifest_sft.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
