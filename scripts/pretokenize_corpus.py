from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyse_ai.dataset_mixtures import iter_named_text_sources
from lyse_ai.tokenizer import SentencePieceTokenizer



def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())



def iter_texts(args):
    if args.sources_config:
        with open(args.sources_config, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sources = payload.get("sources", payload)
        for _, text in iter_named_text_sources(sources):
            yield text
    elif args.dataset:
        ds = load_dataset(args.dataset, args.dataset_config or None, split=args.split, streaming=True)
        for row in ds:
            text = row.get(args.text_key, "")
            if text:
                yield normalize_text(text)
    elif args.jsonl_glob:
        for path in sorted(glob.glob(args.jsonl_glob)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    text = row.get(args.text_key, "")
                    if text:
                        yield normalize_text(text)
    elif args.txt_glob:
        for path in sorted(glob.glob(args.txt_glob)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = normalize_text(line)
                    if line:
                        yield line
    else:
        raise ValueError("Fournissez --sources-config, --dataset, --jsonl-glob ou --txt-glob")



def flush_split(rows, output_dir: Path, split_name: str, shard_id: int, dtype):
    arr = np.asarray(rows, dtype=dtype)
    path = output_dir / f"{split_name}-{shard_id:05d}.npy"
    np.save(path, arr)
    return path, int(arr.shape[0])



def main():
    parser = argparse.ArgumentParser(description="Packe un corpus tokenisé en shards numpy")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--sources-config", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset-config", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--jsonl-glob", type=str, default="")
    parser.add_argument("--txt-glob", type=str, default="")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data/packed")
    args = parser.parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.uint16 if tokenizer.vocab_size < 65535 else np.int32
    rows = {"train": [], "val": []}
    shard_ids = {"train": 0, "val": 0}
    buffer = []
    written = []

    for doc_id, text in enumerate(tqdm(iter_texts(args), desc="Pretokenization")):
        if args.max_docs and doc_id >= args.max_docs:
            break
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        buffer.extend(ids)
        block = args.seq_len + 1
        while len(buffer) >= block:
            row = buffer[:block]
            del buffer[:block]
            split_name = "val" if (doc_id % max(1, int(1 / max(args.val_ratio, 1e-6))) == 0) else "train"
            rows[split_name].append(row)
            if len(rows[split_name]) >= args.shard_size:
                path, count = flush_split(rows[split_name], output_dir, split_name, shard_ids[split_name], dtype)
                written.append({"path": str(path), "rows": count, "split": split_name})
                rows[split_name].clear()
                shard_ids[split_name] += 1

    for split_name in ["train", "val"]:
        if rows[split_name]:
            path, count = flush_split(rows[split_name], output_dir, split_name, shard_ids[split_name], dtype)
            written.append({"path": str(path), "rows": count, "split": split_name})

    manifest = {
        "tokenizer": os.path.abspath(args.tokenizer),
        "seq_len": args.seq_len,
        "dtype": str(dtype),
        "files": written,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
