from __future__ import annotations

# Pré-tokenisation d'un corpus de CODE.
# C'est une variante de pretokenize_corpus.py : la version d'origine passe chaque
# texte dans normalize_text(), qui compacte chaque ligne sur une seule espace.
# Très bien pour de la prose, catastrophique pour du code — l'indentation Python
# saute, et avec elle une bonne partie du signal. Ici on garde la mise en forme
# et on se contente de jeter les fichiers qui n'ont rien à faire là-dedans
# (binaire, minifié, données tabulaires déguisées en .py...).

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyse_ai.tokenizer import SentencePieceTokenizer

# filtres anti-bruit, ajustables si besoin
MAX_CHARS_PER_FILE = 200_000
MAX_AVG_LINE_LEN = 120   # au-delà c'est presque toujours du minifié ou des données
MAX_LINE_LEN = 2_000
MIN_LINES = 3


def normalize_code(text: str) -> str:
    """Nettoie un fichier de code sans toucher à l'indentation ni aux lignes vides."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > MAX_CHARS_PER_FILE:
        return ""
    if any(ord(ch) < 32 and ch not in "\t\n" for ch in text):
        return ""  # caractères de contrôle : quasi sûrement du binaire
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) < MIN_LINES:
        return ""
    longest = max(len(ln) for ln in lines)
    if longest > MAX_LINE_LEN:
        return ""
    avg = sum(len(ln) for ln in lines) / len(lines)
    if avg > MAX_AVG_LINE_LEN:
        return ""
    return "\n".join(lines)


def _license_ok(row, license_key: str, allowed_licenses) -> bool:
    if not allowed_licenses:
        return True
    lic = str(row.get(license_key, "")).upper()
    return any(a in lic for a in allowed_licenses)


def iter_source_texts(source: dict):
    dataset = source["dataset"]
    config = source.get("dataset_config") or None
    data_dir = source.get("data_dir") or None
    split = source.get("split", "train")
    text_key = source.get("text_key", "content")
    license_key = source.get("license_key", "license")
    allowed = {str(a).upper() for a in source.get("allowed_licenses", [])}
    max_docs = int(source.get("max_docs", 0) or 0)

    ds = load_dataset(dataset, config, split=split, data_dir=data_dir, streaming=True)
    emitted = 0
    for row in ds:
        if not _license_ok(row, license_key, allowed):
            continue
        text = normalize_code(row.get(text_key, ""))
        if not text:
            continue
        yield text
        emitted += 1
        if max_docs and emitted >= max_docs:
            break


def iter_texts(args):
    if args.mixture:
        with open(args.mixture, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sources = payload.get("sources", payload)
        for idx, source in enumerate(sources):
            if not source.get("enabled", True):
                print(f"[INFO] source désactivée : {source.get('name', f'source_{idx}')}")
                continue
            name = str(source.get("name") or source.get("dataset") or f"source_{idx}")
            try:
                for text in iter_source_texts(source):
                    yield name, text
            except Exception as exc:
                # on logue et on continue, comme dans dataset_mixtures.py
                print(f"[WARN] source ignorée : {name} ({exc})")
    elif args.jsonl_glob:
        # pour recycler des dumps locaux (clé "content" ou "text")
        for path in sorted(glob.glob(args.jsonl_glob, recursive=True)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    text = normalize_code(row.get(args.text_key, "") or row.get("text", ""))
                    if text:
                        yield f"local:{path}", text
    else:
        raise ValueError("il faut --mixture ou --jsonl-glob")


def flush_split(rows, output_dir: Path, split_name: str, shard_id: int, dtype):
    arr = np.asarray(rows, dtype=dtype)
    path = output_dir / f"{split_name}-{shard_id:05d}.npy"
    np.save(path, arr)
    return path, int(arr.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Packe un corpus de code en shards .npy (indentation préservée)")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--mixture", type=str, default="", help="json des datasets, ex data/mixtures/code_pretrain.json")
    parser.add_argument("--jsonl-glob", type=str, default="", help="dumps locaux jsonl (clé 'content' ou 'text')")
    parser.add_argument("--text-key", type=str, default="content")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data/packed_code")
    args = parser.parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.uint16 if tokenizer.vocab_size < 65535 else np.int32
    rows = {"train": [], "val": []}
    shard_ids = {"train": 0, "val": 0}
    buffer = []
    written = []
    source_counts: Counter = Counter()

    # même logique d'empaquetage que pretokenize_corpus.py : blocs de seq_len+1,
    # un document sur N part en val, flush par shards
    for doc_id, (source_name, text) in enumerate(tqdm(iter_texts(args), desc="Pretokenization code")):
        if args.max_docs and doc_id >= args.max_docs:
            break
        source_counts[source_name] += 1
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
        "mixture": os.path.abspath(args.mixture) if args.mixture else "",
        "docs_by_source": dict(source_counts),
        "files": written,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
