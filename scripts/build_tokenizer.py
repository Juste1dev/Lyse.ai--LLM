from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sentencepiece as spm
from datasets import load_dataset

from lyse_ai.dataset_mixtures import iter_named_text_sources
from lyse_ai.tokenizer import SPECIAL_TOKENS



def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())



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



def main():
    parser = argparse.ArgumentParser(description="Entraîne un tokenizer SentencePiece pour Lyse AI")
    parser.add_argument("--sources-config", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset-config", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--jsonl-glob", type=str, default="")
    parser.add_argument("--txt-glob", type=str, default="")
    parser.add_argument("--sample-lines", type=int, default=200000)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--output-prefix", type=str, default="artifacts/tokenizer")
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for i, text in enumerate(tqdm(iter_texts(args), total=args.sample_lines, desc="Échantillonnage tokenizer")):
            if i >= args.sample_lines:
                break
            tmp.write(text + "\n")
        tmp_path = tmp.name

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=str(output_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=SPECIAL_TOKENS[3:],
        max_sentencepiece_length=32,
        split_digits=True,
        byte_fallback=True,
        train_extremely_large_corpus=True,
    )
    print(f"Tokenizer créé: {output_prefix}.model")


if __name__ == "__main__":
    main()
