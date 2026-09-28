from __future__ import annotations

# Tokenizer 32k entraîné sur du vrai code (voie B, entraînement from scratch).
# Le build_tokenizer.py du projet échantillonne avec " ".join(text.split()) :
# le tokenizer ne voit alors jamais une suite d'espaces ou une tab, et chaque
# indentation lui coûte une tripotée de tokens à l'inférence. Ici on lui donne
# le code tel quel. Les paramètres SentencePiece restent ceux du projet
# (pad=0, bos=1, eos=2, unk=3, mêmes symboles spéciaux) pour rester cohérent
# avec src/lyse_ai/tokenizer.py.

import argparse
import json
import sys
import tempfile
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sentencepiece as spm

from lyse_ai.tokenizer import SPECIAL_TOKENS

# on tronque les gros fichiers pour équilibrer l'échantillon
MAX_CHARS_PER_DOC = 20_000


def sample_code_doc(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ""
    return text[:MAX_CHARS_PER_DOC]


def iter_mixture_samples(mixture_path: str):
    with open(mixture_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sources = payload.get("sources", payload)
    for idx, source in enumerate(sources):
        if not source.get("enabled", True):
            continue
        name = str(source.get("name") or source.get("dataset") or f"source_{idx}")
        text_key = source.get("text_key", "content")
        max_docs = int(source.get("max_docs", 0) or 0)
        try:
            ds = load_dataset(
                source["dataset"],
                source.get("dataset_config") or None,
                split=source.get("split", "train"),
                data_dir=source.get("data_dir") or None,
                streaming=True,
            )
            emitted = 0
            for row in ds:
                text = sample_code_doc(row.get(text_key, ""))
                if text:
                    yield text
                    emitted += 1
                if max_docs and emitted >= max_docs:
                    break
        except Exception as exc:
            print(f"[WARN] source ignorée : {name} ({exc})")


def main():
    parser = argparse.ArgumentParser(description="Entraîne un tokenizer SentencePiece sur du code")
    parser.add_argument("--mixture", type=str, required=True, help="même json que le corpus, ex data/mixtures/code_pretrain.json")
    parser.add_argument("--sample-docs", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--output-prefix", type=str, default="artifacts/tokenizer_code")
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for i, text in enumerate(tqdm(iter_mixture_samples(args.mixture), total=args.sample_docs, desc="Échantillonnage tokenizer")):
            if i >= args.sample_docs:
                break
            tmp.write(text + "\n\n")  # petite séparation entre documents
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
    print(f"Tokenizer créé : {output_prefix}.model")


if __name__ == "__main__":
    main()
