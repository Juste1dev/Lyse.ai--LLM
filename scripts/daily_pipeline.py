from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print(f"\n[RUN] {printable}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def file_exists(path_str: str) -> bool:
    return (ROOT / path_str).exists()


def glob_exists(pattern: str) -> bool:
    return any(ROOT.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrateur quotidien 1h/jour pour Lyse AI")
    parser.add_argument("--config", type=str, default="configs/daily_67m.yaml")
    parser.add_argument("--pretrain-sources", type=str, default="configs/pretrain_sources.json")
    parser.add_argument("--sft-sources", type=str, default="configs/sft_sources.json")
    parser.add_argument("--phase", type=str, default="auto", choices=["auto", "prepare", "tokenizer", "pack", "pretrain", "build_sft", "sft"])
    parser.add_argument("--sample-lines", type=int, default=300000)
    parser.add_argument("--max-docs", type=int, default=400000)
    parser.add_argument("--max-sft-total", type=int, default=180000)
    parser.add_argument("--tokenizer-output-prefix", type=str, default="artifacts/tokenizer")
    parser.add_argument("--packed-dir", type=str, default="data/packed")
    parser.add_argument("--train-sft-output", type=str, default="data/sft/train.jsonl")
    parser.add_argument("--val-sft-output", type=str, default="data/sft/val.jsonl")
    args = parser.parse_args()

    python = sys.executable
    tokenizer_model = f"{args.tokenizer_output_prefix}.model"
    packed_manifest = str(Path(args.packed_dir) / "manifest.json")
    sft_train = args.train_sft_output
    sft_val = args.val_sft_output

    needs_tokenizer = not file_exists(tokenizer_model)
    needs_packed = not file_exists(packed_manifest) or not glob_exists(f"{args.packed_dir}/train-*.npy")
    needs_sft = not file_exists(sft_train) or not file_exists(sft_val)

    do_prepare = args.phase in {"auto", "prepare", "pretrain", "sft", "pack", "build_sft", "tokenizer"}

    if do_prepare and (args.phase in {"auto", "prepare", "tokenizer", "pretrain", "sft"}) and needs_tokenizer:
        run_command([
            python,
            "scripts/build_tokenizer.py",
            "--sources-config",
            args.pretrain_sources,
            "--sample-lines",
            str(args.sample_lines),
            "--output-prefix",
            args.tokenizer_output_prefix,
        ])

    if do_prepare and (args.phase in {"auto", "prepare", "pack", "pretrain", "sft"}) and needs_packed:
        run_command([
            python,
            "scripts/pretokenize_corpus.py",
            "--tokenizer",
            tokenizer_model,
            "--sources-config",
            args.pretrain_sources,
            "--seq-len",
            "512",
            "--output-dir",
            args.packed_dir,
            "--max-docs",
            str(args.max_docs),
        ])

    if do_prepare and (args.phase in {"auto", "prepare", "build_sft", "sft"}) and needs_sft:
        run_command([
            python,
            "scripts/build_sft_dataset.py",
            "--sources-config",
            args.sft_sources,
            "--train-output",
            sft_train,
            "--val-output",
            sft_val,
            "--max-total",
            str(args.max_sft_total),
        ])

    if args.phase == "tokenizer":
        return
    if args.phase == "pack":
        return
    if args.phase == "build_sft":
        return
    if args.phase == "prepare":
        print("Préparation terminée.")
        return

    if args.phase in {"auto", "pretrain"}:
        run_command([
            python,
            "scripts/train.py",
            "--config",
            args.config,
            "--resume",
            "auto",
        ])
        return

    if args.phase == "sft":
        run_command([
            python,
            "scripts/train_sft.py",
            "--config",
            args.config,
            "--base-checkpoint",
            "auto",
            "--resume",
            "auto",
        ])
        return


if __name__ == "__main__":
    main()
