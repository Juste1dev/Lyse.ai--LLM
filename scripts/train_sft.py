from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyse_ai.config import load_config
from lyse_ai.model import LyseTransformer
from lyse_ai.sft import create_sft_dataloader
from lyse_ai.tokenizer import SentencePieceTokenizer
from lyse_ai.trainer import train_loop
from lyse_ai.utils import latest_checkpoint_path, set_seed


def main():
    parser = argparse.ArgumentParser(description="SFT conversationnel pour Lyse AI")
    parser.add_argument("--config", type=str, default="configs/base_67m.yaml")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    tokenizer = SentencePieceTokenizer(cfg.training.tokenizer_path)
    train_loader = create_sft_dataloader(
        cfg.sft.train_file,
        tokenizer=tokenizer,
        batch_size=cfg.sft.micro_batch_size,
        shuffle=True,
        max_seq_len=cfg.sft.max_seq_len,
    )
    val_loader = create_sft_dataloader(
        cfg.sft.val_file,
        tokenizer=tokenizer,
        batch_size=cfg.sft.micro_batch_size,
        shuffle=False,
        max_seq_len=cfg.sft.max_seq_len,
    )

    base_checkpoint = args.base_checkpoint
    if base_checkpoint == "auto":
        latest_pretrain = latest_checkpoint_path(cfg.training.checkpoint_dir)
        if latest_pretrain is None:
            raise FileNotFoundError(
                f"Aucun checkpoint prétrain trouvé dans {cfg.training.checkpoint_dir}."
            )
        base_checkpoint = str(latest_pretrain)

    base = torch.load(base_checkpoint, map_location="cpu")
    model = LyseTransformer(cfg.model)
    model.load_state_dict(base["model_state"])
    model.to(device)

    resume = args.resume
    if resume == "auto":
        latest = latest_checkpoint_path(cfg.sft.checkpoint_dir)
        resume = str(latest) if latest else ""

    try:
        result = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.sft,
            device=device,
            use_8bit_optimizer=False,
            model_config_dict=cfg.model.__dict__,
            run_name="sft",
            resume_from=resume,
        )
        print(result)
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur. Relancez avec --resume auto pour reprendre.")


if __name__ == "__main__":
    main()
