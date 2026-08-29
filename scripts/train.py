from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyse_ai.config import load_config
from lyse_ai.data import create_pretrain_dataloader
from lyse_ai.model import LyseTransformer
from lyse_ai.trainer import train_loop
from lyse_ai.utils import latest_checkpoint_path, set_seed


def main():
    parser = argparse.ArgumentParser(description="Pré-entraînement causal LM pour Lyse AI")
    parser.add_argument("--config", type=str, default="configs/base_67m.yaml")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    train_loader = create_pretrain_dataloader(
        cfg.data.train_shards_glob,
        batch_size=cfg.training.micro_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = create_pretrain_dataloader(
        cfg.data.val_shards_glob,
        batch_size=cfg.training.micro_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = LyseTransformer(cfg.model).to(device)
    print(f"Paramètres entraînables: {model.estimate_num_params():,}")

    resume = args.resume or cfg.training.resume_from
    if resume == "auto":
        latest = latest_checkpoint_path(cfg.training.checkpoint_dir)
        resume = str(latest) if latest else ""

    try:
        result = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.training,
            device=device,
            use_8bit_optimizer=cfg.training.use_8bit_optimizer and device == "cuda",
            model_config_dict=cfg.model.__dict__,
            run_name="pretrain",
            resume_from=resume,
        )
        print(result)
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur. Relancez avec --resume auto pour reprendre.")


if __name__ == "__main__":
    main()
