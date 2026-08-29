from __future__ import annotations

import math
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast

from .checkpoint import load_checkpoint, save_checkpoint
from .config import SFTConfig, TrainingConfig
from .utils import (
    choose_autocast_dtype,
    count_tokens,
    ensure_dir,
    format_seconds,
    now_ts,
    resolve_precision,
    save_jsonl,
)


try:
    import bitsandbytes as bnb
except Exception:  # pragma: no cover
    bnb = None



def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig | SFTConfig, use_8bit: bool = False):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": float(cfg.weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    betas = (getattr(cfg, "beta1", 0.9), getattr(cfg, "beta2", 0.95))
    if use_8bit and bnb is not None:
        return bnb.optim.Adam8bit(groups, lr=cfg.learning_rate, betas=betas)
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=betas, fused=torch.cuda.is_available())



def cosine_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)



def evaluate(model, loader, device: str, precision: str, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    amp_enabled = precision in {"bf16", "fp16"}
    amp_dtype = choose_autocast_dtype(precision)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                loss = model(x, labels=y)["loss"]
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))



def _restore_training_state(payload: Dict, model, optimizer, scaler):
    model.load_state_dict(payload["model_state"])
    if payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    scaler_state = payload.get("scaler_state")
    if scaler_state and scaler is not None:
        scaler.load_state_dict(scaler_state)
    return int(payload.get("step", 0)), int(payload.get("seen_tokens", 0))



def train_loop(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    cfg: TrainingConfig | SFTConfig,
    device: str,
    use_8bit_optimizer: bool,
    model_config_dict: Dict,
    run_name: str,
    resume_from: str = "",
) -> Dict:
    precision = resolve_precision(getattr(cfg, "precision", "bf16"))
    amp_enabled = precision in {"bf16", "fp16"}
    amp_dtype = choose_autocast_dtype(precision)
    scaler = GradScaler(enabled=precision == "fp16")

    optimizer = build_optimizer(model, cfg, use_8bit=use_8bit_optimizer)
    step = 0
    seen_tokens = 0

    checkpoint_dir = ensure_dir(cfg.checkpoint_dir)
    logs_dir = ensure_dir(cfg.logs_dir)
    metrics_path = Path(logs_dir) / f"{run_name}_metrics.jsonl"

    if resume_from:
        payload = load_checkpoint(resume_from, map_location="cpu")
        step, seen_tokens = _restore_training_state(payload, model, optimizer, scaler)

    if hasattr(torch, "compile") and getattr(cfg, "compile_model", False):
        model = torch.compile(model)

    train_iter = cycle(train_loader)
    model.train()
    started_at = now_ts()
    last_ckpt_at = now_ts()
    stop_reason = "max_steps"
    max_runtime_seconds = max(0, int(getattr(cfg, "max_runtime_minutes", 0) or 0)) * 60

    while step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                out = model(x, labels=y)
                loss = out["loss"] / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accum_loss += float(loss.item())
            seen_tokens += count_tokens(x)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        lr = cosine_lr(
            step=step,
            max_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            max_lr=cfg.learning_rate,
            min_lr=cfg.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % cfg.log_interval == 0:
            elapsed = now_ts() - started_at
            row = {
                "step": step,
                "train_loss": accum_loss,
                "lr": lr,
                "seen_tokens": seen_tokens,
                "elapsed_seconds": round(elapsed, 2),
            }
            save_jsonl(metrics_path, row)
            print(
                f"[{run_name}] step={step} loss={accum_loss:.4f} lr={lr:.6e} "
                f"tokens={seen_tokens} elapsed={format_seconds(elapsed)}"
            )

        should_eval = step % cfg.eval_interval == 0
        should_ckpt = step % cfg.checkpoint_interval_steps == 0 or (now_ts() - last_ckpt_at) / 60 >= cfg.checkpoint_interval_minutes

        if should_eval:
            val_loss = evaluate(model, val_loader, device=device, precision=precision, max_batches=cfg.eval_batches)
            row = {"step": step, "val_loss": val_loss}
            save_jsonl(metrics_path, row)
            print(f"[{run_name}] step={step} val_loss={val_loss:.4f}")

        if should_ckpt:
            payload = {
                "step": step,
                "seen_tokens": seen_tokens,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "model_config": model_config_dict,
                "run_name": run_name,
            }
            ckpt_path = save_checkpoint(str(checkpoint_dir), step=step, payload=payload)
            last_ckpt_at = now_ts()
            print(f"Checkpoint sauvegardé: {ckpt_path}")

        if max_runtime_seconds and (now_ts() - started_at) >= max_runtime_seconds:
            stop_reason = "max_runtime_reached"
            print(f"[{run_name}] arrêt propre après limite de temps ({format_seconds(now_ts() - started_at)})")
            break

    final_payload = {
        "step": step,
        "seen_tokens": seen_tokens,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "model_config": model_config_dict,
        "run_name": run_name,
    }
    final_path = save_checkpoint(str(checkpoint_dir), step=step, payload=final_payload)
    return {"checkpoint_path": str(final_path), "step": step, "seen_tokens": seen_tokens, "stop_reason": stop_reason}
