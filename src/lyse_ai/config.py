from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    name: str = "lyse-67m"
    vocab_size: int = 32000
    max_seq_len: int = 512
    n_layers: int = 16
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = True
    layer_norm_epsilon: float = 1e-5


@dataclass
class DataConfig:
    seq_len: int = 512
    train_shards_glob: str = "data/packed/train-*.npy"
    val_shards_glob: str = "data/packed/val-*.npy"
    shuffle_shards: bool = True
    text_key: str = "text"
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    compile_model: bool = False
    use_8bit_optimizer: bool = True
    batch_size: int = 32
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 500
    max_steps: int = 50000
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_interval_steps: int = 250
    checkpoint_interval_minutes: int = 15
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "logs"
    tokenizer_path: str = "artifacts/tokenizer.model"
    resume_from: str = ""
    max_runtime_minutes: int = 0


@dataclass
class SFTConfig:
    train_file: str = "data/sft/train.jsonl"
    val_file: str = "data/sft/val.jsonl"
    max_seq_len: int = 512
    batch_size: int = 8
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    max_steps: int = 5000
    grad_clip: float = 1.0
    precision: str = "bf16"
    compile_model: bool = False
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval_steps: int = 100
    checkpoint_interval_minutes: int = 10
    checkpoint_dir: str = "checkpoints_sft"
    logs_dir: str = "logs_sft"
    max_runtime_minutes: int = 0


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _merge_dataclass(dc_cls, payload: Dict[str, Any]):
    default = dc_cls()
    for key, value in payload.items():
        if hasattr(default, key):
            setattr(default, key, value)
    return default



def load_config(path: str | Path) -> FullConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return FullConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        data=_merge_dataclass(DataConfig, raw.get("data", {})),
        training=_merge_dataclass(TrainingConfig, raw.get("training", {})),
        sft=_merge_dataclass(SFTConfig, raw.get("sft", {})),
    )



def save_config(cfg: FullConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
