"""Typed configuration objects for data, model, diffusion and runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

ScheduleName = Literal["linear", "cosine", "quadratic", "sigmoid"]
LossName = Literal["l1", "l2", "huber"]
SamplerName = Literal["ddpm", "ddim"]
DataSource = Literal["synthetic", "imagefolder", "cifar10", "hf"]

SCHEDULES: tuple[str, ...] = ("linear", "cosine", "quadratic", "sigmoid")
LOSSES: tuple[str, ...] = ("l1", "l2", "huber")
SAMPLERS: tuple[str, ...] = ("ddpm", "ddim")
DATA_SOURCES: tuple[str, ...] = ("synthetic", "imagefolder", "cifar10", "hf")


def to_dict(value: Any) -> Any:
    """Convert dataclasses, paths and tuples into JSON-friendly values."""

    if is_dataclass(value):
        return {item.name: to_dict(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    return value


def _filtered(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


@dataclass
class ModelConfig:
    """The U-Net shape. Defaults are intentionally small enough for CPU runs."""

    dim: int = 32
    dim_mults: tuple[int, ...] = (1, 2, 4)
    channels: int = 3
    resnet_block_groups: int = 8
    self_condition: bool = False

    def __post_init__(self) -> None:
        self.dim_mults = tuple(int(mult) for mult in self.dim_mults)
        if self.dim < 4:
            raise ValueError("model.dim must be >= 4")
        if not self.dim_mults or any(mult < 1 for mult in self.dim_mults):
            raise ValueError("model.dim_mults must contain positive integers")
        if self.channels not in (1, 3):
            raise ValueError("model.channels must be 1 or 3")
        if self.resnet_block_groups < 1:
            raise ValueError("model.resnet_block_groups must be >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**_filtered(cls, dict(data)))


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    loss_type: str = "huber"

    def __post_init__(self) -> None:
        if self.timesteps < 2:
            raise ValueError("diffusion.timesteps must be >= 2")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"diffusion.schedule must be one of {SCHEDULES}")
        if self.loss_type not in LOSSES:
            raise ValueError(f"diffusion.loss_type must be one of {LOSSES}")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("need 0 < beta_start < beta_end < 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffusionConfig:
        return cls(**_filtered(cls, dict(data)))


@dataclass
class DataConfig:
    source: str = "synthetic"
    image_size: int = 64
    channels: int = 3
    batch_size: int = 8
    num_workers: int = 0
    data_dir: str | None = None
    hf_dataset: str | None = None
    hf_split: str = "train"
    hf_image_key: str = "image"
    cifar_root: str = "data"
    download: bool = False
    synthetic_size: int = 512
    augment: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.source not in DATA_SOURCES:
            raise ValueError(f"data.source must be one of {DATA_SOURCES}")
        if self.image_size < 8:
            raise ValueError("data.image_size must be >= 8")
        if self.channels not in (1, 3):
            raise ValueError("data.channels must be 1 or 3")
        if self.batch_size < 1:
            raise ValueError("data.batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be >= 0")
        if self.synthetic_size < 1:
            raise ValueError("data.synthetic_size must be >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        return cls(**_filtered(cls, dict(data)))


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    output_dir: str = "runs/ddpm"
    epochs: int = 10
    max_steps: int | None = None
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    log_every: int = 50
    sample_every: int = 1000
    sample_steps: int = 50
    sampler: str = "ddim"
    num_sample_images: int = 16
    device: str = "cpu"
    seed: int = 42
    resume: str | None = None

    def __post_init__(self) -> None:
        if self.data.channels != self.model.channels:
            raise ValueError("data.channels and model.channels must match")
        downsamples = max(0, len(self.model.dim_mults) - 1)
        if self.data.image_size % (2**downsamples) != 0:
            raise ValueError(
                "data.image_size must be divisible by 2 ** (len(model.dim_mults) - 1); "
                f"got image_size={self.data.image_size}, dim_mults={self.model.dim_mults}"
            )
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"train.sampler must be one of {SAMPLERS}")
        if self.sample_every < 0:
            raise ValueError("sample_every must be >= 0")
        if self.sample_steps < 1:
            raise ValueError("sample_steps must be >= 1")
        if self.num_sample_images < 0:
            raise ValueError("num_sample_images must be >= 0")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return to_dict(self)


@dataclass
class SampleConfig:
    checkpoint: str
    output: str = "samples/ddpm.png"
    num_samples: int = 16
    batch_size: int = 8
    sampler: str = "ddim"
    steps: int = 50
    eta: float = 0.0
    seed: int = 42
    device: str = "cpu"
    use_ema: bool = True
    grid_cols: int = 4
    individual: bool = False

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint is required")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sample.sampler must be one of {SAMPLERS}")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if self.grid_cols < 1:
            raise ValueError("grid_cols must be >= 1")
