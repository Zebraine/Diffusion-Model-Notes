"""CPU-friendly Denoising Diffusion Probabilistic Models.

The package is split by responsibility:

* ``model``      – U-Net, residual/attention blocks
* ``diffusion``  – schedules, forward process, losses, DDPM/DDIM sampling
* ``data``       – datasets and transforms
* ``train``      – training loop, EMA, checkpoints
* ``sample``     – checkpoint loading and image generation
* ``cli``        – ``dppm train|sample|inspect|smoke``
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import DataConfig, DiffusionConfig, ModelConfig, SampleConfig, TrainConfig
from .diffusion import GaussianDiffusion
from .model import UNet, Unet, build_unet


def main(argv: list[str] | None = None) -> int:
    """Lazy CLI entry point for ``from dppm import main`` compatibility."""

    from .cli import main as cli_main

    return cli_main(argv)


__all__ = [
    "DataConfig",
    "DiffusionConfig",
    "GaussianDiffusion",
    "ModelConfig",
    "SampleConfig",
    "TrainConfig",
    "UNet",
    "Unet",
    "__version__",
    "build_unet",
    "main",
]
