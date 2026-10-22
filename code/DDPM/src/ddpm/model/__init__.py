"""Model components and a config-driven factory."""

from __future__ import annotations

from ..config import ModelConfig
from .attention import Attention, LinearAttention, PreNorm
from .blocks import (
    Block,
    Downsample,
    Residual,
    ResnetBlock,
    SinusoidalPositionEmbeddings,
    Upsample,
    WeightStandardizedConv2d,
)
from .unet import UNet, Unet


def build_unet(config: ModelConfig) -> UNet:
    return UNet(
        dim=config.dim,
        dim_mults=config.dim_mults,
        channels=config.channels,
        self_condition=config.self_condition,
        resnet_block_groups=config.resnet_block_groups,
    )


__all__ = [
    "Attention",
    "Block",
    "Downsample",
    "LinearAttention",
    "PreNorm",
    "Residual",
    "ResnetBlock",
    "SinusoidalPositionEmbeddings",
    "UNet",
    "Unet",
    "Upsample",
    "WeightStandardizedConv2d",
    "build_unet",
]
