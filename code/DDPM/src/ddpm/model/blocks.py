"""Building blocks copied from the annotated-diffusion U-Net."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from ..utils import exists


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int, dim_out: int | None = None) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out or dim, 3, padding=1),
    )


def Downsample(dim: int, dim_out: int | None = None) -> nn.Module:
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out or dim, 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("sinusoidal embedding dim must be >= 4")
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings[..., : self.dim]


class WeightStandardizedConv2d(nn.Conv2d):
    """Weight standardization keeps activations well behaved without BatchNorm."""

    def forward(self, input: Tensor) -> Tensor:
        eps = 1e-5 if input.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(
            input,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self, x: Tensor, scale_shift: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: int | None = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor | None = None) -> Tensor:
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[:, :, None, None]
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


def validate_dim_mults(dim_mults: Sequence[int]) -> None:
    if not dim_mults or any(mult < 1 for mult in dim_mults):
        raise ValueError("dim_mults must be a non-empty sequence of positive integers")
