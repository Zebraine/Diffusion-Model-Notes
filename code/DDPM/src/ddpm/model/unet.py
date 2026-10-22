"""The DDPM U-Net: residual blocks, linear attention and time conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from itertools import pairwise

import torch
from torch import Tensor, nn

from ..utils import default
from .attention import Attention, LinearAttention, PreNorm
from .blocks import (
    Downsample,
    Residual,
    ResnetBlock,
    SinusoidalPositionEmbeddings,
    Upsample,
)


class UNet(nn.Module):
    """A compact U-Net with group norm and sinusoidal time embeddings.

    The architecture follows the annotated-diffusion blog, which in turn follows
    the PixelCNN++/DDPM backbone. It is fully convolutional, so the spatial size
    only needs to be divisible by ``2 ** (len(dim_mults) - 1)``.
    """

    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        channels: int = 3,
        self_condition: bool = False,
        resnet_block_groups: int = 4,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        assert init_dim is not None
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *[dim * mult for mult in dim_mults]]
        in_out = list(pairwise(dims))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs: nn.ModuleList = nn.ModuleList()
        self.ups: nn.ModuleList = nn.ModuleList()
        num_resolutions = len(in_out)

        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= num_resolutions - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = index == num_resolutions - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        assert self.out_dim is not None
        self.final_res_block = block_klass(dim + init_dim, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(
        self, x: Tensor, time: Tensor, x_self_cond: Tensor | None = None
    ) -> Tensor:
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        residual = x.clone()
        time_emb = self.time_mlp(time)
        skip_connections: list[Tensor] = []

        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time_emb)
            skip_connections.append(x)
            x = block2(x, time_emb)
            x = attention(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        for block1, block2, attention, upsample in self.ups:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block1(x, time_emb)
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block2(x, time_emb)
            x = attention(x)
            x = upsample(x)

        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x, time_emb)
        return self.final_conv(x)


# The blog calls the class ``Unet``; keep that spelling as an alias.
Unet = UNet
