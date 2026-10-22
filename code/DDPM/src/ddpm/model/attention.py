"""Self-attention blocks used by the U-Net."""

from __future__ import annotations

from einops import rearrange
from torch import Tensor, einsum, nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(tensor, "b (h d) x y -> b h d (x y)", h=self.heads)
            for tensor in (q, k, v)
        )
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(tensor, "b (h d) x y -> b h d (x y)", h=self.heads)
            for tensor in (q, k, v)
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = einsum("b h d n, b h e n -> b h d e", k, v)
        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h d (x y) -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))
