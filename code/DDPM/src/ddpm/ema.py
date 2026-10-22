"""Exponential moving average of model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    """Keep a shadow copy of a model and update it after every optimizer step."""

    def __init__(self, beta: float) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("EMA beta must be in [0, 1)")
        self.beta = beta
        self.model: nn.Module | None = None
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.model is None:
            self.model = copy.deepcopy(model).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        else:
            for ema_parameter, parameter in zip(
                self.model.parameters(), model.parameters()
            ):
                ema_parameter.mul_(self.beta).add_(
                    parameter.detach(), alpha=1.0 - self.beta
                )
            for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)
        self.updates += 1

    def state_dict(self) -> dict[str, torch.Tensor] | None:
        return None if self.model is None else self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.model is None:
            raise RuntimeError("EMA has no shadow model to load into")
        self.model.load_state_dict(state_dict)
