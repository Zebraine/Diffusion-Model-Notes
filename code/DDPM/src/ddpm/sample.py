"""Checkpoint loading, sampling and image export."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import DiffusionConfig, ModelConfig, SampleConfig
from .diffusion import GaussianDiffusion
from .model import build_unet
from .utils import resolve_device, set_seed
from .visualize import save_image_grid, save_individual_images

logger = logging.getLogger(__name__)


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu", use_ema: bool = True
) -> tuple[nn.Module, GaussianDiffusion, int, int, dict[str, Any]]:
    """Load a training checkpoint and return model, diffusion, size, channels, raw."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_data = checkpoint.get("model_config") or checkpoint.get("config", {}).get(
        "model", {}
    )
    diffusion_data = checkpoint.get("diffusion_config") or checkpoint.get(
        "config", {}
    ).get("diffusion", {})
    data_data = checkpoint.get("data_config") or checkpoint.get("config", {}).get(
        "data", {}
    )

    model_config = ModelConfig.from_dict(dict(model_data))
    diffusion_config = DiffusionConfig.from_dict(dict(diffusion_data))
    image_size = int(data_data.get("image_size", 64))
    channels = int(data_data.get("channels", model_config.channels))

    model: nn.Module = build_unet(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if use_ema and checkpoint.get("ema"):
        ema_model = copy.deepcopy(model).eval()
        ema_model.load_state_dict(checkpoint["ema"])
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        model = ema_model
        logger.info("using EMA weights")
    elif use_ema:
        logger.warning("checkpoint has no EMA weights; using the raw model")

    diffusion = GaussianDiffusion(diffusion_config).to(device)
    return model, diffusion, image_size, channels, checkpoint


def generate(config: SampleConfig) -> Path:
    """Generate samples and save a grid. Returns the output path."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    model, diffusion, image_size, channels, _ = load_checkpoint(
        config.checkpoint, device=device, use_ema=config.use_ema
    )

    batches: list[torch.Tensor] = []
    remaining = config.num_samples
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(config.batch_size, remaining)
            if config.sampler == "ddim":
                images = diffusion.ddim_sample(
                    model,
                    image_size=image_size,
                    batch_size=batch_size,
                    channels=channels,
                    device=device,
                    steps=config.steps,
                    eta=config.eta,
                    progress=True,
                )
            else:
                images = diffusion.sample(
                    model,
                    image_size=image_size,
                    batch_size=batch_size,
                    channels=channels,
                    device=device,
                    progress=True,
                )
            batches.append(images.detach().cpu())
            remaining -= batch_size

    samples = torch.cat(batches, dim=0)
    output = Path(config.output)
    if output.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        output = output.with_suffix(".png")
    save_image_grid(samples, output, nrow=config.grid_cols)
    logger.info("saved %d samples to %s", len(samples), output)

    if config.individual:
        paths = save_individual_images(samples, output.parent, prefix=output.stem)
        logger.info("saved %d individual images under %s", len(paths), output.parent)

    return output
