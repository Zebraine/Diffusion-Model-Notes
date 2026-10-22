"""CPU-friendly DDPM training loop with EMA, checkpoints and sample previews."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import DiffusionConfig, ModelConfig, TrainConfig, to_dict
from .data import build_dataloader
from .diffusion import GaussianDiffusion
from .ema import EMA
from .model import build_unet
from .utils import (
    AverageMeter,
    append_jsonl,
    count_parameters,
    ensure_dir,
    format_duration,
    resolve_device,
    set_seed,
    write_json,
)
from .visualize import save_image_grid

logger = logging.getLogger(__name__)
CHECKPOINT_FORMAT = "dppm-checkpoint-v1"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    step: int,
) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "model_config": to_dict(config.model),
        "diffusion_config": to_dict(config.diffusion),
        "data_config": to_dict(config.data),
        "epoch": int(epoch),
        "step": int(step),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _checkpoint_configs(checkpoint: dict[str, Any], config: TrainConfig) -> None:
    saved_model = ModelConfig.from_dict(checkpoint.get("model_config", {}))
    if saved_model != config.model:
        raise ValueError(
            "the checkpoint was trained with a different model config: "
            f"{saved_model} != {config.model}"
        )
    saved_diffusion = DiffusionConfig.from_dict(checkpoint.get("diffusion_config", {}))
    if saved_diffusion != config.diffusion:
        raise ValueError(
            "the checkpoint was trained with a different diffusion config: "
            f"{saved_diffusion} != {config.diffusion}"
        )


def _sample_and_save(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    config: TrainConfig,
    device: torch.device,
    sample_dir: Path,
    step: int,
) -> Path | None:
    if config.num_sample_images < 1:
        return None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if config.sampler == "ddim":
                images = diffusion.ddim_sample(
                    model,
                    image_size=config.data.image_size,
                    batch_size=config.num_sample_images,
                    channels=config.data.channels,
                    device=device,
                    steps=config.sample_steps,
                    progress=False,
                )
            else:
                images = diffusion.sample(
                    model,
                    image_size=config.data.image_size,
                    batch_size=config.num_sample_images,
                    channels=config.data.channels,
                    device=device,
                    progress=False,
                )
        path = sample_dir / f"step_{step:06d}.png"
        save_image_grid(images, path, nrow=min(4, config.num_sample_images))
        logger.info("sample preview: %s", path)
        return path
    finally:
        if was_training:
            model.train()


def train(config: TrainConfig) -> dict[str, Any]:
    """Train a DDPM and return a small summary.

    The function is intentionally usable both from the CLI and from Python:

    >>> config = TrainConfig(output_dir="runs/demo", max_steps=10)
    >>> train(config)  # doctest: +SKIP
    """

    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    sample_dir = ensure_dir(run_dir / "samples")
    metrics_path = run_dir / "metrics.jsonl"
    write_json(run_dir / "config.json", config.to_dict())

    dataloader = build_dataloader(config.data)
    model = build_unet(config.model).to(device)
    diffusion = GaussianDiffusion(config.diffusion).to(device)
    ema = EMA(config.ema_decay) if config.ema_decay > 0 else None
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    start_epoch = 1
    global_step = 0
    if config.resume:
        checkpoint = load_checkpoint(config.resume, device)
        _checkpoint_configs(checkpoint, config)
        model.load_state_dict(checkpoint["model"])
        if ema is not None:
            ema.update(model)
            if checkpoint.get("ema"):
                ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        logger.info(
            "resumed from %s (epoch=%d, step=%d)",
            config.resume,
            start_epoch,
            global_step,
        )
    elif ema is not None:
        ema.update(model)

    logger.info("device: %s", device)
    logger.info("model parameters: %s", f"{count_parameters(model):,}")
    logger.info(
        "data: %s · %d samples · image_size=%d · batch_size=%d",
        config.data.source,
        len(dataloader.dataset),  # type: ignore[arg-type]
        config.data.image_size,
        config.data.batch_size,
    )
    logger.info(
        "diffusion: T=%d · schedule=%s · loss=%s · sampler=%s",
        config.diffusion.timesteps,
        config.diffusion.schedule,
        config.diffusion.loss_type,
        config.sampler,
    )

    loss_meter = AverageMeter()
    started = time.time()
    stopped_early = False
    final_loss: float | None = None
    last_checkpoint: Path | None = None
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        progress = tqdm(
            dataloader,
            desc=f"epoch {epoch}/{config.epochs}",
            leave=False,
        )
        for batch in progress:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=False)
            batch_size = images.shape[0]
            timesteps = torch.randint(
                0,
                diffusion.timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            loss = diffusion.p_losses(model, images, timesteps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            global_step += 1
            loss_value = float(loss.detach().cpu())
            final_loss = loss_value
            loss_meter.update(loss_value, batch_size)
            progress.set_postfix(loss=f"{loss_value:.4f}", step=global_step)

            if global_step == 1 or global_step % config.log_every == 0:
                final_loss = loss_meter.average
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": final_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed": round(time.time() - started, 2),
                }
                append_jsonl(metrics_path, record)
                logger.info(
                    "step %d · epoch %d · loss %.4f · elapsed %s",
                    global_step,
                    epoch,
                    final_loss,
                    format_duration(time.time() - started),
                )
                loss_meter.reset()

            if config.sample_every > 0 and global_step % config.sample_every == 0:
                sample_model = ema.model if ema is not None else model
                assert sample_model is not None
                _sample_and_save(
                    sample_model,
                    diffusion,
                    config,
                    device,
                    sample_dir,
                    global_step,
                )
                model.train()

            if config.max_steps is not None and global_step >= config.max_steps:
                stopped_early = True
                break

        progress.close()
        last_epoch = epoch
        last_checkpoint = save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            ema,
            optimizer,
            config,
            epoch,
            global_step,
        )
        if stopped_early:
            break

    if last_checkpoint is None:
        last_checkpoint = save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            ema,
            optimizer,
            config,
            min(start_epoch - 1, config.epochs),
            global_step,
        )

    summary = {
        "checkpoint": str(last_checkpoint),
        "steps": global_step,
        "epochs": last_epoch,
        "final_loss": final_loss,
        "elapsed_seconds": round(time.time() - started, 2),
        "device": str(device),
    }
    write_json(run_dir / "summary.json", summary)
    logger.info(
        "done: steps=%d · loss=%s · checkpoint=%s · elapsed=%s",
        global_step,
        "n/a" if final_loss is None else f"{final_loss:.4f}",
        last_checkpoint,
        format_duration(time.time() - started),
    )
    return summary
