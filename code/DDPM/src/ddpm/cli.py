"""Command line interface: train, sample, inspect and smoke."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import (
    DATA_SOURCES,
    LOSSES,
    SAMPLERS,
    SCHEDULES,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    SampleConfig,
    TrainConfig,
)
from .utils import configure_logging, count_parameters, resolve_device, write_json

logger = logging.getLogger(__name__)


def _dim_mults(value: str) -> tuple[int, ...]:
    try:
        parts = tuple(
            int(part.strip())
            for part in value.replace("x", ",").split(",")
            if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid dim_mults: {value!r}") from error
    if not parts or any(part < 1 for part in parts):
        raise argparse.ArgumentTypeError("dim_mults must be positive integers")
    return parts


def _add_common_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dim", type=int, default=32, help="base U-Net width")
    parser.add_argument(
        "--dim-mults",
        type=_dim_mults,
        default=(1, 2, 4),
        help="channel multipliers, e.g. 1,2,4",
    )
    parser.add_argument("--groups", type=int, default=8, help="GroupNorm groups")
    parser.add_argument("--self-condition", action="store_true")


def _add_diffusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=SCHEDULES, default="linear")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--loss", choices=LOSSES, default="huber")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dppm",
        description="CPU-friendly Denoising Diffusion Probabilistic Models.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a DDPM")
    train_parser.add_argument(
        "--source",
        choices=DATA_SOURCES,
        default=None,
        help="dataset source (inferred from data-dir / hf-dataset / download)",
    )
    train_parser.add_argument("--data-dir", help="ImageFolder directory")
    train_parser.add_argument("--hf-dataset", help="HuggingFace dataset name")
    train_parser.add_argument("--hf-split", default="train")
    train_parser.add_argument("--hf-image-key", default="image")
    train_parser.add_argument("--cifar-root", default="data")
    train_parser.add_argument("--download", action="store_true")
    train_parser.add_argument("--synthetic-size", type=int, default=512)
    train_parser.add_argument("--image-size", type=int, default=64)
    train_parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--no-augment", action="store_true")
    _add_model_args(train_parser)
    _add_diffusion_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--max-steps", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--ema-decay", type=float, default=0.9999)
    train_parser.add_argument("--no-ema", action="store_true")
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--sample-every", type=int, default=1000)
    train_parser.add_argument("--sample-steps", type=int, default=50)
    train_parser.add_argument("--sampler", choices=SAMPLERS, default="ddim")
    train_parser.add_argument("--num-sample-images", type=int, default=16)
    train_parser.add_argument("--output-dir", default="runs/ddpm")
    train_parser.add_argument("--resume", default=None)
    _add_common_device(train_parser)

    sample_parser = subparsers.add_parser("sample", help="sample from a checkpoint")
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--output", default="samples/ddpm.png")
    sample_parser.add_argument("--num-samples", type=int, default=16)
    sample_parser.add_argument("--batch-size", type=int, default=8)
    sample_parser.add_argument("--sampler", choices=SAMPLERS, default="ddim")
    sample_parser.add_argument("--steps", type=int, default=50)
    sample_parser.add_argument("--eta", type=float, default=0.0)
    sample_parser.add_argument("--no-ema", action="store_true")
    sample_parser.add_argument("--grid-cols", type=int, default=4)
    sample_parser.add_argument("--individual", action="store_true")
    _add_common_device(sample_parser)

    inspect_parser = subparsers.add_parser(
        "inspect", help="show parameter count and schedule information"
    )
    inspect_parser.add_argument("--image-size", type=int, default=64)
    inspect_parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    _add_model_args(inspect_parser)
    _add_diffusion_args(inspect_parser)
    inspect_parser.add_argument("--device", default="cpu")

    smoke_parser = subparsers.add_parser(
        "smoke", help="tiny synthetic CPU run that exercises train and sample"
    )
    smoke_parser.add_argument("--output-dir", default="runs/smoke")
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--seed", type=int, default=0)
    smoke_parser.add_argument("--verbose", action="store_true")

    return parser


def _train_from_args(args: argparse.Namespace) -> dict:
    source = args.source
    if source is None:
        if args.data_dir:
            source = "imagefolder"
        elif args.hf_dataset:
            source = "hf"
        elif args.download:
            source = "cifar10"
        else:
            source = "synthetic"

    data = DataConfig(
        source=source,
        image_size=args.image_size,
        channels=args.channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_image_key=args.hf_image_key,
        cifar_root=args.cifar_root,
        download=args.download,
        synthetic_size=args.synthetic_size,
        augment=not args.no_augment,
        seed=args.seed,
    )
    model = ModelConfig(
        dim=args.dim,
        dim_mults=args.dim_mults,
        channels=args.channels,
        resnet_block_groups=args.groups,
        self_condition=args.self_condition,
    )
    diffusion = DiffusionConfig(
        timesteps=args.timesteps,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        loss_type=args.loss,
    )
    config = TrainConfig(
        data=data,
        model=model,
        diffusion=diffusion,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        ema_decay=0.0 if args.no_ema else args.ema_decay,
        log_every=args.log_every,
        sample_every=args.sample_every,
        sample_steps=args.sample_steps,
        sampler=args.sampler,
        num_sample_images=args.num_sample_images,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
    )
    from .train import train

    return train(config)


def _sample_from_args(args: argparse.Namespace) -> Path:
    config = SampleConfig(
        checkpoint=args.checkpoint,
        output=args.output,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        sampler=args.sampler,
        steps=args.steps,
        eta=args.eta,
        seed=args.seed,
        device=args.device,
        use_ema=not args.no_ema,
        grid_cols=args.grid_cols,
        individual=args.individual,
    )
    from .sample import generate

    return generate(config)


def _inspect_from_args(args: argparse.Namespace) -> int:
    from .diffusion import GaussianDiffusion
    from .model import build_unet

    model_config = ModelConfig(
        dim=args.dim,
        dim_mults=args.dim_mults,
        channels=args.channels,
        resnet_block_groups=args.groups,
        self_condition=args.self_condition,
    )
    diffusion_config = DiffusionConfig(
        timesteps=args.timesteps,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        loss_type=args.loss,
    )
    down = 2 ** max(0, len(model_config.dim_mults) - 1)
    if args.image_size % down != 0:
        raise SystemExit(
            f"image_size={args.image_size} must be divisible by {down} for dim_mults={model_config.dim_mults}"
        )

    model = build_unet(model_config)
    diffusion = GaussianDiffusion(diffusion_config)
    print(f"device: {resolve_device(args.device)}")
    print(f"parameters: {count_parameters(model):,}")
    print(
        f"image: {args.image_size}x{args.image_size}x{args.channels} "
        f"-> spatial / {down}"
    )
    print(f"dim: {model_config.dim} · dim_mults: {model_config.dim_mults}")
    print(f"timesteps: {diffusion.timesteps} · schedule: {diffusion_config.schedule}")
    print(f"betas: {float(diffusion.betas[0]):.6f} .. {float(diffusion.betas[-1]):.6f}")  # type: ignore
    print(
        "alphas_cumprod: "
        f"{float(diffusion.alphas_cumprod[0]):.6f} .. "  # type: ignore
        f"{float(diffusion.alphas_cumprod[-1]):.6f}"  # type: ignore
    )
    return 0


def _smoke(args: argparse.Namespace) -> int:
    from .sample import generate
    from .train import train

    run_dir = Path(args.output_dir)
    train_config = TrainConfig(
        data=DataConfig(
            source="synthetic",
            image_size=32,
            channels=3,
            batch_size=2,
            num_workers=0,
            synthetic_size=64,
            augment=False,
            seed=args.seed,
        ),
        model=ModelConfig(
            dim=16,
            dim_mults=(1, 2),
            channels=3,
            resnet_block_groups=4,
        ),
        diffusion=DiffusionConfig(timesteps=100, schedule="linear", loss_type="huber"),
        output_dir=str(run_dir),
        epochs=1,
        max_steps=2,
        lr=2e-4,
        ema_decay=0.9,
        log_every=1,
        sample_every=1,
        sample_steps=2,
        sampler="ddim",
        num_sample_images=2,
        device=args.device,
        seed=args.seed,
    )
    summary = train(train_config)
    sample_config = SampleConfig(
        checkpoint=summary["checkpoint"],
        output=str(run_dir / "smoke_sample.png"),
        num_samples=2,
        batch_size=2,
        sampler="ddim",
        steps=2,
        seed=args.seed,
        device=args.device,
        use_ema=True,
        grid_cols=2,
    )
    output = generate(sample_config)
    write_json(
        run_dir / "smoke.json",
        {"train": summary, "sample": str(output)},
    )
    logger.info("smoke test passed: %s", output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))

    try:
        if args.command == "train":
            summary = _train_from_args(args)
            logger.info("training finished: %s", summary["checkpoint"])
            return 0
        if args.command == "sample":
            output = _sample_from_args(args)
            logger.info("sampling finished: %s", output)
            return 0
        if args.command == "inspect":
            return _inspect_from_args(args)
        if args.command == "smoke":
            return _smoke(args)
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        if getattr(args, "verbose", False):
            logger.exception("details")
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
