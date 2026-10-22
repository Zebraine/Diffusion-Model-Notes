"""Datasets, transforms and DataLoader construction.

The synthetic source is deliberately dependency-light so the full pipeline can
be exercised on CPU without downloading a dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import DataConfig


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


class SyntheticShapes(Dataset):
    """Procedural colored shapes in ``[-1, 1]`` for smoke tests and demos."""

    def __init__(
        self,
        size: int = 512,
        image_size: int = 64,
        channels: int = 3,
        seed: int = 0,
    ) -> None:
        if image_size < 8:
            raise ValueError("image_size must be >= 8")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        self.size = int(size)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + int(index))
        size = self.image_size
        channels = self.channels

        y = torch.linspace(-1.0, 1.0, size).view(size, 1)
        x = torch.linspace(-1.0, 1.0, size).view(1, size)
        background = torch.rand(channels, 1, 1, generator=generator) * 2.0 - 1.0
        image = background + 0.16 * x + 0.12 * y

        # One large soft-edged circle and one smaller square.
        cx = _uniform(generator, -0.45, 0.45)
        cy = _uniform(generator, -0.45, 0.45)
        radius = _uniform(generator, 0.22, 0.42)
        circle = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).to(image.dtype)
        circle_color = torch.rand(channels, 1, 1, generator=generator) * 2.0 - 1.0
        image = image * (1.0 - circle) + circle_color * circle

        sx = _uniform(generator, -0.7, 0.5)
        sy = _uniform(generator, -0.7, 0.5)
        side = _uniform(generator, 0.12, 0.26)
        square = ((x >= sx) & (x <= sx + side) & (y >= sy) & (y <= sy + side)).to(
            image.dtype
        )
        square_color = torch.rand(channels, 1, 1, generator=generator) * 2.0 - 1.0
        image = image * (1.0 - square) + square_color * square

        image = image + 0.04 * torch.randn(channels, size, size, generator=generator)
        return image.clamp(-1.0, 1.0)


class ImageOnly(Dataset):
    """Drop class labels from ImageFolder/CIFAR10 items."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        item = self.dataset[index]
        return item[0] if isinstance(item, (tuple, list)) else item


class HuggingFaceImageDataset(Dataset):
    """A thin adapter around a HuggingFace ``datasets`` split."""

    def __init__(
        self,
        name: str,
        split: str,
        image_key: str,
        transform: transforms.Compose,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "The 'datasets' package is required for --source hf"
            ) from error

        self.dataset = load_dataset(name, split=split)
        self.image_key = image_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tensor:
        item = self.dataset[index]
        if self.image_key not in item:
            available = ", ".join(sorted(item))
            raise KeyError(
                f"image key {self.image_key!r} not found; available columns: {available}"
            )
        image = item[self.image_key]
        if image is None:
            raise ValueError(f"dataset item {index} has no image at {self.image_key!r}")
        image = image.convert("RGB")
        return self.transform(image)


def build_transform(config: DataConfig, training: bool = True) -> transforms.Compose:
    operations: list[Any] = [
        transforms.Resize(config.image_size, antialias=True),
        transforms.CenterCrop(config.image_size),
    ]
    if training and config.augment:
        operations.insert(1, transforms.RandomHorizontalFlip())
    if config.channels == 1:
        operations.append(transforms.Grayscale(num_output_channels=1))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * config.channels, std=[0.5] * config.channels
            ),
        ]
    )
    return transforms.Compose(operations)


def build_dataset(config: DataConfig) -> Dataset:
    source = config.source.lower()
    transform = build_transform(config, training=True)

    if source == "synthetic":
        return SyntheticShapes(
            size=config.synthetic_size,
            image_size=config.image_size,
            channels=config.channels,
            seed=config.seed,
        )

    if source == "imagefolder":
        if not config.data_dir:
            raise ValueError("--data-dir is required when --source imagefolder")
        root = Path(config.data_dir)
        if not root.exists():
            raise FileNotFoundError(f"image folder does not exist: {root}")
        return ImageOnly(datasets.ImageFolder(root=str(root), transform=transform))

    if source == "cifar10":
        return ImageOnly(
            datasets.CIFAR10(
                root=config.cifar_root,
                train=True,
                download=config.download,
                transform=transform,
            )
        )

    if source == "hf":
        if not config.hf_dataset:
            raise ValueError("--hf-dataset is required when --source hf")
        return HuggingFaceImageDataset(
            name=config.hf_dataset,
            split=config.hf_split,
            image_key=config.hf_image_key,
            transform=transform,
        )

    raise ValueError(f"unknown data source: {config.source}")


def build_dataloader(config: DataConfig) -> DataLoader:
    dataset = build_dataset(config)
    if len(dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("dataset is empty")
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def denormalize(images: Tensor) -> Tensor:
    """Map images from ``[-1, 1]`` to ``[0, 1]`` for saving/visualization."""

    return (images.clamp(-1.0, 1.0) + 1.0) / 2.0
