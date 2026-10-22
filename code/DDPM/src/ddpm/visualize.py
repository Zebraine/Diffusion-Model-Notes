"""Image-grid saving without opening a GUI or relying on matplotlib."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torchvision.utils import make_grid, save_image

from .utils import ensure_dir


def save_image_grid(
    images: Tensor,
    path: str | Path,
    nrow: int = 4,
    padding: int = 2,
) -> Path:
    """Save a ``(B, C, H, W)`` batch in ``[-1, 1]`` as one PNG grid."""

    destination = Path(path)
    ensure_dir(destination.parent)
    images = images.detach().cpu()
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    grid = make_grid(images, nrow=max(1, nrow), padding=padding)
    save_image(grid, str(destination))
    return destination


def save_individual_images(
    images: Tensor,
    directory: str | Path,
    prefix: str = "sample",
    padding: int = 0,
) -> list[Path]:
    destination = ensure_dir(directory)
    paths: list[Path] = []
    for index, image in enumerate(images.detach().cpu()):
        path = destination / f"{prefix}_{index:03d}.png"
        save_image(
            (image.clamp(-1.0, 1.0) + 1.0) / 2.0,
            str(path),
            padding=padding,
        )
        paths.append(path)
    return paths
