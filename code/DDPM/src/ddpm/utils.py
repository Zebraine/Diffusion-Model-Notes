"""Small shared helpers: reproducibility, devices, logging and JSON files."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, TypeGuard

import torch


def exists[T](value: T | None) -> TypeGuard[T]:
    return value is not None


def default[T](value: T | None, fallback: T | Any) -> T | Any:
    """Return value when it exists, otherwise call/return fallback."""

    if exists(value):
        return value
    return fallback() if callable(fallback) else fallback


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is optional for the core path
        pass


def resolve_device(name: str = "cpu") -> torch.device:
    name = (name or "cpu").strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if name == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{seconds:.1f}s"


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


class AverageMeter:
    """Track a running average for loss logging."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
