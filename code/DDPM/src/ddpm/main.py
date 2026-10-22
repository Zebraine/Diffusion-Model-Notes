"""Backward-compatible module entry point.

The implementation now lives in focused modules; this shim keeps
``python -m dppm.main`` and ``from dppm.main import main`` working.
"""

from .cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
