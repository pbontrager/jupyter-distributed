"""Persistent SPMD execution for Jupyter notebooks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("jupyter-distributed")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.0"


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    """Advertise the bundled Jupyter Server extension."""

    return [{"module": "jupyter_distributed.server_extension"}]


def init_process_group(*args: Any, timeout: str | int | float = "24h", **kwargs: Any) -> Any:
    """Initialize PyTorch distributed with an interactive-friendly timeout.

    This is a lazy import so importing :mod:`jupyter_distributed` does not
    initialize PyTorch or CUDA. Explicit ``datetime.timedelta`` values can be
    passed via ``timeout`` unchanged; strings such as ``"24h"`` are parsed by
    the runtime helper.
    """

    from .timeout import init_process_group as _init_process_group

    return _init_process_group(*args, timeout=timeout, **kwargs)


__all__ = ["__version__", "init_process_group"]
