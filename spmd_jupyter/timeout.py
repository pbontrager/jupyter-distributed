"""Interactive-friendly PyTorch process-group initialization."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$", re.IGNORECASE)
_SECONDS_PER_UNIT = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
}


def parse_timeout(value: timedelta | str | int | float) -> timedelta:
    """Convert a short duration value into ``datetime.timedelta``.

    Strings use one unit suffix: ``s`` (seconds), ``m`` (minutes), ``h``
    (hours), or ``d`` (days). Numeric values are interpreted as seconds.
    """

    if isinstance(value, timedelta):
        result = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = timedelta(seconds=value)
    elif isinstance(value, str):
        match = _DURATION.fullmatch(value)
        if match is None:
            raise ValueError("timeout must look like '30s', '15m', '24h', or '2d'")
        amount, unit = match.groups()
        result = timedelta(seconds=float(amount) * _SECONDS_PER_UNIT[unit.lower()])
    else:
        raise TypeError("timeout must be a timedelta, duration string, or number of seconds")

    if result <= timedelta(0):
        raise ValueError("timeout must be greater than zero")
    return result


def init_process_group(
    *args: Any,
    timeout: timedelta | str | int | float = "24h",
    **kwargs: Any,
) -> Any:
    """Call ``torch.distributed.init_process_group`` with a parsed timeout.

    PyTorch is imported only when this helper is called, keeping it optional for
    users who only need the Jupyter integration. Install it with
    ``uv sync --extra distributed`` when it is not already available.
    """

    try:
        import torch.distributed as dist
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "PyTorch is required for init_process_group; run "
            "`uv sync --extra distributed` or install a compatible CUDA build"
        ) from error

    kwargs["timeout"] = parse_timeout(timeout)
    return dist.init_process_group(*args, **kwargs)


__all__ = ["init_process_group", "parse_timeout"]
