"""Persistent SPMD execution for Jupyter notebooks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jupyter-distributed")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.1"


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    """Advertise the bundled Jupyter Server extension."""

    return [{"module": "jupyter_distributed.server_extension"}]


def _jupyter_labextension_paths() -> list[dict[str, str]]:
    """Advertise the source extension for editable development installs."""

    return [{"src": "labextension", "dest": "jupyter-distributed"}]


__all__ = ["__version__"]
