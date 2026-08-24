"""Install and describe the logical SPMD Python kernelspec."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager

KERNEL_NAME = "spmd-python"
DISPLAY_NAME = "SPMD Python"


def kernel_spec() -> dict[str, object]:
    """Return the portable kernel specification installed by this package."""

    return {
        "argv": [
            sys.executable,
            "-m",
            "spmd_jupyter.kernel",
            "-f",
            "{connection_file}",
        ],
        "display_name": DISPLAY_NAME,
        "language": "python",
        "metadata": {
            "debugger": True,
            "spmd_jupyter": {"world_size": 1},
        },
    }


def install_kernel_spec(*, user: bool = False, prefix: str | None = None) -> str:
    """Install ``spmd-python`` and return its destination directory."""

    if user and prefix:
        raise ValueError("user and prefix are mutually exclusive")

    with tempfile.TemporaryDirectory(prefix="spmd-jupyter-kernelspec-") as tmp:
        source = Path(tmp) / KERNEL_NAME
        source.mkdir()
        (source / "kernel.json").write_text(
            json.dumps(kernel_spec(), indent=2) + "\n", encoding="utf-8"
        )
        destination = KernelSpecManager().install_kernel_spec(
            str(source),
            kernel_name=KERNEL_NAME,
            user=user,
            prefix=prefix,
            replace=True,
        )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the SPMD Python kernelspec")
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--user", action="store_true", help="install for the current user")
    location.add_argument("--prefix", help="install under this environment prefix")
    args = parser.parse_args(argv)
    destination = install_kernel_spec(user=args.user, prefix=args.prefix)
    print(f"Installed {KERNEL_NAME} kernelspec in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
