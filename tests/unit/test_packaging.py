from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from jupyter_distributed.kernelspec import kernel_spec
from jupyter_distributed.timeout import parse_timeout


def test_static_and_generated_kernelspecs_use_message_interrupts() -> None:
    generated = kernel_spec()
    static_path = Path("jupyter-config/kernels/jupyter-distributed/kernel.json")
    static = json.loads(static_path.read_text(encoding="utf-8"))

    assert generated["interrupt_mode"] == "message"
    assert static["interrupt_mode"] == "message"
    assert generated["metadata"]["debugger"] is False
    assert static["metadata"]["debugger"] is False
    assert generated["argv"][1:3] == ["-m", "jupyter_distributed.kernel"]
    assert static["argv"][1:3] == ["-m", "jupyter_distributed.kernel"]


def test_interactive_timeout_parser() -> None:
    assert parse_timeout("24h") == timedelta(hours=24)
    assert parse_timeout(30) == timedelta(seconds=30)


def test_prebuilt_labextension_is_present() -> None:
    package = Path("jupyter_distributed/labextension/package.json")
    metadata = json.loads(package.read_text(encoding="utf-8"))
    remote_entry = metadata["jupyterlab"]["_build"]["load"]

    assert metadata["name"] == "jupyter-distributed"
    assert (package.parent / remote_entry).is_file()
