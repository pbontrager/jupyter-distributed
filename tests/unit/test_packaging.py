from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from jupyter_distributed.timeout import parse_timeout


def test_interactive_timeout_parser() -> None:
    assert parse_timeout("24h") == timedelta(hours=24)
    assert parse_timeout(30) == timedelta(seconds=30)


def test_prebuilt_labextension_is_present() -> None:
    package = Path("jupyter_distributed/labextension/package.json")
    metadata = json.loads(package.read_text(encoding="utf-8"))
    remote_entry = metadata["jupyterlab"]["_build"]["load"]

    assert metadata["name"] == "jupyter-distributed"
    assert (package.parent / remote_entry).is_file()
