from __future__ import annotations

import json
from pathlib import Path


def test_prebuilt_labextension_is_present() -> None:
    package = Path("jupyter_distributed/labextension/package.json")
    metadata = json.loads(package.read_text(encoding="utf-8"))
    remote_entry = metadata["jupyterlab"]["_build"]["load"]

    assert metadata["name"] == "jupyter-distributed"
    assert (package.parent / remote_entry).is_file()
