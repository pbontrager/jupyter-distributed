from __future__ import annotations

import json
from pathlib import Path

import jupyter_distributed


def test_notebook_toolbar_schema_matches_frontend_plugin_id() -> None:
    extension = Path(jupyter_distributed.__file__).parent / "labextension"
    schema_dir = extension / "schemas" / "jupyter-distributed"
    schema = json.loads((schema_dir / "notebook.json").read_text(encoding="utf-8"))

    assert not (schema_dir / "plugin.json").exists()
    assert schema["jupyter.lab.toolbars"]["Notebook"] == [
        {"name": "jupyter-distributed-processes", "rank": 1001}
    ]
