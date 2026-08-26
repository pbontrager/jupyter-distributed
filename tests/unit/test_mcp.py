from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from jupyter_distributed import mcp
from jupyter_distributed.kernel_proxy import RANK_MIME


@dataclass
class FakeCoordinator:
    def describe(self, kernel_id: str) -> dict[str, Any]:
        assert kernel_id == "kernel-id"
        return {
            "kernel_id": kernel_id,
            "kernel_name": "python3",
            "world_size": 4,
            "distributed": True,
        }


class FakeSessionManager:
    async def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "path": "work/demo.ipynb",
                "kernel": {"id": "kernel-id", "name": "python3"},
            }
        ]


class FakeWebApp:
    settings = {"jupyter_distributed_coordinator": FakeCoordinator()}


@dataclass
class FakeServer:
    root_dir: str
    session_manager: FakeSessionManager = FakeSessionManager()
    web_app: FakeWebApp = FakeWebApp()


@pytest.mark.asyncio
async def test_reports_live_distributed_notebook_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = tmp_path / "work" / "demo.ipynb"
    notebook.parent.mkdir()
    notebook.write_text('{"metadata": {}}', encoding="utf-8")
    server = FakeServer(str(tmp_path))
    monkeypatch.setattr(mcp, "_server_app", lambda: server)

    result = await mcp.get_distributed_notebook_info("work/demo.ipynb")

    assert result == {
        "notebook_path": "work/demo.ipynb",
        "running": True,
        "kernel_id": "kernel-id",
        "kernel_name": "python3",
        "world_size": 4,
        "distributed": True,
        "world_size_source": "live_kernel",
    }


@pytest.mark.asyncio
async def test_reads_compact_outputs_for_every_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = tmp_path / "demo.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "metadata": {"jupyter_distributed": {"world_size": 2}},
                "cells": [
                    {
                        "id": "cell-id",
                        "cell_type": "code",
                        "source": "value",
                        "outputs": [
                            {
                                "output_type": "display_data",
                                "data": {
                                    RANK_MIME: {
                                        "execution_id": "execution-id",
                                        "execution_count": 1,
                                        "status": "error",
                                        "world_size": 2,
                                        "ranks": [
                                            {
                                                "rank": 0,
                                                "status": "ok",
                                                "outputs": [
                                                    {
                                                        "type": "execute_result",
                                                        "content": {
                                                            "data": {
                                                                "text/plain": "tensor(1)",
                                                                "text/html": "<b>1</b>",
                                                            }
                                                        },
                                                    }
                                                ],
                                            },
                                            {
                                                "rank": 1,
                                                "status": "error",
                                                "outputs": [
                                                    {
                                                        "type": "error",
                                                        "content": {
                                                            "ename": "ValueError",
                                                            "evalue": "bad rank",
                                                            "traceback": ["trace"],
                                                        },
                                                    }
                                                ],
                                            },
                                        ],
                                    },
                                    "text/plain": "rank fallback",
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))

    result = await mcp.read_distributed_cell_outputs(str(notebook), "cell-id")

    execution = result["executions"][0]
    assert execution["world_size"] == 2
    assert execution["ranks"][0]["outputs"][0] == {
        "type": "execute_result",
        "text": "tensor(1)",
        "mime_types": ["text/html", "text/plain"],
    }
    assert execution["ranks"][1]["outputs"][0]["evalue"] == "bad rank"


@pytest.mark.asyncio
async def test_selects_debug_rank_through_jupyterlab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, int]]] = []

    async def execute_command(command: str, arguments: dict[str, int]) -> dict[str, Any]:
        calls.append((command, arguments))
        return {
            "success": True,
            "result": {"rank": 2, "availableRanks": [0, 1, 2, 3]},
        }

    import sys
    import types

    package = types.ModuleType("jupyterlab_commands_toolkit")
    tools = types.ModuleType("jupyterlab_commands_toolkit.tools")
    tools.execute_command = execute_command  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jupyterlab_commands_toolkit", package)
    monkeypatch.setitem(sys.modules, "jupyterlab_commands_toolkit.tools", tools)

    result = await mcp.select_distributed_debug_rank(2)

    assert calls == [("jupyter-distributed:select-debug-rank", {"rank": 2})]
    assert result == {"rank": 2, "availableRanks": [0, 1, 2, 3]}


def test_mcp_entrypoint_is_declared() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '[project.entry-points."jupyter_server_mcp.tools"]' in project
    assert 'jupyter_distributed = "jupyter_distributed.mcp:TOOLS"' in project
