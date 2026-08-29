from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
            "proxied": True,
        }


@dataclass
class FakeServer:
    root_dir: str

    def __post_init__(self) -> None:
        self.web_app = SimpleNamespace(
            settings={"jupyter_distributed_coordinator": FakeCoordinator()}
        )


async def _async_value(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_reports_live_distributed_notebook_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FakeServer(str(tmp_path))

    async def find_session(*_args: Any) -> dict[str, Any]:
        return {"kernel": {"id": "kernel-id"}}

    monkeypatch.setattr(mcp, "_server_app", lambda: server)
    monkeypatch.setattr(mcp, "_find_session", find_session)

    result = await mcp.get_distributed_notebook_info("work/demo.ipynb")

    assert result == {
        "notebook_path": "work/demo.ipynb",
        "running": True,
        "kernel_id": "kernel-id",
        "kernel_name": "python3",
        "world_size": 4,
        "distributed": True,
        "world_size_source": "live_kernel",
        "process_control": {
            "change_tool": "set_distributed_processes",
            "restarts_kernel": True,
            "loses_in_memory_state": True,
            "do_not_edit_notebook_metadata": True,
            "do_not_use_generic_kernel_restart": True,
        },
        "cell_workflow": {
            "tool_preference": (
                "Prefer dedicated MCP tools. Use list_all_commands and execute_command only "
                "when no dedicated MCP tool covers the operation."
            ),
            "new_bottom_cell_tool": "append_cell",
            "reuses_first_trailing_blank_cell": True,
            "run_existing_cell_tool": "run_cell",
            "retry_in_place": ["edit_cell", "run_cell"],
            "do_not_append_replacement_cells_when_debugging": True,
            "superseded_raw_commands": [
                "jupyterlab-ai-commands:add-cell",
                "jupyterlab-ai-commands:run-cell",
            ],
        },
        "framework_environment": {
            "active": True,
            "provided_by_managed_kernel": {
                "pytorch": [
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                ],
                "jax": [
                    "JAX_COORDINATOR_ADDRESS",
                    "JAX_PROCESS_ID",
                    "JAX_NUM_PROCESSES",
                ],
            },
            "guidance": (
                "Do not set these variables in generated notebook code unless the user is "
                "intentionally overriding a default before framework initialization. "
                "Initialize PyTorch with torch.distributed.init_process_group(), then use "
                "torch.distributed.get_rank() and torch.distributed.get_world_size(). "
                "Initialize JAX with jax.distributed.initialize(), then use "
                "jax.process_index() and jax.process_count(). Backend selection, device "
                "placement, collectives, and data or model sharding remain the user's "
                "responsibility."
            ),
        },
    }


@pytest.mark.asyncio
async def test_reads_compact_outputs_for_every_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = {
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

    async def read_notebook(*_args: Any) -> dict[str, Any]:
        return document

    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))
    monkeypatch.setattr(mcp, "_read_notebook", read_notebook)

    result = await mcp.read_distributed_cell_outputs("cell-id", notebook_path="demo.ipynb")

    execution = result["executions"][0]
    assert result["cell_id"] == "cell-id"
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

    async def execute(command: str, args: dict[str, int]) -> dict[str, Any]:
        calls.append((command, args))
        return {
            "success": True,
            "result": {"rank": 2, "availableRanks": [0, 1, 2, 3]},
        }

    import jupyterlab_commands_toolkit.tools

    monkeypatch.setattr(jupyterlab_commands_toolkit.tools, "execute_command", execute)

    result = await mcp.select_distributed_debug_rank(2)

    assert calls == [("jupyter-distributed:select-debug-rank", {"rank": 2})]
    assert result == {"rank": 2, "availableRanks": [0, 1, 2, 3]}


@pytest.mark.asyncio
async def test_returns_selected_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def active_cell(_path: str) -> str:
        return "cell-id"

    async def read_cell(_path: str, cell_id: str) -> tuple[dict[str, Any], int]:
        assert cell_id == "cell-id"
        return {"id": cell_id, "source": "model"}, 3

    import jupyter_ai_tools.toolkits.notebook

    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))
    monkeypatch.setattr(mcp, "_notebook_path", lambda _path: _async_value(tmp_path / "demo.ipynb"))
    monkeypatch.setattr(jupyter_ai_tools.toolkits.notebook, "get_active_cell_id", active_cell)
    monkeypatch.setattr(jupyter_ai_tools.toolkits.notebook, "read_cell_json", read_cell)

    result = await mcp.get_selected_notebook_cell()

    assert result == {
        "notebook_path": "demo.ipynb",
        "cell_id": "cell-id",
        "cell_index": 3,
        "cell": {"id": "cell-id", "source": "model"},
    }


@pytest.mark.asyncio
async def test_appends_cell_when_notebook_ends_with_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []

    async def add_cell(path: str, *, content: str, cell_type: str) -> None:
        calls.append(("add", (path, content, cell_type)))

    async def read_notebook(_path: str) -> dict[str, Any]:
        return {"cells": [{"id": "new-cell", "source": "model(dummy)"}]}

    import jupyter_ai_tools.toolkits.notebook

    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))
    monkeypatch.setattr(mcp, "_notebook_path", lambda _path: _async_value(tmp_path / "demo.ipynb"))
    monkeypatch.setattr(jupyter_ai_tools.toolkits.notebook, "add_cell", add_cell)
    monkeypatch.setattr(jupyter_ai_tools.toolkits.notebook, "read_notebook_json", read_notebook)

    result = await mcp.append_cell("model(dummy)")

    assert result == {
        "notebook_path": "demo.ipynb",
        "cell_id": "new-cell",
        "cell_index": 0,
        "reused_blank_cell": False,
    }
    assert calls == [("add", ("demo.ipynb", "model(dummy)", "code"))]


@pytest.mark.asyncio
async def test_reuses_first_cell_in_trailing_blank_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []
    cells = [
        {"id": "0", "source": "first"},
        {"id": "1", "source": "second"},
        {"id": "2", "source": ""},
        {"id": "3", "source": "fourth"},
        {"id": "4", "source": ""},
        {"id": "5", "source": ["  ", "\n"]},
        {"id": "6", "source": ""},
    ]

    async def read_notebook(_path: str) -> dict[str, Any]:
        return {"cells": cells}

    async def add_cell(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a new cell should not be appended")

    async def edit_cell(
        path: str,
        cell_id: str,
        *,
        content: str,
        cell_type: str,
    ) -> None:
        calls.append(("edit", (path, cell_id, content, cell_type)))

    import jupyter_ai_tools.toolkits.notebook

    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))
    monkeypatch.setattr(mcp, "_notebook_path", lambda _path: _async_value(tmp_path / "demo.ipynb"))
    monkeypatch.setattr(jupyter_ai_tools.toolkits.notebook, "add_cell", add_cell)
    monkeypatch.setattr(jupyter_ai_tools.toolkits.notebook, "edit_cell", edit_cell)
    monkeypatch.setattr(jupyter_ai_tools.toolkits.notebook, "read_notebook_json", read_notebook)

    result = await mcp.append_cell("rank * world_size")

    assert result["cell_id"] == "4"
    assert result["cell_index"] == 4
    assert result["reused_blank_cell"] is True
    assert calls == [("edit", ("demo.ipynb", "4", "rank * world_size", "code"))]


@pytest.mark.asyncio
async def test_run_cell_returns_complete_rank_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions = [
        {"notebook_path": "demo.ipynb", "cell_id": "cell-id", "executions": []},
        {
            "notebook_path": "demo.ipynb",
            "cell_id": "cell-id",
            "executions": [
                {
                    "execution_id": "new",
                    "status": "ok",
                    "world_size": 2,
                    "ranks": [
                        {"rank": 0, "status": "ok", "outputs": [{"text": "0"}]},
                        {"rank": 1, "status": "ok", "outputs": [{"text": "2"}]},
                    ],
                }
            ],
        },
    ]

    async def info(_path: str) -> dict[str, Any]:
        return {"world_size": 2}

    async def outputs(*_args: Any) -> dict[str, Any]:
        return executions.pop(0)

    async def run_cell(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True, "result": {"success": True, "status": "ok"}}

    import jupyter_ai_tools.toolkits.jupyterlab

    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))
    monkeypatch.setattr(mcp, "_notebook_path", lambda _path: _async_value(tmp_path / "demo.ipynb"))
    monkeypatch.setattr(mcp, "get_distributed_notebook_info", info)
    monkeypatch.setattr(mcp, "read_distributed_cell_outputs", outputs)
    monkeypatch.setattr(jupyter_ai_tools.toolkits.jupyterlab, "run_cell", run_cell)

    result = await mcp.run_cell("cell-id", "demo.ipynb")

    assert result["success"] is True
    assert result["classification"] == "ok"
    assert result["rank_output_counts"] == {"0": 1, "1": 1}


@pytest.mark.asyncio
async def test_run_cell_classifies_incomplete_aggregation_as_extension_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def info(_path: str) -> dict[str, Any]:
        return {"world_size": 4}

    async def outputs(*_args: Any) -> dict[str, Any]:
        return {"executions": []}

    async def run_cell(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True, "result": {"success": True, "status": "ok"}}

    async def incomplete(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "execution_id": "new",
            "status": "busy",
            "world_size": 4,
            "ranks": [{"rank": 0, "status": "running", "outputs": [{"text": "0"}]}],
        }

    import jupyter_ai_tools.toolkits.jupyterlab

    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))
    monkeypatch.setattr(mcp, "_notebook_path", lambda _path: _async_value(tmp_path / "demo.ipynb"))
    monkeypatch.setattr(mcp, "get_distributed_notebook_info", info)
    monkeypatch.setattr(mcp, "read_distributed_cell_outputs", outputs)
    monkeypatch.setattr(mcp, "_wait_for_distributed_execution", incomplete)
    monkeypatch.setattr(jupyter_ai_tools.toolkits.jupyterlab, "run_cell", run_cell)

    result = await mcp.run_cell("cell-id", "demo.ipynb")

    assert result["success"] is False
    assert result["classification"] == "jupyter_distributed_error"
    assert "not evidence that the cell source is wrong" in result["message"]
    assert "do not append a replacement" in result["recommended_action"]


@pytest.mark.asyncio
async def test_sets_processes_through_live_notebook_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(command: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append((command, args))
        return {
            "success": True,
            "result": {
                "kernel_id": "kernel-id",
                "kernel_name": "python3",
                "world_size": 4,
                "distributed": True,
            },
        }

    import jupyterlab_commands_toolkit.tools

    monkeypatch.setattr(mcp, "_server_app", lambda: FakeServer(str(tmp_path)))
    monkeypatch.setattr(mcp, "_notebook_path", lambda _path: _async_value(tmp_path / "demo.ipynb"))
    monkeypatch.setattr(jupyterlab_commands_toolkit.tools, "execute_command", execute)

    result = await mcp.set_distributed_processes(4, "demo.ipynb")

    assert calls == [
        (
            "jupyter-distributed:set-processes",
            {"processes": 4, "notebookPath": "demo.ipynb"},
        )
    ]
    assert result["world_size"] == 4
    assert result["distributed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("processes", [True, 0, -1, 1.5])
async def test_rejects_invalid_mcp_process_counts(processes: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        await mcp.set_distributed_processes(processes)  # type: ignore[arg-type]


def test_jupyter_server_mcp_tools_and_optional_extra_are_declared() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '[project.entry-points."jupyter_server_mcp.tools"]' in project
    assert 'jupyter_distributed = "jupyter_distributed.mcp:TOOLS"' in project
    assert "jupyter-server-mcp>=0.2.2,<1" in project
    assert "jupyter-ai-tools>=0.7.0rc0,<1" in project
    assert "jupyterlab-ai-commands>=0.4.0rc0,<1" in project
    assert "jupyterlab-commands-toolkit>=0.1.6,<1" in project
    assert "jupyter-collaboration" not in project
    assert "jupyterlab-notebook-awareness" not in project
    assert "jupyter-mcp-server" not in project
    assert "jupyter-mcp-tools" not in project


def test_tool_list_contains_notebook_and_distributed_tools() -> None:
    assert "jupyter_ai_tools.toolkits.notebook:read_notebook" in mcp.TOOLS
    assert "jupyter_ai_tools.toolkits.notebook:edit_cell" in mcp.TOOLS
    assert "jupyter_ai_tools.toolkits.notebook:insert_cell" in mcp.TOOLS
    assert "jupyter_ai_tools.toolkits.notebook:add_cell" not in mcp.TOOLS
    assert "jupyter_ai_tools.toolkits.jupyterlab:run_cell" not in mcp.TOOLS
    assert "jupyter_distributed.mcp:get_distributed_notebook_info" in mcp.TOOLS
    assert "jupyter_distributed.mcp:read_distributed_cell_outputs" in mcp.TOOLS
    assert "jupyter_distributed.mcp:append_cell" in mcp.TOOLS
    assert "jupyter_distributed.mcp:run_cell" in mcp.TOOLS
    assert "jupyter_distributed.mcp:append_execute_distributed_cell" not in mcp.TOOLS
    assert "jupyter_distributed.mcp:run_distributed_cell" not in mcp.TOOLS
    assert "jupyter_distributed.mcp:set_distributed_processes" in mcp.TOOLS
    assert "jupyter_distributed.mcp:select_distributed_debug_rank" in mcp.TOOLS
