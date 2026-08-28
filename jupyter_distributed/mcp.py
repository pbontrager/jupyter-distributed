"""Optional MCP tools for agents working with distributed notebooks."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jupyter_server.serverapp import ServerApp

from .kernel_proxy import RANK_MIME

# jupyter-server-mcp discovers these functions when both optional integrations
# are installed. The Jupyter AI tools provide general live-notebook operations;
# this package adds the rank-aware operations specific to distributed execution.
TOOLS = [
    "jupyter_ai_tools.toolkits.notebook:read_notebook",
    "jupyter_ai_tools.toolkits.notebook:read_notebook_cells",
    "jupyter_ai_tools.toolkits.notebook:read_cell",
    "jupyter_ai_tools.toolkits.notebook:add_cell",
    "jupyter_ai_tools.toolkits.notebook:insert_cell",
    "jupyter_ai_tools.toolkits.notebook:delete_cell",
    "jupyter_ai_tools.toolkits.notebook:edit_cell",
    "jupyter_ai_tools.toolkits.notebook:select_cell",
    "jupyter_ai_tools.toolkits.notebook:get_cell_id_from_index",
    "jupyter_ai_tools.toolkits.notebook:get_active_notebook",
    "jupyter_ai_tools.toolkits.notebook:get_active_cell_id",
    "jupyter_ai_tools.toolkits.notebook:create_notebook",
    "jupyter_ai_tools.toolkits.jupyterlab:open_file",
    "jupyter_ai_tools.toolkits.jupyterlab:run_cell",
    "jupyter_ai_tools.toolkits.jupyterlab:run_all_cells",
    "jupyter_distributed.mcp:get_distributed_notebook_info",
    "jupyter_distributed.mcp:read_distributed_cell_outputs",
    "jupyter_distributed.mcp:get_selected_notebook_cell",
    "jupyter_distributed.mcp:append_execute_distributed_cell",
    "jupyter_distributed.mcp:select_distributed_debug_rank",
]


async def get_distributed_notebook_info(
    notebook_path: str | None = None,
) -> dict[str, Any]:
    """Inspect the process model for a Jupyter Distributed notebook.

    Call this before reasoning about execution. With more than one process,
    ordinary cells run on every persistent rank using SPMD semantics and each
    rank has independent state. Use ``%%rank N`` for rank-local analysis and
    ``read_distributed_cell_outputs`` to inspect every rank.
    """
    path = await _notebook_path(notebook_path)
    server = _server_app()
    session = await _find_session(server, path)

    if session is not None:
        kernel = session.get("kernel")
        if not isinstance(kernel, Mapping) or not isinstance(kernel.get("id"), str):
            raise RuntimeError(f"Notebook session has no kernel: {path}")
        kernel_id = kernel["id"]
        coordinator = server.web_app.settings.get("jupyter_distributed_coordinator")
        if coordinator is not None:
            model = coordinator.describe(kernel_id)
            return {
                "notebook_path": _relative_notebook_path(server, path),
                "running": True,
                "kernel_id": kernel_id,
                "kernel_name": model["kernel_name"],
                "world_size": model["world_size"],
                "distributed": model["distributed"],
                "world_size_source": "live_kernel",
            }

    document = await _read_notebook(path)
    world_size = _saved_world_size(document)
    return {
        "notebook_path": _relative_notebook_path(server, path),
        "running": session is not None,
        "world_size": world_size,
        "distributed": world_size > 1,
        "world_size_source": "notebook_metadata" if world_size > 1 else "default",
    }


async def read_distributed_cell_outputs(
    cell_id: str,
    notebook_path: str | None = None,
) -> dict[str, Any]:
    """Read every rank's output for one cell in the live notebook.

    Rank tabs and dropdowns only change the browser view. This returns the
    structured output behind them, including text, errors, and available rich
    MIME types for every rank.
    """
    path = await _notebook_path(notebook_path)
    document = await _read_notebook(path)
    cells = document.get("cells", [])
    cell = next(
        (item for item in cells if isinstance(item, Mapping) and item.get("id") == cell_id),
        None,
    )
    if cell is None:
        raise LookupError(f"No cell found with cell_id={cell_id!r}")

    executions = []
    for output in cell.get("outputs", []):
        if not isinstance(output, Mapping):
            continue
        data = output.get("data")
        if not isinstance(data, Mapping):
            continue
        payload = data.get(RANK_MIME)
        if isinstance(payload, Mapping):
            executions.append(_compact_execution(payload))

    return {
        "notebook_path": _relative_notebook_path(_server_app(), path),
        "cell_id": cell_id,
        "executions": executions,
    }


async def get_selected_notebook_cell() -> dict[str, Any]:
    """Return the cell currently selected in the open JupyterLab notebook."""
    from jupyter_ai_tools.toolkits.notebook import get_active_cell_id, read_cell_json

    path = await _notebook_path(None)
    notebook_path = _relative_notebook_path(_server_app(), path)
    cell_id = await get_active_cell_id(notebook_path)
    if cell_id is None:
        raise RuntimeError("No selected notebook cell was found")
    cell, cell_index = await read_cell_json(notebook_path, cell_id)
    return {
        "notebook_path": notebook_path,
        "cell_id": cell_id,
        "cell_index": cell_index,
        "cell": cell,
    }


async def append_execute_distributed_cell(source: str) -> dict[str, Any]:
    """Append and execute a code cell through the open notebook frontend.

    Ordinary source runs on every rank. Prefix the source with ``%%rank N`` to
    target one rank. Execution follows the notebook's normal frontend path so
    live rank outputs, widgets, and existing kernel state are preserved.
    """
    from jupyter_ai_tools.toolkits.jupyterlab import run_cell
    from jupyter_ai_tools.toolkits.notebook import add_cell, read_notebook_json

    path = await _notebook_path(None)
    notebook_path = _relative_notebook_path(_server_app(), path)
    await add_cell(notebook_path, content=source, cell_type="code")
    document = await read_notebook_json(notebook_path)
    cells = document.get("cells", [])
    if not cells or not isinstance(cells[-1], Mapping):
        raise RuntimeError("The appended notebook cell could not be found")
    cell_id = cells[-1].get("id")
    if not isinstance(cell_id, str):
        raise RuntimeError("The appended notebook cell has no id")
    result = await run_cell(cell_id, file_path=notebook_path)
    return {
        "notebook_path": notebook_path,
        "cell_id": cell_id,
        "cell_index": len(cells) - 1,
        "execution": result,
    }


async def select_distributed_debug_rank(rank: int) -> dict[str, Any]:
    """Select which stopped rank JupyterLab's debugger displays.

    Use this after the debugger is attached and execution is paused. Variables,
    Call Stack, and Debug Console evaluation switch to the chosen rank;
    continue and step commands still apply to every rank together.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")

    from jupyterlab_commands_toolkit.tools import execute_command

    response = await execute_command(
        "jupyter-distributed:select-debug-rank",
        {"rank": rank},
    )
    if not response.get("success"):
        raise RuntimeError(str(response.get("error", "Unable to select debugger rank")))
    result = response.get("result")
    return result if isinstance(result, dict) else {"rank": rank}


def _server_app() -> ServerApp:
    return ServerApp.instance()


async def _notebook_path(notebook_path: str | None) -> Path:
    if notebook_path is None:
        from jupyter_ai_tools.toolkits.notebook import get_active_notebook

        notebook_path = await get_active_notebook()
        if notebook_path is None:
            raise RuntimeError("No active notebook was found")

    path = Path(notebook_path)
    if not path.is_absolute():
        path = Path(_server_app().root_dir) / path
    return path.resolve()


async def _read_notebook(path: Path) -> Mapping[str, Any]:
    from jupyter_ai_tools.toolkits.notebook import read_notebook_json

    notebook_path = _relative_notebook_path(_server_app(), path)
    document = await read_notebook_json(notebook_path)
    if not isinstance(document, Mapping):
        raise TypeError(f"Notebook did not contain an object: {notebook_path}")
    return document


async def _find_session(server: ServerApp, path: Path) -> Mapping[str, Any] | None:
    sessions = server.session_manager.list_sessions()
    if inspect.isawaitable(sessions):
        sessions = await sessions
    relative_path = _relative_notebook_path(server, path)
    for session in sessions:
        session_path = session.get("path")
        if session_path is None and isinstance(session.get("notebook"), Mapping):
            session_path = session["notebook"].get("path")
        if isinstance(session_path, str) and session_path.lstrip("/") == relative_path:
            return session
    return None


def _relative_notebook_path(server: ServerApp, path: Path) -> str:
    try:
        return path.relative_to(Path(server.root_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def _saved_world_size(document: Mapping[str, Any]) -> int:
    metadata = document.get("metadata", {})
    distributed = metadata.get("jupyter_distributed", {}) if isinstance(metadata, Mapping) else {}
    world_size = distributed.get("world_size") if isinstance(distributed, Mapping) else None
    if isinstance(world_size, int) and not isinstance(world_size, bool) and world_size > 0:
        return world_size
    return 1


def _compact_execution(payload: Mapping[str, Any]) -> dict[str, Any]:
    ranks = payload.get("ranks", [])
    return {
        "execution_id": payload.get("execution_id"),
        "execution_count": payload.get("execution_count"),
        "status": payload.get("status"),
        "world_size": payload.get("world_size"),
        "ranks": [
            {
                "rank": rank.get("rank"),
                "status": rank.get("status"),
                "outputs": [_compact_output(output) for output in rank.get("outputs", [])],
            }
            for rank in ranks
            if isinstance(rank, Mapping)
        ],
    }


def _compact_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        return {"type": "unknown", "text": str(output)}
    output_type = output.get("type", output.get("output_type"))
    content = output.get("content", output)
    if not isinstance(content, Mapping):
        return {"type": output_type, "text": str(content)}
    if output_type == "stream":
        return {
            "type": "stream",
            "name": content.get("name"),
            "text": _text(content.get("text")),
        }
    if output_type == "error":
        return {
            "type": "error",
            "ename": content.get("ename"),
            "evalue": content.get("evalue"),
            "traceback": content.get("traceback", []),
        }
    data = content.get("data", {})
    if not isinstance(data, Mapping):
        return {"type": output_type, "text": str(data)}
    return {
        "type": output_type,
        "text": _text(data.get("text/plain")),
        "mime_types": sorted(str(mime_type) for mime_type in data),
    }


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


__all__ = [
    "TOOLS",
    "append_execute_distributed_cell",
    "get_distributed_notebook_info",
    "get_selected_notebook_cell",
    "read_distributed_cell_outputs",
    "select_distributed_debug_rank",
]
