"""Optional MCP tools for agents working with Jupyter Distributed notebooks."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jupyter_server.serverapp import ServerApp

from .kernel_proxy import RANK_MIME

TOOLS = [
    "jupyter_distributed.mcp:get_distributed_notebook_info",
    "jupyter_distributed.mcp:read_distributed_cell_outputs",
    "jupyter_distributed.mcp:select_distributed_debug_rank",
]


async def get_distributed_notebook_info(notebook_path: str | None = None) -> dict[str, Any]:
    """Use this first when working with a Jupyter Distributed notebook.

    The selected kernel may run as N persistent kernels using SPMD: every code
    cell runs on every rank with independent rank-local state. This tool reports
    N. Use read_distributed_cell_outputs to inspect every rank without switching
    output tabs. Start a cell with %%rank N to run analysis code only on rank N.
    While debugging in JupyterLab, use select_distributed_debug_rank to choose
    the rank shown by Variables, Call Stack, and the Debug Console.
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

    world_size = _saved_world_size(path)
    return {
        "notebook_path": _relative_notebook_path(server, path),
        "running": session is not None,
        "world_size": world_size,
        "distributed": world_size > 1,
        "world_size_source": "notebook_metadata" if world_size > 1 else "default",
    }


async def read_distributed_cell_outputs(
    notebook_path: str,
    cell_id: str,
) -> dict[str, Any]:
    """Read every rank's output for one Jupyter Distributed notebook cell.

    Output tabs are only a visual selector; do not switch them to inspect ranks.
    This tool returns the saved structured outputs for all ranks, including each
    rank's status, text, errors, and available rich MIME types.
    """
    path = await _notebook_path(notebook_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    cell = next((item for item in document.get("cells", []) if item.get("id") == cell_id), None)
    if cell is None:
        raise LookupError(f"No cell found with cell_id={cell_id!r}")

    executions = []
    for output in cell.get("outputs", []):
        data = output.get("data")
        if not isinstance(data, Mapping):
            continue
        payload = data.get(RANK_MIME)
        if not isinstance(payload, Mapping):
            continue
        executions.append(_compact_execution(payload))

    return {
        "notebook_path": _relative_notebook_path(_server_app(), path),
        "cell_id": cell_id,
        "executions": executions,
    }


async def select_distributed_debug_rank(rank: int) -> dict[str, Any]:
    """Select which stopped rank JupyterLab's debugger displays.

    Use this only after the debugger is attached and the distributed program is
    stopped. The chosen rank supplies Variables, Call Stack, and Debug Console
    evaluation. Continue and step commands still apply to every rank.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")

    try:
        from jupyterlab_commands_toolkit.tools import execute_command
    except ImportError as error:
        raise RuntimeError("Selecting a debug rank requires Jupyter AI in JupyterLab") from error

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
        try:
            from jupyter_ai_tools.toolkits.notebook import get_active_notebook
        except ImportError as error:
            raise RuntimeError("notebook_path is required outside Jupyter AI") from error
        notebook_path = await get_active_notebook()
        if notebook_path is None:
            raise RuntimeError("No active notebook was found")

    path = Path(notebook_path)
    if not path.is_absolute():
        path = Path(_server_app().root_dir) / path
    return path.resolve()


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


def _saved_world_size(path: Path) -> int:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1
    metadata = document.get("metadata", {}).get("jupyter_distributed", {})
    world_size = metadata.get("world_size") if isinstance(metadata, Mapping) else None
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
    "get_distributed_notebook_info",
    "read_distributed_cell_outputs",
    "select_distributed_debug_rank",
]
