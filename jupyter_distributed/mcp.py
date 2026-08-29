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
    "jupyter_ai_tools.toolkits.notebook:insert_cell",
    "jupyter_ai_tools.toolkits.notebook:delete_cell",
    "jupyter_ai_tools.toolkits.notebook:edit_cell",
    "jupyter_ai_tools.toolkits.notebook:select_cell",
    "jupyter_ai_tools.toolkits.notebook:get_cell_id_from_index",
    "jupyter_ai_tools.toolkits.notebook:get_active_notebook",
    "jupyter_ai_tools.toolkits.notebook:get_active_cell_id",
    "jupyter_ai_tools.toolkits.notebook:create_notebook",
    "jupyter_ai_tools.toolkits.jupyterlab:open_file",
    "jupyter_ai_tools.toolkits.jupyterlab:run_all_cells",
    "jupyter_distributed.mcp:get_distributed_notebook_info",
    "jupyter_distributed.mcp:read_distributed_cell_outputs",
    "jupyter_distributed.mcp:get_selected_notebook_cell",
    "jupyter_distributed.mcp:append_cell",
    "jupyter_distributed.mcp:run_cell",
    "jupyter_distributed.mcp:set_distributed_processes",
    "jupyter_distributed.mcp:select_distributed_debug_rank",
]


async def get_distributed_notebook_info(
    notebook_path: str | None = None,
) -> dict[str, Any]:
    """Inspect the process model for a Jupyter Distributed notebook.

    Call this before reasoning about execution. With more than one process,
    ordinary cells run on every persistent rank using SPMD semantics and each
    rank has independent state. Use ``%%rank N`` for rank-local analysis and
    ``read_distributed_cell_outputs`` to inspect every rank. The current rank
    and process count are available through the ``RANK`` and ``WORLD_SIZE``
    environment variables. To change the process count, call
    ``set_distributed_processes``; do not edit notebook metadata or use a
    generic kernel restart operation.
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
                "process_control": _process_control(),
                "cell_workflow": _cell_workflow(),
                "process_environment": _process_environment(
                    active=bool(model.get("proxied", model["distributed"]))
                ),
            }

    document = await _read_notebook(path)
    world_size = _saved_world_size(document)
    return {
        "notebook_path": _relative_notebook_path(server, path),
        "running": session is not None,
        "world_size": world_size,
        "distributed": world_size > 1,
        "world_size_source": "notebook_metadata" if world_size > 1 else "default",
        "process_control": _process_control(),
        "cell_workflow": _cell_workflow(),
        "process_environment": _process_environment(active=False),
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


async def append_cell(source: str, cell_type: str = "code") -> dict[str, Any]:
    """Write a cell at the bottom of the open notebook.

    The first empty cell in the trailing run of blank cells is reused; a new
    cell is appended only when the notebook ends with content. Use ``run_cell``
    with the returned ``cell_id`` when execution is requested. Use
    ``insert_cell`` instead when the location is not the bottom of the notebook.

    Prefer this dedicated MCP tool over the raw
    ``jupyterlab-ai-commands:add-cell`` command.
    """
    if cell_type not in {"code", "markdown", "raw"}:
        raise ValueError("cell_type must be code, markdown, or raw")

    from jupyter_ai_tools.toolkits.notebook import add_cell, edit_cell, read_notebook_json

    path = await _notebook_path(None)
    notebook_path = _relative_notebook_path(_server_app(), path)
    document = await read_notebook_json(notebook_path)
    cells = document.get("cells", [])
    target = _first_trailing_blank_cell(cells)
    reused_blank_cell = target is not None
    if target is not None:
        cell_index, cell_id = target
        await edit_cell(notebook_path, cell_id, content=source, cell_type=cell_type)
    else:
        await add_cell(notebook_path, content=source, cell_type=cell_type)
        document = await read_notebook_json(notebook_path)
        cells = document.get("cells", [])
        if not cells or not isinstance(cells[-1], Mapping):
            raise RuntimeError("The appended notebook cell could not be found")
        cell_index = len(cells) - 1
        cell_id = cells[cell_index].get("id")
    if not isinstance(cell_id, str):
        raise RuntimeError("The bottom notebook cell has no id")
    return {
        "notebook_path": notebook_path,
        "cell_id": cell_id,
        "cell_index": cell_index,
        "reused_blank_cell": reused_blank_cell,
    }


async def run_cell(
    cell_id: str,
    notebook_path: str | None = None,
) -> dict[str, Any]:
    """Run one existing cell and verify its distributed rank aggregation.

    This is the canonical MCP execution tool for a Jupyter Distributed
    notebook. User-code failures are reported from the completed rank payload.
    If Jupyter reports success but the rank payload is missing or incomplete,
    the result is classified as an extension error. Prefer this tool over the raw
    ``jupyterlab-ai-commands:run-cell`` command returned by
    ``list_all_commands``.
    """
    from jupyterlab_commands_toolkit.tools import execute_command

    path = await _notebook_path(notebook_path)
    relative_path = _relative_notebook_path(_server_app(), path)
    info = await get_distributed_notebook_info(relative_path)
    world_size = int(info["world_size"])
    response = await execute_command(
        "jupyter-distributed:run-cell",
        {"cellId": cell_id, "notebookPath": relative_path},
    )
    result = response.get("result") if isinstance(response, Mapping) else None
    if (
        not isinstance(response, Mapping)
        or not response.get("success")
        or not isinstance(result, Mapping)
    ):
        return {
            "success": False,
            "classification": "execution_error",
            "notebook_path": relative_path,
            "cell_id": cell_id,
            "world_size": world_size,
            "execution": response,
            "message": str(
                response.get("error", "Unable to execute the notebook cell")
                if isinstance(response, Mapping)
                else "Unable to execute the notebook cell"
            ),
        }
    execution = result.get("execution")
    if _tool_pending(execution):
        return {
            "success": True,
            "classification": "execution_pending",
            "notebook_path": relative_path,
            "cell_id": cell_id,
            "world_size": world_size,
            "execution": execution,
            "message": (
                "The frontend stopped waiting, but the cell may still be running. "
                "Inspect the cell outputs again later."
            ),
        }
    distributed_value = result.get("distributed_execution")
    distributed = (
        _compact_execution(distributed_value)
        if isinstance(distributed_value, Mapping)
        else None
    )
    command_succeeded = _tool_succeeded(execution)
    has_output = isinstance(execution, Mapping) and execution.get("hasOutput") is not False
    if distributed is None and command_succeeded and not has_output:
        return {
            "success": True,
            "classification": "ok",
            "notebook_path": relative_path,
            "cell_id": cell_id,
            "world_size": world_size,
            "execution": execution,
            "distributed_execution": None,
            "rank_output_counts": {str(rank): 0 for rank in range(world_size)},
        }
    if distributed is None or not _execution_complete(distributed, world_size):
        return {
            "success": False,
            "classification": (
                "jupyter_distributed_error" if command_succeeded else "execution_error"
            ),
            "notebook_path": relative_path,
            "cell_id": cell_id,
            "world_size": world_size,
            "execution": execution,
            "distributed_execution": distributed,
            "message": (
                "Jupyter reported that the cell ran, but Jupyter Distributed did not "
                "receive a complete final payload from every rank. Treat this as an "
                "extension or reconnect problem, not evidence that the cell source is wrong."
                if command_succeeded
                else "Jupyter did not complete the cell execution request. Inspect the cell "
                "and kernel state."
            ),
        }

    status = distributed.get("status")
    classification = "ok" if status == "ok" else "user_code_error"
    return {
        "success": status == "ok" and _tool_succeeded(execution),
        "classification": classification,
        "notebook_path": relative_path,
        "cell_id": cell_id,
        "world_size": world_size,
        "execution": execution,
        "distributed_execution": distributed,
        "rank_output_counts": {
            str(rank.get("rank")): len(rank.get("outputs", []))
            for rank in distributed.get("ranks", [])
            if isinstance(rank, Mapping)
        },
    }


async def set_distributed_processes(
    processes: int,
    notebook_path: str | None = None,
) -> dict[str, Any]:
    """Set the live notebook process count and restart its kernel group.

    Always use this tool to change processes. Do not edit notebook metadata or
    call Jupyter's generic restart endpoint: neither operation configures the
    distributed process group correctly. This tool performs the coordinated
    restart and updates the notebook's Processes control and saved metadata.
    All in-memory kernel state is lost. After this returns, use
    ``run_cell`` for an existing cell, or ``append_cell`` followed by
    ``run_cell`` for new work. The process tool already performed the required
    restart, so do not restart again.
    """
    if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
        raise ValueError("processes must be a positive integer")

    path = await _notebook_path(notebook_path)
    relative_path = _relative_notebook_path(_server_app(), path)

    from jupyterlab_commands_toolkit.tools import execute_command

    response = await execute_command(
        "jupyter-distributed:set-processes",
        {"processes": processes, "notebookPath": relative_path},
    )
    if not response.get("success"):
        raise RuntimeError(str(response.get("error", "Unable to set distributed processes")))
    result = response.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("The process controller returned an invalid response")
    return dict(result)


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


def _process_environment(*, active: bool) -> dict[str, Any]:
    return {
        "active": active,
        "execution_model": (
            "Ordinary cells execute on every persistent process using SPMD semantics. "
            "Each process has independent state."
        ),
        "rank": "RANK",
        "world_size": "WORLD_SIZE",
        "framework_convenience_variables": {
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
        "framework_note": (
            "When the notebook uses PyTorch or JAX distributed APIs, these convenience "
            "variables are already present, so separate environment setup is unnecessary."
        ),
    }


def _process_control() -> dict[str, Any]:
    return {
        "change_tool": "set_distributed_processes",
        "restarts_kernel": True,
        "loses_in_memory_state": True,
        "do_not_edit_notebook_metadata": True,
        "do_not_use_generic_kernel_restart": True,
    }


def _cell_workflow() -> dict[str, Any]:
    return {
        "tool_preference": (
            "Prefer dedicated MCP tools. Use list_all_commands and execute_command only "
            "when no dedicated MCP tool covers the operation."
        ),
        "new_bottom_cell_tool": "append_cell",
        "reuses_first_trailing_blank_cell": True,
        "run_existing_cell_tool": "run_cell",
        "editing_preference": (
            "When debugging or revising an existing cell, prefer editing that cell in place. "
            "Add a new cell for conceptually new work."
        ),
        "superseded_raw_commands": [
            "jupyterlab-ai-commands:add-cell",
            "jupyterlab-ai-commands:run-cell",
        ],
    }


def _first_trailing_blank_cell(cells: Any) -> tuple[int, str] | None:
    if not isinstance(cells, list):
        return None
    last_content = -1
    for index, cell in enumerate(cells):
        if isinstance(cell, Mapping) and _source_text(cell.get("source")).strip():
            last_content = index
    target_index = last_content + 1
    if target_index >= len(cells):
        return None
    target = cells[target_index]
    if not isinstance(target, Mapping) or _source_text(target.get("source")).strip():
        return None
    cell_id = target.get("id")
    return (target_index, cell_id) if isinstance(cell_id, str) else None


def _source_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return "" if source is None else str(source)


def _execution_complete(execution: Mapping[str, Any], world_size: int) -> bool:
    ranks = execution.get("ranks", [])
    return (
        execution.get("status") in {"ok", "error", "aborted"}
        and execution.get("world_size") == world_size
        and isinstance(ranks, list)
        and len(ranks) == world_size
        and all(
            isinstance(rank, Mapping) and rank.get("status") in {"ok", "error", "aborted"}
            for rank in ranks
        )
    )


def _tool_succeeded(result: Any) -> bool:
    if not isinstance(result, Mapping) or not result.get("success", False):
        return False
    inner = result.get("result")
    return not isinstance(inner, Mapping) or inner.get("success", True) is True


def _tool_pending(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("status") == "timed_out":
        return True
    inner = result.get("result")
    return isinstance(inner, Mapping) and inner.get("status") == "timed_out"


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
    "append_cell",
    "get_distributed_notebook_info",
    "get_selected_notebook_cell",
    "read_distributed_cell_outputs",
    "run_cell",
    "set_distributed_processes",
    "select_distributed_debug_rank",
]
