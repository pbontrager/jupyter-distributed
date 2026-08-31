from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from jupyter_client import AsyncKernelManager
from jupyter_client.asynchronous.client import AsyncKernelClient

from jupyter_distributed.coordinator import DistributedKernelCoordinator
from jupyter_distributed.kernel_group import DistributedKernelGroup
from jupyter_distributed.kernel_proxy import RANK_MIME, RANK_UPDATE_COMM_TARGET


def output_text(result: object) -> str:
    return "".join(output.plain_text() for output in result.outputs)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_two_rank_environment_state_outputs_errors_and_restart() -> None:
    group = DistributedKernelGroup(2)
    try:
        await group.start()
        first = await group.execute(
            "import os, sys\n"
            "x = int(os.environ['RANK'])\n"
            "print(os.environ['RANK'], os.environ['LOCAL_RANK'], "
            "os.environ['WORLD_SIZE'], os.environ['LOCAL_WORLD_SIZE'])\n"
            "print(f'err-{x}', file=sys.stderr)\n"
            "from IPython.display import display\n"
            "display({'rank': x})\n"
            "x"
        )
        assert first.status == "ok"
        assert [output_text(rank) for rank in first.ranks] == [
            "0 0 2 2\nerr-0\n{'rank': 0}0",
            "1 1 2 2\nerr-1\n{'rank': 1}1",
        ]
        assert {output.kind for rank in first.ranks for output in rank.outputs} >= {
            "stream",
            "display_data",
            "execute_result",
        }

        persisted = await group.execute("x + 10")
        assert [output_text(rank) for rank in persisted.ranks] == ["10", "11"]

        errored = await group.execute("if x == 1:\n    raise ValueError('rank one')")
        assert errored.status == "error"
        assert [rank.status for rank in errored.ranks] == ["ok", "error"]
        assert any(output.kind == "error" for output in errored.ranks[1].outputs)

        unattached_breakpoint = await group.execute("breakpoint(); 'still running'")
        assert unattached_breakpoint.status == "ok"
        assert all(
            "enable the notebook debugger and run the cell again" in output_text(rank)
            for rank in unattached_breakpoint.ranks
        )
        assert all("'still running'" in output_text(rank) for rank in unattached_breakpoint.ranks)

        old_ranks = tuple(group.ranks)
        await group.restart(1)
        assert group.world_size == 1
        assert len(group.ranks) == 1
        assert not any(await asyncio.gather(*(rank.is_alive() for rank in old_ranks)))
        restarted = await group.execute("'x' in globals()")
        assert output_text(restarted.ranks[0]) == "False"
    finally:
        await group.shutdown(now=True)

    assert not group.ranks
    assert (await group.status()).state == "stopped"


async def execute_through_proxy(
    client: AsyncKernelClient, code: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    comm_id = open_rank_update_comm(client)
    message_id = client.execute(code)
    data = None
    while True:
        message = await client.get_iopub_msg(timeout=20)
        snapshot = rank_update_snapshot(message, comm_id, message_id)
        if snapshot is not None:
            data = snapshot["data"]
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        if message["msg_type"] in {"display_data", "update_display_data"}:
            data = message["content"]["data"]
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            break
    reply = await client.get_shell_msg(timeout=5)
    client.shell_channel.send(
        client.session.msg("comm_close", content={"comm_id": comm_id, "data": {}})
    )
    return reply["content"], data


def open_rank_update_comm(client: AsyncKernelClient) -> str:
    comm_id = uuid4().hex
    client.shell_channel.send(
        client.session.msg(
            "comm_open",
            content={
                "comm_id": comm_id,
                "target_name": RANK_UPDATE_COMM_TARGET,
                "data": {},
            },
        )
    )
    return comm_id


def rank_update_snapshot(
    message: dict[str, Any],
    comm_id: str,
    parent_id: str | None = None,
) -> dict[str, Any] | None:
    if (
        message.get("msg_type") != "comm_msg"
        or message.get("content", {}).get("comm_id") != comm_id
    ):
        return None
    if parent_id is not None and message.get("parent_header", {}).get("msg_id") != parent_id:
        return None
    data = message["content"].get("data", {})
    snapshot = data.get("snapshot") if data.get("method") == "update" else None
    return snapshot if isinstance(snapshot, dict) else None


async def debug_through_proxy(
    client: AsyncKernelClient,
    sequence: int,
    command: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = client.session.msg(
        "debug_request",
        content={
            "seq": sequence,
            "type": "request",
            "command": command,
            "arguments": arguments or {},
        },
    )
    message_id = request["header"]["msg_id"]
    client.control_channel.send(request)
    while True:
        reply = await client.get_control_msg(timeout=30)
        if reply.get("parent_header", {}).get("msg_id") == message_id:
            return reply["content"]


async def shell_reply(client: AsyncKernelClient, message_id: str) -> dict[str, Any]:
    while True:
        reply = await client.get_shell_msg(timeout=10)
        if reply.get("parent_header", {}).get("msg_id") == message_id:
            return reply["content"]


async def iopub_message(
    client: AsyncKernelClient,
    message_type: str,
    predicate: Any = None,
) -> dict[str, Any]:
    while True:
        message = await client.get_iopub_msg(timeout=20)
        if message.get("msg_type") != message_type:
            continue
        if predicate is None or predicate(message):
            return message


@pytest.mark.asyncio
async def test_server_coordinator_wraps_selected_kernel_and_standard_lifecycle(
    tmp_path: Path,
) -> None:
    manager = AsyncKernelManager(kernel_name="python3")
    await manager.start_kernel(cwd=str(Path.cwd()))

    class SingleKernelManager:
        def get_kernel(self, kernel_id: str) -> AsyncKernelManager:
            assert kernel_id == "logical-kernel"
            return manager

        async def restart_kernel(self, kernel_id: str) -> None:
            assert kernel_id == "logical-kernel"
            await manager.restart_kernel(now=False)

    coordinator = DistributedKernelCoordinator(SingleKernelManager())
    client = manager.client()
    client.start_channels()
    try:
        await client.wait_for_ready(timeout=20)
        initial = coordinator.describe("logical-kernel")
        assert initial["world_size"] == 1
        assert initial["proxied"] is False

        model = await coordinator.set_world_size("logical-kernel", 1)
        assert model["proxied"] is True
        await client.wait_for_ready(timeout=20)
        reply, data = await execute_through_proxy(
            client,
            "import os\n"
            "required = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', "
            "'MASTER_ADDR', 'MASTER_PORT', 'JAX_COORDINATOR_ADDRESS', "
            "'JAX_PROCESS_ID', 'JAX_NUM_PROCESSES']\n"
            "all(os.environ.get(name) for name in required) and "
            "(os.environ['RANK'], os.environ['WORLD_SIZE'], "
            "os.environ['JAX_PROCESS_ID'], os.environ['JAX_NUM_PROCESSES']) "
            "== ('0', '1', '0', '1')",
        )
        assert reply["status"] == "ok"
        assert data[RANK_MIME]["world_size"] == 1
        rank_output = data[RANK_MIME]["ranks"][0]["outputs"][-1]
        assert rank_output["content"]["data"]["text/plain"] == "True"

        model = await coordinator.set_world_size("logical-kernel", 2)
        assert model == {
            "kernel_id": "logical-kernel",
            "kernel_name": "python3",
            "world_size": 2,
            "distributed": True,
            "proxied": True,
        }
        assert manager.kernel_name == "python3"
        await client.wait_for_ready(timeout=20)

        info_ids = {client.kernel_info() for _ in range(3)}
        info_replies: dict[str, dict[str, Any]] = {}
        while info_ids - info_replies.keys():
            info_reply = await client.get_shell_msg(timeout=5)
            message_id = info_reply.get("parent_header", {}).get("msg_id")
            if message_id in info_ids:
                info_replies[message_id] = info_reply["content"]
        assert all(
            reply["implementation"] == "jupyter_distributed" for reply in info_replies.values()
        )
        assert all(reply["language_info"]["name"] == "python" for reply in info_replies.values())
        assert all(reply["debugger"] is True for reply in info_replies.values())

        initialized = await debug_through_proxy(
            client,
            1,
            "initialize",
            {
                "clientID": "jupyter-distributed-test",
                "clientName": "jupyter-distributed-test",
                "adapterID": "python",
                "pathFormat": "path",
                "linesStartAt1": True,
                "columnsStartAt1": True,
                "supportsVariableType": True,
                "supportsVariablePaging": True,
            },
        )
        assert initialized["success"] is True
        assert (await debug_through_proxy(client, 2, "attach"))["success"] is True

        debug_code = (
            "import os\ndebug_value = int(os.environ['RANK'])\ndebug_value += 10\ndebug_value"
        )
        dumped = await debug_through_proxy(client, 3, "dumpCell", {"code": debug_code})
        source_path = dumped["body"]["sourcePath"]
        breakpoints = await debug_through_proxy(
            client,
            4,
            "setBreakpoints",
            {
                "source": {"path": source_path},
                "breakpoints": [{"line": 3}],
                "sourceModified": False,
            },
        )
        assert breakpoints["success"] is True
        assert breakpoints["body"]["breakpoints"][0]["verified"] is True

        debug_comm_id = open_rank_update_comm(client)
        debug_execution_id = client.execute(debug_code)
        stopped_threads: list[int] = []
        while len(stopped_threads) < 2:
            message = await client.get_iopub_msg(timeout=30)
            if (
                message["msg_type"] == "debug_event"
                and message["content"].get("event") == "stopped"
            ):
                stopped_threads.append(message["content"]["body"]["threadId"])

        debug_info = await debug_through_proxy(client, 5, "debugInfo")
        assert set(debug_info["body"]["stoppedThreads"]) == set(stopped_threads)
        threads = await debug_through_proxy(client, 6, "threads")
        assert {
            thread["name"].split(":", 1)[0]
            for thread in threads["body"]["threads"]
            if thread["id"] in stopped_threads
        } == {"Rank 0", "Rank 1"}

        rank_values: set[str] = set()
        evaluated_values: set[str] = set()
        for sequence, thread_id in enumerate(stopped_threads, start=7):
            stack = await debug_through_proxy(
                client,
                sequence,
                "stackTrace",
                {"threadId": thread_id, "startFrame": 0, "levels": 20},
            )
            frames = stack["body"]["stackFrames"]
            assert frames
            assert frames[0]["source"]["path"] == source_path
            assert frames[0]["name"].startswith("Rank ")
            scopes = await debug_through_proxy(
                client,
                sequence + 10,
                "scopes",
                {"frameId": frames[0]["id"]},
            )
            variables = await debug_through_proxy(
                client,
                sequence + 20,
                "variables",
                {"variablesReference": scopes["body"]["scopes"][0]["variablesReference"]},
            )
            rank_values.update(
                variable["value"]
                for variable in variables["body"]["variables"]
                if variable["name"] == "debug_value"
            )
            evaluated = await debug_through_proxy(
                client,
                sequence + 30,
                "evaluate",
                {
                    "expression": "debug_value * 100",
                    "frameId": frames[0]["id"],
                    "context": "repl",
                },
            )
            evaluated_values.add(evaluated["body"]["result"])
        assert rank_values == {"0", "1"}
        assert evaluated_values == {"0", "100"}

        stepped = await debug_through_proxy(
            client,
            39,
            "next",
            {"threadId": stopped_threads[0]},
        )
        assert stepped["success"] is True
        stepped_threads: list[int] = []
        while len(stepped_threads) < 2:
            message = await client.get_iopub_msg(timeout=30)
            if (
                message["msg_type"] == "debug_event"
                and message["content"].get("event") == "stopped"
            ):
                stepped_threads.append(message["content"]["body"]["threadId"])
        for sequence, thread_id in enumerate(stepped_threads, start=30):
            stack = await debug_through_proxy(
                client,
                sequence,
                "stackTrace",
                {"threadId": thread_id, "startFrame": 0, "levels": 1},
            )
            assert stack["body"]["stackFrames"][0]["line"] == 4

        continued = await debug_through_proxy(
            client,
            40,
            "continue",
            {"threadId": stepped_threads[0]},
        )
        assert continued["success"] is True
        assert continued["body"]["allThreadsContinued"] is True

        debug_data = None
        while True:
            message = await client.get_iopub_msg(timeout=30)
            snapshot = rank_update_snapshot(message, debug_comm_id, debug_execution_id)
            if snapshot is not None:
                debug_data = snapshot["data"]
            if message.get("parent_header", {}).get("msg_id") != debug_execution_id:
                continue
            if message["msg_type"] in {"display_data", "update_display_data"}:
                debug_data = message["content"]["data"]
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                break
        while True:
            debug_execution_reply = await client.get_shell_msg(timeout=5)
            if debug_execution_reply.get("parent_header", {}).get("msg_id") == debug_execution_id:
                break
        assert debug_execution_reply["content"]["status"] == "ok"
        assert debug_data is not None
        assert [
            rank["outputs"][-1]["content"]["data"]["text/plain"]
            for rank in debug_data[RANK_MIME]["ranks"]
        ] == ["10", "11"]

        external_source = tmp_path / "debug_target.py"
        external_source.write_text(
            "import os\n\n"
            "def external_value():\n"
            "    rank = int(os.environ['RANK'])\n"
            "    breakpoint()\n"
            "    return rank + 20\n",
            encoding="utf-8",
        )
        external_comm_id = open_rank_update_comm(client)
        external_execution_id = client.execute(
            f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\n"
            "from debug_target import external_value\nexternal_value()"
        )
        external_stopped_threads: list[int] = []
        while len(external_stopped_threads) < 2:
            message = await client.get_iopub_msg(timeout=30)
            if (
                message["msg_type"] == "debug_event"
                and message["content"].get("event") == "stopped"
            ):
                external_stopped_threads.append(message["content"]["body"]["threadId"])
        for sequence, thread_id in enumerate(external_stopped_threads, start=42):
            stack = await debug_through_proxy(
                client,
                sequence,
                "stackTrace",
                {"threadId": thread_id, "startFrame": 0, "levels": 1},
            )
            frame = stack["body"]["stackFrames"][0]
            assert Path(frame["source"]["path"]) == external_source
            assert frame["line"] == 6

        assert (
            await debug_through_proxy(
                client,
                44,
                "continue",
                {"threadId": external_stopped_threads[0]},
            )
        )["success"] is True
        external_data = None
        while True:
            message = await client.get_iopub_msg(timeout=30)
            snapshot = rank_update_snapshot(message, external_comm_id, external_execution_id)
            if snapshot is not None:
                external_data = snapshot["data"]
            if message.get("parent_header", {}).get("msg_id") != external_execution_id:
                continue
            if message["msg_type"] in {"display_data", "update_display_data"}:
                external_data = message["content"]["data"]
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                break
        while True:
            external_execution_reply = await client.get_shell_msg(timeout=5)
            if (
                external_execution_reply.get("parent_header", {}).get("msg_id")
                == external_execution_id
            ):
                break
        assert external_execution_reply["content"]["status"] == "ok"
        assert external_data is not None
        assert [
            rank["outputs"][-1]["content"]["data"]["text/plain"]
            for rank in external_data[RANK_MIME]["ranks"]
        ] == ["20", "21"]
        assert (
            await debug_through_proxy(
                client,
                45,
                "disconnect",
                {"restart": False, "terminateDebuggee": False},
            )
        )["success"] is True

        live_comm_id = open_rank_update_comm(client)
        live_id = client.execute(
            "import time\n"
            "for token in ['Tensor', ' parallel', ' streaming', ' works']:\n"
            "    print(token, end='', flush=True)\n"
            "    time.sleep(0.1)\n"
            "print()"
        )
        saw_live_output = False
        final_snapshot = None
        output_messages = []
        while True:
            message = await client.get_iopub_msg(timeout=10)
            if (
                message["msg_type"] == "comm_msg"
                and message["content"].get("comm_id") == live_comm_id
            ):
                data = message["content"].get("data", {})
                snapshot = data.get("snapshot") if data.get("method") == "update" else None
                payload = snapshot.get("data", {}).get(RANK_MIME) if snapshot else None
                if payload and payload["status"] == "busy":
                    text = str(payload["ranks"][0]["outputs"])
                    saw_live_output = saw_live_output or "Tensor" in text
                if snapshot and snapshot.get("final"):
                    final_snapshot = snapshot
            if message.get("parent_header", {}).get("msg_id") != live_id:
                continue
            if message["msg_type"] in {"display_data", "update_display_data"}:
                output_messages.append(message)
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                break
        live_reply = await client.get_shell_msg(timeout=5)
        assert live_reply["content"]["status"] == "ok"
        assert saw_live_output
        assert final_snapshot is not None
        assert final_snapshot["data"][RANK_MIME]["status"] == "ok"
        assert output_messages[0]["msg_type"] == "display_data"
        assert all(message["msg_type"] == "update_display_data" for message in output_messages[1:])
        display_ids = [message["content"]["transient"]["display_id"] for message in output_messages]
        assert len(set(display_ids)) == 1
        assert output_messages[0]["content"]["data"][RANK_MIME]["status"] == "busy"
        assert output_messages[-1]["content"]["data"][RANK_MIME]["status"] == "ok"
        assert len(output_messages[-1]["content"]["data"][RANK_MIME]["ranks"]) == 2
        final_outputs = output_messages[-1]["content"]["data"][RANK_MIME]["ranks"]
        assert [rank["outputs"][0]["content"]["text"] for rank in final_outputs] == [
            "Tensor parallel streaming works\n",
            "Tensor parallel streaming works\n",
        ]

        reconnect_comm_id = open_rank_update_comm(client)
        restored_message = await iopub_message(
            client,
            "comm_msg",
            lambda message: (
                message["content"].get("comm_id") == reconnect_comm_id
                and message["content"].get("data", {}).get("method") == "snapshots"
            ),
        )
        restored_snapshots = restored_message["content"]["data"]["snapshots"]
        execution_id = output_messages[0]["content"]["data"][RANK_MIME]["execution_id"]
        restored = next(
            snapshot for snapshot in restored_snapshots if snapshot["execution_id"] == execution_id
        )
        assert restored["final"] is True
        assert restored["data"][RANK_MIME]["status"] == "ok"

        reply, data = await execute_through_proxy(client, "silent_value = 123")
        assert reply["status"] == "ok"
        assert data is None

        reply, data = await execute_through_proxy(
            client,
            "import os; saved = int(os.environ['RANK']); (saved, int(os.environ['WORLD_SIZE']))",
        )
        assert reply["status"] == "ok"
        assert data is not None
        assert set(data) >= {RANK_MIME, "text/html", "text/plain"}
        assert data[RANK_MIME]["world_size"] == 2

        reply, data = await execute_through_proxy(client, "saved + 10")
        assert reply["status"] == "ok"
        values = [
            rank["outputs"][-1]["content"]["data"]["text/plain"]
            for rank in data[RANK_MIME]["ranks"]
        ]
        assert values == ["10", "11"]

        reply, data = await execute_through_proxy(
            client,
            "import os\nrank = int(os.environ['RANK'])\nraise ValueError(f'failure-{rank}')",
        )
        assert reply["status"] == "error"
        assert data is not None
        assert data[RANK_MIME]["status"] == "error"
        assert [rank["status"] for rank in data[RANK_MIME]["ranks"]] == ["error", "error"]
        assert [rank["outputs"][-1]["content"]["evalue"] for rank in data[RANK_MIME]["ranks"]] == [
            "failure-0",
            "failure-1",
        ]

        reply, data = await execute_through_proxy(
            client,
            "%%rank 1\nrank_only = saved + 100\nrank_only",
        )
        assert reply["status"] == "ok"
        assert data is not None
        assert data[RANK_MIME]["target_rank"] == 1
        assert data[RANK_MIME]["ranks"][0]["outputs"] == []
        assert data[RANK_MIME]["ranks"][1]["outputs"][-1]["content"]["data"]["text/plain"] == "101"
        assert "Rank 0" not in data["text/plain"]
        assert "[Rank 1 — ok]" in data["text/plain"]

        reply, data = await execute_through_proxy(
            client,
            "('rank_only' in globals(), len(In))",
        )
        assert reply["status"] == "ok"
        assert data is not None
        values = [
            rank["outputs"][-1]["content"]["data"]["text/plain"]
            for rank in data[RANK_MIME]["ranks"]
        ]
        assert values[0].startswith("(False, ")
        assert values[1].startswith("(True, ")
        assert values[0].split(", ", 1)[1] == values[1].split(", ", 1)[1]

        reply, data = await execute_through_proxy(
            client,
            "%%rank 0\n%%capture rank_capture\nprint('captured on rank zero')",
        )
        assert reply["status"] == "ok"
        assert data is None
        reply, data = await execute_through_proxy(
            client,
            "rank_capture.stdout if 'rank_capture' in globals() else None",
        )
        assert reply["status"] == "ok"
        assert data is not None
        assert (
            data[RANK_MIME]["ranks"][0]["outputs"][-1]["content"]["data"]["text/plain"]
            == "'captured on rank zero\\n'"
        )
        assert data[RANK_MIME]["ranks"][1]["outputs"] == []

        reply, data = await execute_through_proxy(client, "%%rank 2\nvalue = 1")
        assert reply["status"] == "error"
        assert reply["ename"] == "RankMagicError"
        assert data is None

        interrupt_comm_id = open_rank_update_comm(client)
        message_id = client.execute("import time; time.sleep(60)")
        while True:
            message = await client.get_iopub_msg(timeout=10)
            if (
                message.get("parent_header", {}).get("msg_id") == message_id
                and message["msg_type"] == "status"
                and message["content"]["execution_state"] == "busy"
            ):
                break
        await asyncio.sleep(0.5)
        await manager.interrupt_kernel()
        interrupted_data = None
        while True:
            message = await client.get_iopub_msg(timeout=10)
            snapshot = rank_update_snapshot(message, interrupt_comm_id, message_id)
            if snapshot is not None:
                interrupted_data = snapshot["data"]
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            if message["msg_type"] in {"display_data", "update_display_data"}:
                interrupted_data = message["content"]["data"]
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                break
        interrupted_reply = await client.get_shell_msg(timeout=5)
        assert interrupted_reply["content"]["status"] == "error"
        assert interrupted_data[RANK_MIME]["status"] == "error"

        await manager.restart_kernel(now=False)
        await client.wait_for_ready(timeout=20)
        reply, data = await execute_through_proxy(client, "import os; os.environ['WORLD_SIZE']")
        assert reply["status"] == "ok"
        assert data[RANK_MIME]["world_size"] == 2

        model = await coordinator.set_world_size("logical-kernel", 1)
        assert model["distributed"] is False
        assert model["proxied"] is True
        assert manager.kernel_name == "python3"
        await client.wait_for_ready(timeout=20)
        reply, data = await execute_through_proxy(client, "1 + 1")
        assert reply["status"] == "ok"
        assert data[RANK_MIME]["world_size"] == 1
    finally:
        client.stop_channels()
        await manager.shutdown_kernel(now=False)


@pytest.mark.asyncio
async def test_widgets_route_by_rank_and_restore_state_with_binary_buffers() -> None:
    pytest.importorskip("ipywidgets")
    manager = AsyncKernelManager(kernel_name="python3")
    await manager.start_kernel(cwd=str(Path.cwd()))

    class SingleKernelManager:
        def get_kernel(self, kernel_id: str) -> AsyncKernelManager:
            assert kernel_id == "widget-kernel"
            return manager

        async def restart_kernel(self, kernel_id: str) -> None:
            assert kernel_id == "widget-kernel"
            await manager.restart_kernel(now=False)

    coordinator = DistributedKernelCoordinator(SingleKernelManager())
    client = manager.client()
    client.start_channels()
    try:
        await client.wait_for_ready(timeout=20)
        await coordinator.set_world_size("widget-kernel", 2)
        await client.wait_for_ready(timeout=20)

        rank_comm_id = open_rank_update_comm(client)
        execute_id = client.execute(
            "import ipywidgets as widgets, os\n"
            "from IPython.display import display\n"
            "widget_rank = int(os.environ['RANK'])\n"
            "slider = widgets.IntSlider(value=widget_rank)\n"
            "image = widgets.Image(value=bytes([widget_rank, 17, 23]))\n"
            "display(slider)"
        )
        comm_opens: dict[str, dict[str, Any]] = {}
        rank_data = None
        while True:
            message = await client.get_iopub_msg(timeout=20)
            if message["msg_type"] == "comm_open":
                comm_opens[message["content"]["comm_id"]] = message
            snapshot = rank_update_snapshot(message, rank_comm_id, execute_id)
            if snapshot is not None:
                rank_data = snapshot["data"]
            if message.get("parent_header", {}).get("msg_id") == execute_id and message[
                "msg_type"
            ] in {"display_data", "update_display_data"}:
                rank_data = message["content"]["data"]
            if (
                message.get("parent_header", {}).get("msg_id") == execute_id
                and message["msg_type"] == "status"
                and message["content"]["execution_state"] == "idle"
            ):
                break
        assert (await shell_reply(client, execute_id))["status"] == "ok"
        assert rank_data is not None

        slider_ids = [
            rank["outputs"][-1]["content"]["data"]["application/vnd.jupyter.widget-view+json"][
                "model_id"
            ]
            for rank in rank_data[RANK_MIME]["ranks"]
        ]
        assert len(set(slider_ids)) == 2
        assert all(model_id in comm_opens for model_id in slider_ids)
        image_opens = [
            message
            for message in comm_opens.values()
            if message["content"]["data"]["state"].get("_model_name") == "ImageModel"
        ]
        assert len(image_opens) == 2
        assert all(len(message.get("buffers", [])) == 1 for message in image_opens)
        image_ids = {
            bytes(message["buffers"][0])[0]: message["content"]["comm_id"]
            for message in image_opens
        }

        delayed_id = client.execute(
            "import threading\n"
            "if widget_rank == 0:\n"
            "    threading.Timer(0.5, lambda: setattr(slider, 'value', 7)).start()"
        )
        while True:
            message = await client.get_iopub_msg(timeout=20)
            if (
                message.get("parent_header", {}).get("msg_id") == delayed_id
                and message["msg_type"] == "status"
                and message["content"]["execution_state"] == "idle"
            ):
                break
        assert (await shell_reply(client, delayed_id))["status"] == "ok"
        delayed_update = await iopub_message(
            client,
            "comm_msg",
            lambda message: (
                message["content"].get("comm_id") == slider_ids[0]
                and message["content"].get("data", {}).get("state", {}).get("value") == 7
            ),
        )
        assert delayed_update["content"]["data"]["method"] == "update"

        update = client.session.msg(
            "comm_msg",
            content={
                "comm_id": slider_ids[1],
                "data": {
                    "method": "update",
                    "state": {"value": 42},
                    "buffer_paths": [],
                },
            },
        )
        client.shell_channel.send(update)
        image_update = client.session.msg(
            "comm_msg",
            content={
                "comm_id": image_ids[1],
                "data": {
                    "method": "update",
                    "state": {},
                    "buffer_paths": [["value"]],
                },
            },
        )
        image_update["buffers"] = [memoryview(b"updated")]
        client.shell_channel.send(image_update)
        await asyncio.sleep(0.2)
        reply, data = await execute_through_proxy(client, "slider.value, bytes(image.value)")
        assert reply["status"] == "ok"
        assert [
            rank["outputs"][-1]["content"]["data"]["text/plain"]
            for rank in data[RANK_MIME]["ranks"]
        ] == ["(7, b'\\x00\\x11\\x17')", "(42, b'updated')"]

        info_id = client.comm_info(target_name="jupyter.widget")
        info = await shell_reply(client, info_id)
        assert set(slider_ids).issubset(info["comms"])

        client.stop_channels()
        client = manager.client()
        client.start_channels()
        await client.wait_for_ready(timeout=20)

        control_id = uuid4().hex
        control_open = client.session.msg(
            "comm_open",
            content={
                "comm_id": control_id,
                "target_name": "jupyter.widget.control",
                "data": {},
            },
            metadata={"version": "1.0.0"},
        )
        client.shell_channel.send(control_open)
        control_request = client.session.msg(
            "comm_msg",
            content={"comm_id": control_id, "data": {"method": "request_states"}},
            metadata={"version": "1.0.0"},
        )
        client.shell_channel.send(control_request)
        restored = await iopub_message(
            client,
            "comm_msg",
            lambda message: (
                message["content"].get("comm_id") == control_id
                and message["content"].get("data", {}).get("method") == "update_states"
            ),
        )
        states = restored["content"]["data"]["states"]
        assert set(slider_ids).issubset(states)
        assert len(restored["content"]["data"]["buffer_paths"]) == 2
        assert [bytes(buffer) for buffer in restored.get("buffers", [])] == [
            bytes([0, 17, 23]),
            b"updated",
        ]

        state_request = client.session.msg(
            "comm_msg",
            content={
                "comm_id": slider_ids[1],
                "data": {"method": "request_state"},
            },
        )
        state_request_id = state_request["header"]["msg_id"]
        client.shell_channel.send(state_request)
        state_reply = await iopub_message(
            client,
            "comm_msg",
            lambda message: message.get("parent_header", {}).get("msg_id") == state_request_id,
        )
        assert state_reply["content"]["comm_id"] == slider_ids[1]
        assert state_reply["content"]["data"]["state"]["value"] == 42
    finally:
        client.stop_channels()
        await manager.shutdown_kernel(now=False)


@pytest.mark.asyncio
async def test_two_rank_gloo_collective_persists_across_cells() -> None:
    pytest.importorskip("torch")
    group = DistributedKernelGroup(2)
    try:
        await group.start()
        initialized = await group.execute(
            "from datetime import timedelta\n"
            "import os\n"
            "import torch\n"
            "import torch.distributed as dist\n"
            "dist.init_process_group('gloo', timeout=timedelta(seconds=30))\n"
            "value = torch.tensor([int(os.environ['RANK']) + 1.0])\n"
            "dist.all_reduce(value)"
        )
        assert initialized.status == "ok"

        persisted = await group.execute("value.item(), dist.is_initialized()")
        assert [output_text(rank) for rank in persisted.ranks] == ["(3.0, True)", "(3.0, True)"]

        destroyed = await group.execute("dist.destroy_process_group()")
        assert destroyed.status == "ok"
    finally:
        await group.shutdown(now=True)
