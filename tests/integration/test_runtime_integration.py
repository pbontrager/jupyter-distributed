from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from jupyter_client import AsyncKernelManager
from jupyter_client.asynchronous.client import AsyncKernelClient

from jupyter_distributed.coordinator import DistributedKernelCoordinator
from jupyter_distributed.kernel_group import DistributedKernelGroup
from jupyter_distributed.kernel_proxy import RANK_MIME


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
    message_id = client.execute(code)
    data = None
    while True:
        message = await client.get_iopub_msg(timeout=20)
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        if message["msg_type"] in {"display_data", "update_display_data"}:
            data = message["content"]["data"]
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            break
    reply = await client.get_shell_msg(timeout=5)
    return reply["content"], data


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
        assert coordinator.describe("logical-kernel")["world_size"] == 1

        model = await coordinator.set_world_size("logical-kernel", 2)
        assert model == {
            "kernel_id": "logical-kernel",
            "kernel_name": "python3",
            "world_size": 2,
            "distributed": True,
        }
        assert manager.kernel_name == "python3"
        await client.wait_for_ready(timeout=20)

        info_id = client.kernel_info()
        while True:
            info_reply = await client.get_shell_msg(timeout=5)
            if info_reply.get("parent_header", {}).get("msg_id") == info_id:
                break
        assert info_reply["content"]["implementation"] == "ipython"
        assert info_reply["content"]["debugger"] is True

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

        live_id = client.execute(
            "import time\n"
            "print('started', flush=True)\n"
            "time.sleep(0.5)\n"
            "print('finished', flush=True)"
        )
        saw_live_output = False
        while True:
            message = await client.get_iopub_msg(timeout=10)
            if message.get("parent_header", {}).get("msg_id") != live_id:
                continue
            if message["msg_type"] == "update_display_data":
                payload = message["content"]["data"].get(RANK_MIME)
                if payload and payload["status"] == "busy":
                    text = str(payload["ranks"][0]["outputs"])
                    saw_live_output = saw_live_output or "started" in text
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                break
        live_reply = await client.get_shell_msg(timeout=5)
        assert live_reply["content"]["status"] == "ok"
        assert saw_live_output

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
        assert manager.kernel_name == "python3"
        await client.wait_for_ready(timeout=20)
        reply, data = await execute_through_proxy(client, "1 + 1")
        assert reply["status"] == "ok"
        assert data is None
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
