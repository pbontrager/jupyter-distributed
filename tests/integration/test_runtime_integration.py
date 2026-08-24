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

        blocked_debugger = await group.execute("breakpoint()")
        assert blocked_debugger.status == "error"
        assert all(rank.status == "error" for rank in blocked_debugger.ranks)
        assert "disabled for this Jupyter Distributed kernel" in output_text(
            blocked_debugger.ranks[0]
        )

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


@pytest.mark.asyncio
async def test_server_coordinator_wraps_selected_kernel_and_standard_lifecycle() -> None:
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
        assert info_reply["content"]["debugger"] is False

        debug_request = client.session.msg(
            "debug_request",
            content={
                "seq": 1,
                "type": "request",
                "command": "debugInfo",
                "arguments": {},
            },
        )
        client.control_channel.send(debug_request)
        debug_reply = await client.get_control_msg(timeout=5)
        assert debug_reply["msg_type"] == "debug_reply"
        assert debug_reply["content"] == {
            "type": "response",
            "request_seq": 1,
            "success": False,
            "command": "debugInfo",
            "message": "Debugging is not supported for distributed processes.",
        }

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
