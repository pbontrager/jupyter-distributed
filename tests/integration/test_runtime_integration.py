from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from jupyter_client import AsyncKernelManager
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.kernelspec import KernelSpecManager

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
        if message["msg_type"] == "display_data":
            data = message["content"]["data"]
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            break
    reply = await client.get_shell_msg(timeout=5)
    return reply["content"], data


@pytest.mark.asyncio
async def test_proxy_entrypoint_mime_world_size_and_standard_interrupt(
    tmp_path: Path,
) -> None:
    kernel_dir = tmp_path / "distributed"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "jupyter_distributed.kernel",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Distributed Python",
                "language": "python",
                "interrupt_mode": "message",
            }
        )
    )
    manager = AsyncKernelManager(
        kernel_name="distributed",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(tmp_path)]),
    )
    await manager.start_kernel(cwd=str(Path.cwd()))
    client = manager.client()
    client.start_channels()
    try:
        await client.wait_for_ready(timeout=20)
        reply, _ = await execute_through_proxy(client, "%spmd_world_size 2")
        assert reply["status"] == "ok"
        assert (
            reply["user_expressions"]["jupyter_distributed_world_size"]["data"]["text/plain"]
            == "2"
        )

        reply, data = await execute_through_proxy(client, "import os; int(os.environ['RANK'])")
        assert reply["status"] == "ok"
        assert data is not None
        assert set(data) >= {RANK_MIME, "text/html", "text/plain"}
        assert data[RANK_MIME]["world_size"] == 2

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
            if message["msg_type"] == "display_data":
                interrupted_data = message["content"]["data"]
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                break
        interrupted_reply = await client.get_shell_msg(timeout=5)
        assert interrupted_reply["content"]["status"] == "error"
        assert interrupted_data[RANK_MIME]["status"] == "error"
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
