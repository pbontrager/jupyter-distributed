from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

import jupyter_distributed.kernel_group as kernel_group_module
from jupyter_distributed.kernel_group import DistributedKernelGroup
from jupyter_distributed.protocol import RankExecution
from jupyter_distributed.rank_kernel import _RankOutputBuffer


class FakeRankKernel:
    instances: ClassVar[list[FakeRankKernel]] = []

    def __init__(self, rank: int, env: dict[str, str], **kwargs: Any) -> None:
        self.rank = rank
        self.env = env
        self.started = False
        self.stopped = False
        self.interrupted = False
        self.gate: asyncio.Event | None = None
        self.executions: list[tuple[str, dict[str, Any]]] = []
        self.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def execute(self, code: str, **kwargs: Any) -> RankExecution:
        self.executions.append((code, kwargs))
        if self.gate is not None:
            await self.gate.wait()
        return RankExecution(self.rank, "ok")

    async def interrupt(self) -> None:
        self.interrupted = True

    async def shutdown(self, *, now: bool = False) -> None:
        self.stopped = True

    async def is_alive(self) -> bool:
        return self.started and not self.stopped


@pytest.fixture(autouse=True)
def fake_rank_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRankKernel.instances = []
    monkeypatch.setattr(kernel_group_module, "RankKernel", FakeRankKernel)
    monkeypatch.setattr(kernel_group_module, "_free_port", lambda host: 23456)


@pytest.mark.asyncio
async def test_start_sets_distributed_rank_environment() -> None:
    group = DistributedKernelGroup(2, env={"EXTRA": "yes"})
    await group.start()

    assert [rank.env["RANK"] for rank in group.ranks] == ["0", "1"]
    assert {rank.env["WORLD_SIZE"] for rank in group.ranks} == {"2"}
    assert {rank.env["LOCAL_WORLD_SIZE"] for rank in group.ranks} == {"2"}
    assert {rank.env["MASTER_ADDR"] for rank in group.ranks} == {"127.0.0.1"}
    assert {rank.env["MASTER_PORT"] for rank in group.ranks} == {"23456"}
    assert {rank.env["JAX_COORDINATOR_ADDRESS"] for rank in group.ranks} == {"127.0.0.1:23456"}
    assert [rank.env["JAX_PROCESS_ID"] for rank in group.ranks] == ["0", "1"]
    assert {rank.env["JAX_NUM_PROCESSES"] for rank in group.ranks} == {"2"}
    assert all(rank.env["EXTRA"] == "yes" for rank in group.ranks)
    assert all("PYTHONBREAKPOINT" in rank.env for rank in group.ranks)


@pytest.mark.asyncio
async def test_jax_coordinator_address_brackets_ipv6_hosts() -> None:
    group = DistributedKernelGroup(1, master_addr="::1", master_port=23456)
    await group.start()

    assert group.ranks[0].env["JAX_COORDINATOR_ADDRESS"] == "[::1]:23456"


@pytest.mark.asyncio
async def test_execute_fans_out_concurrently() -> None:
    group = DistributedKernelGroup(2)
    await group.start()
    gate = asyncio.Event()
    for rank in group.ranks:
        rank.gate = gate

    execution_task = asyncio.create_task(group.execute("work()"))
    await asyncio.sleep(0)
    assert (await group.status()).state == "busy"
    gate.set()
    execution = await execution_task

    assert execution.execution_count == 1
    assert [result.rank for result in execution.ranks] == [0, 1]
    assert (await group.status()).state == "idle"


@pytest.mark.asyncio
async def test_execute_can_target_one_rank_and_keep_other_histories_aligned() -> None:
    group = DistributedKernelGroup(3)
    await group.start()

    execution = await group.execute(
        "inspect_rank()",
        user_expressions={"value": "value"},
        target_rank=1,
    )

    assert execution.execution_count == 1
    assert [rank.executions[0][0] for rank in group.ranks] == [
        "pass",
        "inspect_rank()",
        "pass",
    ]
    assert group.ranks[0].executions[0][1]["store_history"] is True
    assert group.ranks[1].executions[0][1]["user_expressions"] == {"value": "value"}
    assert group.ranks[2].executions[0][1]["user_expressions"] is None


@pytest.mark.asyncio
async def test_execute_rejects_unknown_target_rank() -> None:
    group = DistributedKernelGroup(2)
    await group.start()

    with pytest.raises(ValueError, match="between 0 and 1"):
        await group.execute("work()", target_rank=2)

    assert all(not rank.executions for rank in group.ranks)


def test_output_buffer_applies_stream_display_and_clear_updates() -> None:
    buffer = _RankOutputBuffer(2)

    assert buffer.handle("stream", {"name": "stderr", "text": "0%\r"})
    assert buffer.handle("stream", {"name": "stderr", "text": "50%\r"})
    assert buffer.handle("stream", {"name": "stderr", "text": "100%\n"})
    assert buffer.snapshot()[0].content["text"] == "100%\n"

    display = {
        "data": {"text/plain": "first"},
        "metadata": {},
        "transient": {"display_id": "progress"},
    }
    assert buffer.handle("display_data", display)
    assert buffer.handle(
        "update_display_data",
        {
            **display,
            "data": {"text/plain": "second"},
        },
    )
    assert buffer.snapshot()[1].content["data"]["text/plain"] == "second"

    assert not buffer.handle("clear_output", {"wait": True})
    assert buffer.handle("stream", {"name": "stdout", "text": "after clear\n"})
    assert len(buffer.snapshot()) == 1
    assert buffer.snapshot()[0].content["text"] == "after clear\n"


@pytest.mark.asyncio
async def test_restart_replaces_every_rank_and_world_size() -> None:
    group = DistributedKernelGroup(2)
    await group.start()
    original = tuple(group.ranks)

    await group.restart(3)

    assert all(rank.stopped for rank in original)
    assert len(group.ranks) == 3
    assert group.world_size == 3
    assert group.execution_count == 0


@pytest.mark.asyncio
async def test_interrupt_and_shutdown_apply_to_every_rank() -> None:
    group = DistributedKernelGroup(2)
    await group.start()
    ranks = tuple(group.ranks)

    await group.interrupt()
    await group.shutdown(now=True)

    assert all(rank.interrupted for rank in ranks)
    assert all(rank.stopped for rank in ranks)
    assert (await group.status()).state == "stopped"
