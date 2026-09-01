from __future__ import annotations

import asyncio
from collections import deque
from queue import Empty
from typing import Any, cast

import pytest

from jupyter_distributed.rank_kernel import RankKernel


class ConcurrentReadSensitiveClient:
    """Model a ZMQ channel that cannot safely service concurrent readers."""

    def __init__(self) -> None:
        self._next_id = 0
        self._pending: deque[str] = deque()
        self.active_reads = 0
        self.max_active_reads = 0

    def kernel_info(self) -> str:
        self._next_id += 1
        message_id = f"request-{self._next_id}"
        self._pending.append(message_id)
        return message_id

    async def get_shell_msg(self, timeout: float | None = None) -> dict[str, Any]:
        self.active_reads += 1
        self.max_active_reads = max(self.max_active_reads, self.active_reads)
        try:
            if self.active_reads > 1:
                raise Empty
            await asyncio.sleep(0.01)
            return {
                "parent_header": {"msg_id": self._pending.popleft()},
                "content": {"implementation": "ipython"},
            }
        finally:
            self.active_reads -= 1


class FakeSession:
    def __init__(self) -> None:
        self.next_id = 0

    def msg(self, message_type: str, content: dict[str, Any]) -> dict[str, Any]:
        self.next_id += 1
        return {
            "header": {"msg_id": f"debug-{self.next_id}", "msg_type": message_type},
            "content": content,
        }


class FakeControlChannel:
    def __init__(self, pending: deque[str]) -> None:
        self.pending = pending

    def send(self, message: dict[str, Any]) -> None:
        self.pending.append(message["header"]["msg_id"])


class ConcurrentControlReadSensitiveClient:
    def __init__(self) -> None:
        self.pending: deque[str] = deque()
        self.session = FakeSession()
        self.control_channel = FakeControlChannel(self.pending)
        self.active_reads = 0
        self.max_active_reads = 0

    async def get_control_msg(self, timeout: float | None = None) -> dict[str, Any]:
        self.active_reads += 1
        self.max_active_reads = max(self.max_active_reads, self.active_reads)
        try:
            if self.active_reads > 1:
                raise Empty
            await asyncio.sleep(0.01)
            return {
                "parent_header": {"msg_id": self.pending.popleft()},
                "content": {"success": True},
            }
        finally:
            self.active_reads -= 1


class RecordingShellChannel:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


class CommClient:
    def __init__(self) -> None:
        self.shell_channel = RecordingShellChannel()


@pytest.mark.asyncio
async def test_shell_requests_are_serialized_per_rank() -> None:
    kernel = RankKernel(0, {"WORLD_SIZE": "1"})
    client = ConcurrentReadSensitiveClient()
    kernel.client = cast(Any, client)

    replies = await asyncio.gather(
        kernel.request("kernel_info"),
        kernel.request("kernel_info"),
        kernel.request("kernel_info"),
    )

    assert replies == [{"implementation": "ipython"}] * 3
    assert client.max_active_reads == 1


@pytest.mark.asyncio
async def test_debug_requests_are_serialized_per_rank() -> None:
    kernel = RankKernel(0, {"WORLD_SIZE": "1"})
    client = ConcurrentControlReadSensitiveClient()
    kernel.client = cast(Any, client)

    replies = await asyncio.gather(
        kernel.debug_request({"command": "debugInfo"}),
        kernel.debug_request({"command": "threads"}),
        kernel.debug_request({"command": "stackTrace"}),
    )

    assert replies == [{"success": True}] * 3
    assert client.max_active_reads == 1


def test_forwarded_comm_drops_proxy_subshell_id() -> None:
    kernel = RankKernel(0, {"WORLD_SIZE": "2"})
    client = CommClient()
    kernel.client = cast(Any, client)
    parent = {
        "header": {
            "msg_id": "frontend-comm",
            "msg_type": "comm_msg",
            "session": "frontend-session",
            "subshell_id": "proxy-only-subshell",
        },
        "parent_header": {"msg_id": "parent"},
    }

    kernel.send_comm(
        "comm_msg",
        {"comm_id": "widget", "data": {"value": 1}},
        parent=parent,
    )

    assert len(client.shell_channel.messages) == 1
    forwarded = client.shell_channel.messages[0]
    assert forwarded["header"] == {
        "msg_id": "frontend-comm",
        "msg_type": "comm_msg",
        "session": "frontend-session",
    }
    assert parent["header"]["subshell_id"] == "proxy-only-subshell"
