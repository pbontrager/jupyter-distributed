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
