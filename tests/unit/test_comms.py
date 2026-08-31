from __future__ import annotations

import logging
from typing import Any

import pytest

from jupyter_distributed.comms import DistributedCommRouter


class FakeSession:
    def __init__(self) -> None:
        self.messages: list[tuple[Any, ...]] = []

    def send(self, *args: Any, **kwargs: Any) -> None:
        self.messages.append((*args, kwargs))


class FakeGroup:
    world_size = 2

    async def send_comm(self, *args: Any, **kwargs: Any) -> None:
        return None


class FakeKernel:
    def __init__(self) -> None:
        self.group = FakeGroup()
        self.log = logging.getLogger("test-comms")
        self.session = FakeSession()
        self.iopub_socket = object()


@pytest.mark.asyncio
async def test_widget_state_finishes_partially_when_rank_fails() -> None:
    kernel = FakeKernel()
    router = DistributedCommRouter(kernel)  # type: ignore[arg-type]
    comm_id = "control"
    await router.handle_frontend(
        "comm_open",
        {
            "header": {"msg_id": "open"},
            "content": {
                "comm_id": comm_id,
                "target_name": "jupyter.widget.control",
                "data": {},
            },
        },
    )
    await router.handle_frontend(
        "comm_msg",
        {
            "header": {"msg_id": "request"},
            "content": {"comm_id": comm_id, "data": {"method": "request_states"}},
        },
    )
    await router.handle_rank(
        0,
        {
            "msg_type": "comm_msg",
            "content": {
                "comm_id": comm_id,
                "data": {
                    "method": "update_states",
                    "states": {"rank-zero-model": {"state": {}}},
                    "buffer_paths": [],
                },
            },
            "parent_header": {"msg_id": "request"},
        },
    )

    assert kernel.session.messages == []

    router.handle_rank_failure(1, RuntimeError("process exited"))

    assert len(kernel.session.messages) == 1
    content = kernel.session.messages[0][2]
    assert content["data"]["states"] == {"rank-zero-model": {"state": {}}}
    assert content["data"]["jupyter_distributed_missing_ranks"] == [1]
