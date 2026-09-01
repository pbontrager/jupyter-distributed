from __future__ import annotations

from typing import Any

from jupyter_distributed.kernel_proxy import (
    RANK_UPDATE_COMM_TARGET,
    _append_output_patch,
    _LiveRankUpdates,
)
from jupyter_distributed.protocol import RankOutputPatch


class RecordingKernel:
    def __init__(self) -> None:
        self.iopub_socket = object()
        self.messages: list[dict[str, Any]] = []

    def send_response(
        self,
        socket: object,
        message_type: str,
        content: dict[str, Any],
    ) -> None:
        assert socket is self.iopub_socket
        assert message_type == "comm_msg"
        self.messages.append(content)


def _snapshot(text: str) -> dict[str, Any]:
    return {
        "execution_id": "execution",
        "execution_count": 1,
        "status": "busy",
        "world_size": 1,
        "ranks": [
            {
                "rank": 0,
                "status": "running",
                "outputs": [
                    {
                        "rank": 0,
                        "type": "stream",
                        "content": {"name": "stdout", "text": text},
                    }
                ],
            }
        ],
    }


def _patch(text: str) -> dict[str, Any]:
    return {
        "execution_id": "execution",
        "execution_count": 1,
        "status": "busy",
        "world_size": 1,
        "rank_updates": [
            {
                "rank": 0,
                "patches": [{"kind": "append_stream", "index": 0, "text": text}],
            }
        ],
    }


def test_long_stream_is_coalesced_into_one_bounded_interval_patch() -> None:
    pending: list[RankOutputPatch] = []

    for _ in range(10_000):
        _append_output_patch(
            pending,
            RankOutputPatch("append_stream", index=0, text="x"),
        )

    assert len(pending) == 1
    assert pending[0].kind == "append_stream"
    assert pending[0].index == 0
    assert pending[0].text == "x" * 10_000


def test_patches_are_incremental_and_snapshot_recovery_is_current() -> None:
    kernel = RecordingKernel()
    updates = _LiveRankUpdates(kernel)  # type: ignore[arg-type]
    comm_id = "frontend"
    assert updates.handle_frontend(
        "comm_open",
        {
            "content": {
                "comm_id": comm_id,
                "target_name": RANK_UPDATE_COMM_TARGET,
            }
        },
    )
    kernel.messages.clear()

    updates.publish_snapshot(_snapshot("a"))
    updates.publish_patch(_patch("b"))
    updates.publish_patch(_patch("c"))

    assert [message["data"]["kind"] for message in kernel.messages] == [
        "snapshot",
        "patch",
        "patch",
    ]
    assert [message["data"]["sequence"] for message in kernel.messages] == [1, 2, 3]
    assert kernel.messages[1]["data"]["payload"] == _patch("b")
    assert kernel.messages[2]["data"]["payload"] == _patch("c")

    assert updates.handle_frontend(
        "comm_msg",
        {
            "content": {
                "comm_id": comm_id,
                "data": {"method": "request_snapshots"},
            }
        },
    )

    assert len(kernel.messages) == 4
    recovery = kernel.messages[3]["data"]
    assert recovery["method"] == "snapshots"
    assert recovery["snapshots"][0]["sequence"] == 3
    output = recovery["snapshots"][0]["payload"]["ranks"][0]["outputs"][0]
    assert "".join(output["content"]["text"]) == "abc"
