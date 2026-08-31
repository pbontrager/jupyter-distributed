"""Route Jupyter comm messages between the frontend and rank kernels."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .kernel_proxy import SPMDKernel

_COMM_TYPES = {"comm_open", "comm_msg", "comm_close"}
_WIDGET_CONTROL_TARGET = "jupyter.widget.control"


@dataclass(slots=True)
class _CommRecord:
    target_name: str
    ranks: set[int] = field(default_factory=set)
    frontend_opened: bool = False


@dataclass(slots=True)
class _WidgetStateRequest:
    expected: set[int]
    replies: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    missing: set[int] = field(default_factory=set)
    request: Mapping[str, Any] = field(default_factory=dict)


class DistributedCommRouter:
    """Expose child-kernel comms as one stable, bidirectional comm namespace.

    Kernel-created comm IDs are globally unique in practice, so they remain
    untouched. Frontend-created comms are broadcast with the same ID to each
    independent child kernel. This preserves nested comm references such as
    ``IPY_MODEL_<id>`` without coupling the router to a particular widget.
    """

    def __init__(self, kernel: SPMDKernel) -> None:
        self._kernel = kernel
        self._comms: dict[str, _CommRecord] = {}
        self._widget_state_requests: dict[str, _WidgetStateRequest] = {}

    async def handle_frontend(
        self,
        message_type: str,
        message: Mapping[str, Any],
    ) -> None:
        """Route one shell-channel comm message to its child rank(s)."""

        if message_type not in _COMM_TYPES:
            raise ValueError(f"unsupported comm message type: {message_type}")
        content = _mapping(message.get("content"))
        comm_id = str(content.get("comm_id", ""))
        if not comm_id:
            return

        record = self._comms.get(comm_id)
        if message_type == "comm_open":
            if record is None:
                record = _CommRecord(
                    target_name=str(content.get("target_name", "")),
                    ranks=set(self._rank_numbers()),
                    frontend_opened=True,
                )
                self._comms[comm_id] = record
        elif record is None:
            self._kernel.log.warning("Ignoring message for unknown comm %s", comm_id)
            return

        assert record is not None
        if (
            message_type == "comm_msg"
            and record.target_name == _WIDGET_CONTROL_TARGET
            and _mapping(content.get("data")).get("method") == "request_states"
        ):
            self._widget_state_requests[comm_id] = _WidgetStateRequest(
                expected=set(record.ranks), request=copy.deepcopy(dict(message))
            )

        await self._kernel.group.send_comm(
            sorted(record.ranks),
            message_type,
            content,
            metadata=_mapping(message.get("metadata")),
            buffers=_buffers(message),
            parent=message,
        )
        if message_type == "comm_close":
            self._comms.pop(comm_id, None)
            self._widget_state_requests.pop(comm_id, None)

    async def handle_rank(self, rank: int, message: Mapping[str, Any]) -> None:
        """Route one child IOPub comm event to the notebook frontend."""

        message_type = str(message.get("msg_type", ""))
        if message_type not in _COMM_TYPES:
            return
        content = _mapping(message.get("content"))
        comm_id = str(content.get("comm_id", ""))
        if not comm_id:
            return

        record = self._comms.get(comm_id)
        if message_type == "comm_open":
            target_name = str(content.get("target_name", ""))
            if record is None:
                self._comms[comm_id] = _CommRecord(target_name=target_name, ranks={rank})
            elif rank not in record.ranks:
                self._kernel.log.warning(
                    "Ignoring duplicate child comm ID %s from rank %d", comm_id, rank
                )
                return
            self._publish(message_type, content, message)
            return

        if record is None:
            self._kernel.log.warning(
                "Ignoring %s for unregistered child comm %s from rank %d",
                message_type,
                comm_id,
                rank,
            )
            return

        if message_type == "comm_msg" and self._collect_widget_state(
            rank, comm_id, record, content, message
        ):
            return

        if message_type == "comm_close":
            record.ranks.discard(rank)
            pending = self._widget_state_requests.get(comm_id)
            if pending is not None:
                pending.expected.discard(rank)
                pending.missing.add(rank)
                self._finish_widget_state(comm_id, pending)
            if record.frontend_opened and record.ranks:
                return
            self._comms.pop(comm_id, None)
            self._widget_state_requests.pop(comm_id, None)
        self._publish(message_type, content, message)

    def comm_info(self, target_name: str | None = None) -> dict[str, dict[str, str]]:
        """Return the logical comm registry used by Jupyter reconnect flows."""

        return {
            comm_id: {"target_name": record.target_name}
            for comm_id, record in self._comms.items()
            if target_name is None or record.target_name == target_name
        }

    def reset(self) -> None:
        self._comms.clear()
        self._widget_state_requests.clear()

    def handle_rank_failure(self, rank: int, reason: BaseException) -> None:
        """Remove a failed rank from comm ownership and finish partial state requests."""

        for record in self._comms.values():
            record.ranks.discard(rank)
        for comm_id, pending in tuple(self._widget_state_requests.items()):
            if rank in pending.expected:
                pending.expected.discard(rank)
                pending.missing.add(rank)
                self._finish_widget_state(comm_id, pending)
        self._kernel.log.warning("Rank %d comm endpoint failed: %s", rank, reason)

    def _collect_widget_state(
        self,
        rank: int,
        comm_id: str,
        record: _CommRecord,
        content: Mapping[str, Any],
        message: Mapping[str, Any],
    ) -> bool:
        if record.target_name != _WIDGET_CONTROL_TARGET:
            return False
        data = _mapping(content.get("data"))
        if data.get("method") != "update_states":
            return False
        pending = self._widget_state_requests.get(comm_id)
        if pending is None:
            return False

        pending.replies[rank] = message
        if not pending.expected.issubset(pending.replies):
            return True

        self._finish_widget_state(comm_id, pending)
        return True

    def _finish_widget_state(self, comm_id: str, pending: _WidgetStateRequest) -> None:
        if not pending.expected.issubset(pending.replies):
            return

        states: dict[str, Any] = {}
        buffer_paths: list[Any] = []
        buffers: list[Any] = []
        if pending.replies:
            first = pending.replies[min(pending.replies)]
            merged_content = copy.deepcopy(dict(_mapping(first.get("content"))))
            merged_data = copy.deepcopy(dict(_mapping(merged_content.get("data"))))
        else:
            request_header = dict(_mapping(pending.request.get("header")))
            first = {
                "parent_header": request_header,
                "metadata": dict(_mapping(pending.request.get("metadata"))),
            }
            merged_content = {"comm_id": comm_id}
            merged_data = {"method": "update_states"}
        for response_rank in sorted(pending.replies):
            response = pending.replies[response_rank]
            response_data = _mapping(_mapping(response.get("content")).get("data"))
            for model_id, state in _mapping(response_data.get("states")).items():
                if model_id in states:
                    self._kernel.log.warning(
                        "Duplicate widget model ID %s reported by rank %d",
                        model_id,
                        response_rank,
                    )
                    continue
                states[str(model_id)] = copy.deepcopy(state)
            paths = response_data.get("buffer_paths", ())
            if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes, bytearray)):
                buffer_paths.extend(copy.deepcopy(list(paths)))
            buffers.extend(_buffers(response))

        merged_data["states"] = states
        merged_data["buffer_paths"] = buffer_paths
        if pending.missing:
            merged_data["jupyter_distributed_missing_ranks"] = sorted(pending.missing)
        merged_content["data"] = merged_data
        self._publish("comm_msg", merged_content, first, buffers=buffers)
        self._widget_state_requests.pop(comm_id, None)

    def _publish(
        self,
        message_type: str,
        content: Mapping[str, Any],
        source: Mapping[str, Any],
        *,
        buffers: Sequence[Any] | None = None,
    ) -> None:
        if self._kernel.session is None:
            return
        self._kernel.session.send(
            self._kernel.iopub_socket,
            message_type,
            dict(content),
            parent=dict(_mapping(source.get("parent_header"))),
            buffers=list(_buffers(source) if buffers is None else buffers),
            metadata=dict(_mapping(source.get("metadata"))),
        )

    def _rank_numbers(self) -> range:
        return range(self._kernel.group.world_size)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _buffers(message: Mapping[str, Any]) -> list[Any]:
    buffers = message.get("buffers", ())
    if isinstance(buffers, Sequence) and not isinstance(buffers, (str, bytes, bytearray)):
        return list(buffers)
    return []
