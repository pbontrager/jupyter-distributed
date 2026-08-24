"""Multiplex Jupyter debug protocol messages across rank kernels."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .kernel_proxy import SPMDKernel

_BROADCAST_COMMANDS = {
    "initialize",
    "attach",
    "configurationDone",
    "disconnect",
    "setDataBreakpoints",
    "setExceptionBreakpoints",
    "setFunctionBreakpoints",
    "setInstructionBreakpoints",
}
_EXECUTION_COMMANDS = {"continue", "next", "pause", "stepIn", "stepOut"}
_REFERENCE_ARGUMENTS = {
    "threadId": "thread",
    "frameId": "frame",
    "variablesReference": "variable",
    "sourceReference": "source",
    "srcFrameId": "frame",
}


class DistributedDebugger:
    """Present multiple child debug adapters as one logical debugger."""

    def __init__(self, kernel: SPMDKernel) -> None:
        self._kernel = kernel
        self._available: bool | None = None
        self._active_rank = 0
        self._stopped_threads: dict[int, set[int]] = {}
        self._references: dict[str, dict[tuple[int, int], int]] = {
            kind: {} for kind in {"thread", "frame", "variable", "source"}
        }
        self._reverse_references: dict[str, dict[int, tuple[int, int]]] = {
            kind: {} for kind in self._references
        }
        self._next_reference = 1
        self._source_paths: dict[str, dict[int, str]] = {}
        self._canonical_paths: dict[tuple[int, str], str] = {}
        self._event_sequence = 1
        self._initialized_event_sent = False

    def set_available(self, available: bool) -> None:
        self._available = available

    async def request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        command = str(message.get("command", ""))
        if not await self._is_available():
            return self._error(message, "The selected kernel does not support debugging.")

        if command == "debugInfo":
            return await self._debug_info(message)
        if command == "dumpCell":
            return await self._dump_cell(message)
        if command == "setBreakpoints":
            return await self._set_breakpoints(message)
        if command == "threads":
            return await self._threads(message)
        if command in _EXECUTION_COMMANDS:
            return await self._control_all(message)
        if command in _BROADCAST_COMMANDS:
            reply = await self._broadcast(message)
            if command == "disconnect":
                self._reset()
            return reply
        return await self._route_to_rank(message)

    def handle_event(self, rank: int, event: Mapping[str, Any]) -> None:
        forwarded = copy.deepcopy(dict(event))
        event_name = str(forwarded.get("event", ""))
        body = forwarded.setdefault("body", {})
        if not isinstance(body, dict):
            body = {}
            forwarded["body"] = body

        if event_name == "initialized":
            if self._initialized_event_sent:
                return
            self._initialized_event_sent = True
        elif event_name == "stopped":
            raw_thread = self._integer(body.get("threadId"), default=1)
            self._stopped_threads.setdefault(rank, set()).add(raw_thread)
            self._active_rank = rank
            body["threadId"] = self._reference("thread", rank, raw_thread)
            body["allThreadsStopped"] = len(self._stopped_threads) == self._kernel.group.world_size
            body["description"] = self._rank_description(rank, body.get("description"))
        elif event_name == "continued":
            raw_thread = self._integer(body.get("threadId"), default=1)
            if body.get("allThreadsContinued", False):
                self._stopped_threads.pop(rank, None)
            else:
                stopped = self._stopped_threads.get(rank)
                if stopped is not None:
                    stopped.discard(raw_thread)
                    if not stopped:
                        self._stopped_threads.pop(rank, None)
            body["threadId"] = self._reference("thread", rank, raw_thread)
            body["allThreadsContinued"] = not self._stopped_threads
        elif event_name == "thread":
            raw_thread = self._integer(body.get("threadId"), default=1)
            body["threadId"] = self._reference("thread", rank, raw_thread)
        elif event_name == "output":
            body["output"] = f"[Rank {rank}] {body.get('output', '')}"
        elif event_name == "terminated":
            self._stopped_threads.pop(rank, None)
            if rank != 0:
                return

        forwarded["seq"] = self._event_sequence
        self._event_sequence += 1
        self._kernel.send_response(
            self._kernel.iopub_socket,
            "debug_event",
            forwarded,
        )

    async def _is_available(self) -> bool:
        if self._available is None:
            info = await self._kernel.group.kernel_info()
            supported_features = info.get("supported_features", ())
            self._available = bool(info.get("debugger", False)) or (
                isinstance(supported_features, (list, tuple)) and "debugger" in supported_features
            )
        return self._available

    async def _debug_info(self, message: Mapping[str, Any]) -> dict[str, Any]:
        replies = await self._broadcast_replies(message)
        reply = self._first_reply(replies, message)
        body = reply.setdefault("body", {})
        if isinstance(body, dict):
            body["stoppedThreads"] = [
                self._reference("thread", rank, thread)
                for rank, threads in sorted(self._stopped_threads.items())
                for thread in sorted(threads)
            ]
        return reply

    async def _dump_cell(self, message: Mapping[str, Any]) -> dict[str, Any]:
        replies = await self._broadcast_replies(message)
        reply = self._first_reply(replies, message)
        canonical = self._source_path(reply)
        if canonical is not None:
            paths: dict[int, str] = {}
            for rank, candidate in replies.items():
                path = self._source_path(candidate)
                if path is not None:
                    paths[rank] = path
                    self._canonical_paths[(rank, path)] = canonical
            self._source_paths[canonical] = paths
        return reply

    async def _set_breakpoints(self, message: Mapping[str, Any]) -> dict[str, Any]:
        requests: dict[int, Mapping[str, Any]] = {}
        for rank in self._rank_numbers():
            request = copy.deepcopy(dict(message))
            arguments = request.setdefault("arguments", {})
            if isinstance(arguments, dict):
                source = arguments.get("source")
                if isinstance(source, dict) and isinstance(source.get("path"), str):
                    source["path"] = self._path_for_rank(source["path"], rank)
            requests[rank] = request
        replies = await self._kernel.group.debug(requests)
        return self._first_reply(replies, message)

    async def _threads(self, message: Mapping[str, Any]) -> dict[str, Any]:
        replies = await self._broadcast_replies(message)
        threads: list[dict[str, Any]] = []
        for rank, reply in sorted(replies.items()):
            body = reply.get("body", {})
            if not isinstance(body, Mapping):
                continue
            for thread in body.get("threads", []):
                if not isinstance(thread, Mapping):
                    continue
                item = dict(thread)
                raw_id = self._integer(item.get("id"), default=1)
                item["id"] = self._reference("thread", rank, raw_id)
                item["name"] = self._rank_description(rank, item.get("name"))
                threads.append(item)
        reply = self._success(message)
        reply["body"] = {"threads": threads}
        return reply

    async def _control_all(self, message: Mapping[str, Any]) -> dict[str, Any]:
        command = str(message.get("command", ""))
        requests: dict[int, Mapping[str, Any]] = {}
        for rank in self._rank_numbers():
            raw_thread = self._control_thread(rank, message)
            if raw_thread is None and command != "pause":
                continue
            request = copy.deepcopy(dict(message))
            arguments = request.setdefault("arguments", {})
            if isinstance(arguments, dict):
                arguments["threadId"] = raw_thread or 1
            requests[rank] = request
        if not requests:
            return self._success(message)
        replies = await self._kernel.group.debug(requests)
        reply = self._first_reply(replies, message)
        if command == "continue":
            body = reply.setdefault("body", {})
            if isinstance(body, dict):
                body["allThreadsContinued"] = True
        return reply

    async def _route_to_rank(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request = copy.deepcopy(dict(message))
        arguments = request.setdefault("arguments", {})
        rank = self._active_rank
        if isinstance(arguments, dict):
            for name, kind in _REFERENCE_ARGUMENTS.items():
                value = arguments.get(name)
                if not isinstance(value, int) or value == 0:
                    continue
                resolved = self._reverse_references[kind].get(value)
                if resolved is not None:
                    rank, arguments[name] = resolved
                    self._active_rank = rank
                    break
            source = arguments.get("source")
            if isinstance(source, dict) and isinstance(source.get("path"), str):
                source["path"] = self._path_for_rank(source["path"], rank)
        replies = await self._kernel.group.debug({rank: request})
        return self._rewrite_reply(rank, dict(replies[rank]))

    async def _broadcast(self, message: Mapping[str, Any]) -> dict[str, Any]:
        replies = await self._broadcast_replies(message)
        return self._first_reply(replies, message)

    async def _broadcast_replies(self, message: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
        return await self._kernel.group.debug(
            {rank: copy.deepcopy(dict(message)) for rank in self._rank_numbers()}
        )

    def _first_reply(
        self,
        replies: Mapping[int, Mapping[str, Any]],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not replies:
            return self._error(request, "No rank kernels are available.")
        rank = min(replies)
        return self._rewrite_reply(rank, dict(replies[rank]))

    def _rewrite_reply(self, rank: int, reply: dict[str, Any]) -> dict[str, Any]:
        command = str(reply.get("command", ""))
        body = reply.get("body")
        if not isinstance(body, dict):
            return reply

        if command == "threads":
            for thread in body.get("threads", []):
                if isinstance(thread, dict):
                    thread["id"] = self._reference(
                        "thread", rank, self._integer(thread.get("id"), default=1)
                    )
                    thread["name"] = self._rank_description(rank, thread.get("name"))
        elif command == "stackTrace":
            for frame in body.get("stackFrames", []):
                if not isinstance(frame, dict):
                    continue
                frame["id"] = self._reference(
                    "frame", rank, self._integer(frame.get("id"), default=0)
                )
                frame["name"] = self._rank_description(rank, frame.get("name"))
                self._rewrite_source(rank, frame.get("source"))
        elif command == "scopes":
            for scope in body.get("scopes", []):
                if isinstance(scope, dict):
                    scope["variablesReference"] = self._reference(
                        "variable",
                        rank,
                        self._integer(scope.get("variablesReference"), default=0),
                    )
        elif command == "variables":
            self._rewrite_variables(rank, body.get("variables", []))
        elif command in {"evaluate", "setExpression", "setVariable"}:
            reference = self._integer(body.get("variablesReference"), default=0)
            body["variablesReference"] = self._reference("variable", rank, reference)
        return reply

    def _rewrite_variables(self, rank: int, variables: Any) -> None:
        if not isinstance(variables, list):
            return
        for variable in variables:
            if not isinstance(variable, dict):
                continue
            reference = self._integer(variable.get("variablesReference"), default=0)
            variable["variablesReference"] = self._reference("variable", rank, reference)

    def _rewrite_source(self, rank: int, source: Any) -> None:
        if not isinstance(source, dict):
            return
        path = source.get("path")
        if isinstance(path, str):
            source["path"] = self._canonical_paths.get((rank, path), path)
        reference = self._integer(source.get("sourceReference"), default=0)
        source["sourceReference"] = self._reference("source", rank, reference)

    def _control_thread(self, rank: int, message: Mapping[str, Any]) -> int | None:
        arguments = message.get("arguments", {})
        if isinstance(arguments, Mapping) and isinstance(arguments.get("threadId"), int):
            resolved = self._reverse_references["thread"].get(arguments["threadId"])
            if resolved is not None and resolved[0] == rank:
                return resolved[1]
        stopped = self._stopped_threads.get(rank, set())
        return min(stopped) if stopped else None

    def _reference(self, kind: str, rank: int, raw: int) -> int:
        if raw == 0:
            return 0
        key = (rank, raw)
        existing = self._references[kind].get(key)
        if existing is not None:
            return existing
        reference = self._next_reference
        self._next_reference += 1
        self._references[kind][key] = reference
        self._reverse_references[kind][reference] = key
        return reference

    def _path_for_rank(self, canonical: str, rank: int) -> str:
        return self._source_paths.get(canonical, {}).get(rank, canonical)

    def _rank_numbers(self) -> range:
        return range(self._kernel.group.world_size)

    def _reset(self) -> None:
        self._active_rank = 0
        self._stopped_threads.clear()
        self._references = {kind: {} for kind in self._references}
        self._reverse_references = {kind: {} for kind in self._reverse_references}
        self._next_reference = 1
        self._source_paths.clear()
        self._canonical_paths.clear()
        self._initialized_event_sent = False

    @staticmethod
    def _source_path(reply: Mapping[str, Any]) -> str | None:
        body = reply.get("body", {})
        if not isinstance(body, Mapping):
            return None
        path = body.get("sourcePath")
        return str(path) if path else None

    @staticmethod
    def _rank_description(rank: int, description: Any) -> str:
        suffix = f": {description}" if description else ""
        return f"Rank {rank}{suffix}"

    @staticmethod
    def _integer(value: Any, *, default: int) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else default

    @staticmethod
    def _success(request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "response",
            "request_seq": request.get("seq", 0),
            "success": True,
            "command": str(request.get("command", "")),
        }

    @classmethod
    def _error(cls, request: Mapping[str, Any], message: str) -> dict[str, Any]:
        return {**cls._success(request), "success": False, "message": message}


__all__ = ["DistributedDebugger"]
