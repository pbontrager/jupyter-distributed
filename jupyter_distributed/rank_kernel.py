"""Control-plane wrapper around one ordinary Jupyter kernel process."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from jupyter_client import AsyncKernelManager
from jupyter_client.asynchronous.client import AsyncKernelClient

from .protocol import RankExecution, RankOutput

_OUTPUT_TYPES = {"stream", "display_data", "execute_result", "error"}
OutputCallback = Callable[[int, tuple[RankOutput, ...]], Awaitable[None] | None]
DebugEventCallback = Callable[[int, Mapping[str, Any]], Awaitable[None] | None]


class RankKernel:
    """One persistent child kernel and its private Jupyter channels."""

    def __init__(
        self,
        rank: int,
        env: Mapping[str, str],
        *,
        kernel_name: str = "python3",
        cwd: str | None = None,
        ready_timeout: float = 30.0,
        on_debug_event: DebugEventCallback | None = None,
    ) -> None:
        self.rank = rank
        self.env = dict(env)
        self.cwd = cwd
        self.ready_timeout = ready_timeout
        self.on_debug_event = on_debug_event
        self.manager = AsyncKernelManager(kernel_name=kernel_name)
        self.client: AsyncKernelClient | None = None
        self._iopub_queues: dict[str, asyncio.Queue[Mapping[str, Any]]] = {}
        self._iopub_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.manager.start_kernel(env=self.env, cwd=self.cwd)
        client = self.manager.client()
        self.client = client
        client.start_channels()
        try:
            await client.wait_for_ready(timeout=self.ready_timeout)
            self._iopub_task = asyncio.create_task(self._route_iopub())
        except BaseException:
            await self.shutdown(now=True)
            raise

    async def execute(
        self,
        code: str,
        *,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Mapping[str, Any] | None = None,
        on_output: OutputCallback | None = None,
    ) -> RankExecution:
        client = self._client()
        message_id = client.execute(
            code,
            silent=silent,
            store_history=store_history,
            user_expressions=dict(user_expressions or {}),
            allow_stdin=False,
            stop_on_error=False,
        )
        iopub_queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self._iopub_queues[message_id] = iopub_queue
        reply_task = asyncio.create_task(self._shell_reply(message_id))
        output_buffer = _RankOutputBuffer(self.rank)
        try:
            while True:
                message = await iopub_queue.get()
                message_type = message.get("msg_type")
                if (
                    output_buffer.handle(str(message_type), message.get("content", {}))
                    and on_output is not None
                ):
                    notified = on_output(self.rank, output_buffer.snapshot())
                    if inspect.isawaitable(notified):
                        await notified
                if (
                    message_type == "status"
                    and message.get("content", {}).get("execution_state") == "idle"
                ):
                    break
            reply = await reply_task
        except BaseException:
            reply_task.cancel()
            await asyncio.gather(reply_task, return_exceptions=True)
            raise
        finally:
            self._iopub_queues.pop(message_id, None)
        status = reply.get("content", {}).get("status", "error")
        if status not in {"ok", "error", "aborted"}:
            status = "error"
        return RankExecution(
            rank=self.rank,
            status=status,
            outputs=output_buffer.snapshot(),
            reply=reply.get("content", {}),
        )

    async def request(self, message_type: str, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Send a simple shell request and return its matching content."""

        client = self._client()
        sender = getattr(client, message_type)
        message_id = sender(*args, **kwargs)
        reply = await self._shell_reply(message_id)
        return reply.get("content", {})

    async def debug_request(self, content: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send one Jupyter debug request over the child control channel."""

        client = self._client()
        request = client.session.msg("debug_request", content=dict(content))
        message_id = str(request["header"]["msg_id"])
        client.control_channel.send(request)
        while True:
            reply = await client.get_control_msg(timeout=None)
            if reply.get("parent_header", {}).get("msg_id") == message_id:
                return reply.get("content", {})

    async def _shell_reply(self, message_id: str) -> Mapping[str, Any]:
        client = self._client()
        while True:
            reply = await client.get_shell_msg(timeout=None)
            if reply.get("parent_header", {}).get("msg_id") == message_id:
                return reply

    async def interrupt(self) -> None:
        if await self.is_alive():
            await self.manager.interrupt_kernel()

    async def shutdown(self, *, now: bool = False) -> None:
        client, self.client = self.client, None
        iopub_task, self._iopub_task = self._iopub_task, None
        if iopub_task is not None:
            iopub_task.cancel()
            await asyncio.gather(iopub_task, return_exceptions=True)
        try:
            if self.manager.has_kernel:
                await self.manager.shutdown_kernel(now=now)
        finally:
            if client is not None:
                client.stop_channels()

    async def is_alive(self) -> bool:
        return bool(self.manager.has_kernel and await self.manager.is_alive())

    def _client(self) -> AsyncKernelClient:
        if self.client is None:
            raise RuntimeError(f"rank {self.rank} is not running")
        return self.client

    async def _route_iopub(self) -> None:
        client = self._client()
        while True:
            message = await client.get_iopub_msg(timeout=None)
            if message.get("msg_type") == "debug_event":
                if self.on_debug_event is not None:
                    notified = self.on_debug_event(self.rank, message.get("content", {}))
                    if inspect.isawaitable(notified):
                        await notified
                continue
            message_id = message.get("parent_header", {}).get("msg_id")
            queue = self._iopub_queues.get(str(message_id))
            if queue is not None:
                queue.put_nowait(message)


class _RankOutputBuffer:
    """Apply Jupyter output-area semantics to one rank's IOPub messages."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._outputs: list[RankOutput] = []
        self._display_ids: dict[str, list[int]] = {}
        self._clear_next = False
        self._stream_index = 0

    def snapshot(self) -> tuple[RankOutput, ...]:
        return tuple(self._outputs)

    def handle(self, message_type: str, content: Mapping[str, Any]) -> bool:
        if message_type == "clear_output":
            if bool(content.get("wait", False)):
                self._clear_next = True
            else:
                self._clear()
            return not self._clear_next

        if message_type == "update_display_data":
            return self._update_display(content)

        if message_type not in _OUTPUT_TYPES:
            return False

        if self._clear_next:
            self._clear()
        self._append(message_type, content)
        return True

    def _append(self, message_type: str, content: Mapping[str, Any]) -> None:
        if message_type == "stream":
            name = str(content.get("name", "stdout"))
            text = _as_text(content.get("text", ""))
            if (
                self._outputs
                and self._outputs[-1].kind == "stream"
                and self._outputs[-1].content.get("name") == name
            ):
                previous = self._outputs[-1]
                previous_text = _as_text(previous.content.get("text", ""))
                text, self._stream_index = _process_stream_text(
                    self._stream_index, text, previous_text
                )
                self._outputs[-1] = RankOutput(
                    rank=self.rank,
                    kind="stream",
                    content={**previous.content, "text": text},
                )
                return
            text, self._stream_index = _process_stream_text(0, text)
            content = {**content, "name": name, "text": text}
        else:
            self._stream_index = 0

        output = RankOutput(
            rank=self.rank,
            kind=message_type,
            content=dict(content),
        )
        self._outputs.append(output)
        if message_type == "display_data":
            display_id = _display_id(content)
            if display_id is not None:
                self._display_ids.setdefault(display_id, []).append(len(self._outputs) - 1)

    def _update_display(self, content: Mapping[str, Any]) -> bool:
        display_id = _display_id(content)
        if display_id is None:
            return False
        indices = self._display_ids.get(display_id, [])
        for index in indices:
            self._outputs[index] = RankOutput(
                rank=self.rank,
                kind="display_data",
                content=dict(content),
            )
        return bool(indices)

    def _clear(self) -> None:
        self._outputs.clear()
        self._display_ids.clear()
        self._clear_next = False
        self._stream_index = 0


def _display_id(content: Mapping[str, Any]) -> str | None:
    transient = content.get("transient")
    if not isinstance(transient, Mapping):
        return None
    display_id = transient.get("display_id")
    return str(display_id) if display_id else None


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def _process_stream_text(index: int, new_text: str, text: str = "") -> tuple[str, int]:
    """Apply terminal carriage-return and backspace behavior to stream text."""

    for character in new_text:
        if character == "\b":
            if index > 0 and text[index - 1] != "\n":
                text = text[: index - 1] + text[index:]
                index -= 1
        elif character == "\r":
            index = text.rfind("\n", 0, index) + 1
        elif character == "\n":
            text += "\n"
            index = len(text)
        else:
            text = text[:index] + character + text[index + 1 :]
            index += 1
    return text, index
