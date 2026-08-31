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
CommEventCallback = Callable[[int, Mapping[str, Any]], Awaitable[None] | None]
FailureCallback = Callable[[int, BaseException], Awaitable[None] | None]


class RankKernelFailure(RuntimeError):
    """A child kernel or one of its channel routers stopped unexpectedly."""

    def __init__(
        self,
        rank: int,
        reason: BaseException,
        outputs: tuple[RankOutput, ...] = (),
    ) -> None:
        super().__init__(f"rank {rank} failed: {reason}")
        self.rank = rank
        self.reason = reason
        self.outputs = outputs


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
        on_comm_event: CommEventCallback | None = None,
        on_failure: FailureCallback | None = None,
    ) -> None:
        self.rank = rank
        self.env = dict(env)
        self.cwd = cwd
        self.ready_timeout = ready_timeout
        self.on_debug_event = on_debug_event
        self.on_comm_event = on_comm_event
        self.on_failure = on_failure
        self.manager = AsyncKernelManager(kernel_name=kernel_name)
        self.client: AsyncKernelClient | None = None
        self._shell_lock = asyncio.Lock()
        self._iopub_queues: dict[str, asyncio.Queue[Mapping[str, Any]]] = {}
        self._iopub_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._failure: asyncio.Future[BaseException] | None = None
        self._display_buffers: dict[str, dict[_RankOutputBuffer, OutputCallback | None]] = {}
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        self._failure = asyncio.get_running_loop().create_future()
        await self.manager.start_kernel(env=self.env, cwd=self.cwd)
        client = self.manager.client()
        self.client = client
        client.start_channels()
        try:
            await client.wait_for_ready(timeout=self.ready_timeout)
            self._iopub_task = asyncio.create_task(self._route_iopub())
            self._monitor_task = asyncio.create_task(self._monitor_process())
            await self._install_breakpoint_hook()
        except BaseException:
            await self.shutdown(now=True)
            raise

    async def _install_breakpoint_hook(self) -> None:
        if (
            int(self.env["WORLD_SIZE"]) <= 1
            or str(self.manager.kernel_spec.language).lower() != "python"
        ):
            return
        await self.execute(
            "import sys\n"
            "from jupyter_distributed.breakpoint import distributed_breakpoint\n"
            "sys.breakpointhook = distributed_breakpoint",
            silent=True,
            store_history=False,
        )

    async def execute(
        self,
        code: str,
        *,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Mapping[str, Any] | None = None,
        on_output: OutputCallback | None = None,
    ) -> RankExecution:
        async with self._shell_lock:
            return await self._execute_unlocked(
                code,
                silent=silent,
                store_history=store_history,
                user_expressions=user_expressions,
                on_output=on_output,
            )

    async def _execute_unlocked(
        self,
        code: str,
        *,
        silent: bool,
        store_history: bool,
        user_expressions: Mapping[str, Any] | None,
        on_output: OutputCallback | None,
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
                try:
                    message = await self._wait_or_failure(iopub_queue.get())
                except RankKernelFailure as error:
                    raise RankKernelFailure(
                        self.rank, error.reason, output_buffer.snapshot()
                    ) from error
                message_type = message.get("msg_type")
                content = message.get("content", {})
                if message_type == "update_display_data":
                    await self._update_displays(content)
                    changed = False
                else:
                    changed = output_buffer.handle(str(message_type), content)
                    self._sync_display_buffers(output_buffer, on_output)
                if changed and on_output is not None:
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

        async with self._shell_lock:
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
            reply = await self._wait_or_failure(client.get_control_msg(timeout=None))
            if reply.get("parent_header", {}).get("msg_id") == message_id:
                return reply.get("content", {})

    def send_comm(
        self,
        message_type: str,
        content: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        buffers: list[Any] | None = None,
        parent: Mapping[str, Any] | None = None,
    ) -> None:
        """Send an opaque comm message to this child's shell channel."""

        client = self._client()
        source = dict(parent or {})
        header = dict(source.get("header", {}))
        if not header:
            header = dict(client.session.msg(message_type)["header"])
        header["msg_type"] = message_type
        message = {
            "header": header,
            "parent_header": dict(source.get("parent_header", {})),
            "metadata": dict(metadata or {}),
            "content": dict(content),
        }
        message["buffers"] = list(buffers or [])
        client.shell_channel.send(message)

    async def _shell_reply(self, message_id: str) -> Mapping[str, Any]:
        client = self._client()
        while True:
            reply = await self._wait_or_failure(client.get_shell_msg(timeout=None))
            if reply.get("parent_header", {}).get("msg_id") == message_id:
                return reply

    async def interrupt(self) -> None:
        if await self.is_alive():
            await self.manager.interrupt_kernel()

    async def shutdown(self, *, now: bool = False) -> None:
        self._stopping = True
        self._display_buffers.clear()
        client, self.client = self.client, None
        iopub_task, self._iopub_task = self._iopub_task, None
        monitor_task, self._monitor_task = self._monitor_task, None
        tasks = [task for task in (iopub_task, monitor_task) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            if self.manager.has_kernel:
                await self.manager.shutdown_kernel(now=now)
        finally:
            if client is not None:
                client.stop_channels()

    async def is_alive(self) -> bool:
        return bool(self.manager.has_kernel and await self.manager.is_alive())

    @property
    def failure_reason(self) -> BaseException | None:
        if self._failure is not None and self._failure.done():
            return self._failure.result()
        return None

    def _client(self) -> AsyncKernelClient:
        if self.client is None:
            raise RuntimeError(f"rank {self.rank} is not running")
        return self.client

    async def _route_iopub(self) -> None:
        client = self._client()
        try:
            while True:
                message = await client.get_iopub_msg(timeout=None)
                if message.get("msg_type") == "debug_event":
                    if self.on_debug_event is not None:
                        notified = self.on_debug_event(self.rank, message.get("content", {}))
                        if inspect.isawaitable(notified):
                            await notified
                    continue
                if message.get("msg_type") in {"comm_open", "comm_msg", "comm_close"}:
                    if self.on_comm_event is not None:
                        notified = self.on_comm_event(self.rank, message)
                        if inspect.isawaitable(notified):
                            await notified
                    continue
                message_id = message.get("parent_header", {}).get("msg_id")
                queue = self._iopub_queues.get(str(message_id))
                if queue is not None:
                    queue.put_nowait(message)
        except asyncio.CancelledError:
            if not self._stopping:
                self._record_failure(RuntimeError(f"rank {self.rank} IOPub router was cancelled"))
            raise
        except BaseException as error:
            self._record_failure(RuntimeError(f"rank {self.rank} IOPub router stopped: {error}"))

    async def _monitor_process(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.1)
                if not await self.is_alive():
                    self._record_failure(RuntimeError(f"rank {self.rank} process exited"))
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._record_failure(RuntimeError(f"rank {self.rank} process monitor failed: {error}"))

    async def _wait_or_failure(self, awaitable: Awaitable[Any]) -> Any:
        failure = self._failure
        if failure is None:
            failure = asyncio.get_running_loop().create_future()
            self._failure = failure
        operation = asyncio.ensure_future(awaitable)
        try:
            done, _pending = await asyncio.wait(
                {operation, failure}, return_when=asyncio.FIRST_COMPLETED
            )
            if failure in done:
                raise RankKernelFailure(self.rank, failure.result())
            return operation.result()
        finally:
            if not operation.done():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)

    def _record_failure(self, error: BaseException) -> None:
        if self._stopping or self._failure is None or self._failure.done():
            return
        self._failure.set_result(error)
        if self.on_failure is not None:
            notified = self.on_failure(self.rank, error)
            if inspect.isawaitable(notified):
                asyncio.create_task(notified)

    def _sync_display_buffers(
        self,
        buffer: _RankOutputBuffer,
        callback: OutputCallback | None,
    ) -> None:
        current = set(buffer.display_ids)
        for display_id, buffers in tuple(self._display_buffers.items()):
            if buffer in buffers and display_id not in current:
                buffers.pop(buffer, None)
                if not buffers:
                    self._display_buffers.pop(display_id, None)
        for display_id in current:
            self._display_buffers.setdefault(display_id, {})[buffer] = callback

    async def _update_displays(self, content: Mapping[str, Any]) -> None:
        display_id = _display_id(content)
        if display_id is None:
            return
        for buffer, callback in tuple(self._display_buffers.get(display_id, {}).items()):
            if not buffer.update_display(content) or callback is None:
                continue
            notified = callback(self.rank, buffer.snapshot())
            if inspect.isawaitable(notified):
                await notified


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

    @property
    def display_ids(self) -> tuple[str, ...]:
        return tuple(self._display_ids)

    def handle(self, message_type: str, content: Mapping[str, Any]) -> bool:
        if message_type == "clear_output":
            if bool(content.get("wait", False)):
                self._clear_next = True
            else:
                self._clear()
            return not self._clear_next

        if message_type == "update_display_data":
            return self.update_display(content)

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
        if message_type in {"display_data", "execute_result"}:
            display_id = _display_id(content)
            if display_id is not None:
                self._display_ids.setdefault(display_id, []).append(len(self._outputs) - 1)

    def update_display(self, content: Mapping[str, Any]) -> bool:
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
