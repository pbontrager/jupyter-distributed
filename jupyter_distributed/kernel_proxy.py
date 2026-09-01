"""Jupyter kernel facade for a :class:`DistributedKernelGroup`."""

from __future__ import annotations

import asyncio
import html
import os
import signal
import traceback
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from contextvars import ContextVar
from typing import Any, ClassVar
from uuid import uuid4

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel

from .comms import DistributedCommRouter
from .debugger import DistributedDebugger
from .kernel_group import DistributedKernelGroup
from .process_registry import ChildProcessRegistry
from .protocol import GroupExecution, RankExecution, RankOutput
from .rank_magic import RankMagicError, parse_rank_cell

RANK_MIME = "application/vnd.jupyter-distributed.rank+json"
RANK_UPDATE_COMM_TARGET = "jupyter.distributed.rank_updates"
_STREAM_UPDATE_INTERVAL = 0.05
_ON_KERNEL_IO_LOOP: ContextVar[bool] = ContextVar(
    "jupyter_distributed_on_kernel_io_loop", default=False
)


def render_plain(execution: GroupExecution, target_rank: int | None = None) -> str:
    sections: list[str] = []
    for result in _visible_ranks(execution, target_rank):
        body = "".join(output.plain_text() for output in result.outputs)
        sections.append(f"[Rank {result.rank} — {result.status}]\n{body}".rstrip())
    return "\n\n".join(sections)


def render_html(execution: GroupExecution, target_rank: int | None = None) -> str:
    sections: list[str] = ['<div class="jupyter-distributed-rank-output">']
    for index, result in enumerate(_visible_ranks(execution, target_rank)):
        opened = " open" if index == 0 else ""
        body = html.escape("".join(output.plain_text() for output in result.outputs))
        sections.append(
            f'<details data-rank="{result.rank}" data-status="{result.status}"{opened}>'
            f"<summary>Rank {result.rank} — {result.status}</summary>"
            f"<pre>{body}</pre></details>"
        )
    sections.append("</div>")
    return "".join(sections)


def _visible_ranks(execution: GroupExecution, target_rank: int | None) -> tuple[Any, ...]:
    if target_rank is None:
        return execution.ranks
    return tuple(result for result in execution.ranks if result.rank == target_rank)


class _LiveRankDisplay:
    """Publish durable output once and carry transient snapshots over a comm."""

    def __init__(
        self,
        kernel: SPMDKernel,
        execution_count: int,
        target_rank: int | None,
    ) -> None:
        self._kernel = kernel
        self._execution_count = execution_count
        self._execution_id = uuid4().hex
        self._display_id = f"jupyter-distributed-{self._execution_id}"
        self._target_rank = target_rank
        self._outputs: dict[int, tuple[RankOutput, ...]] = {
            rank: () for rank in range(kernel.group.world_size)
        }
        self._flush_task: asyncio.Task[None] | None = None
        self._last_sent = 0.0
        self._dirty = False
        self._started = False
        self._execution: GroupExecution | None = None

    def update(self, rank: int, outputs: tuple[RankOutput, ...]) -> None:
        self._outputs[rank] = outputs
        self._dirty = True
        if self._execution is not None:
            execution = self._updated_execution()
            payload = execution.as_dict()
            if self._target_rank is not None:
                payload["target_rank"] = self._target_rank
            self._kernel.live_updates.publish(self._snapshot(payload, execution=execution))
            self._send_display(
                "update_display_data",
                self._data(payload, execution=execution),
            )
            return
        if not self._started:
            if not any(self._outputs.values()):
                return
            self._started = True
            payload = self._live_payload()
            self._send_display("display_data", self._data(payload))
            self._kernel.live_updates.publish(self._snapshot(payload))
            return
        if self._flush_task is not None:
            return
        loop = asyncio.get_running_loop()
        delay = max(0.0, _STREAM_UPDATE_INTERVAL - (loop.time() - self._last_sent))
        self._flush_task = loop.create_task(self._flush_after(delay))

    async def finish(self, execution: GroupExecution) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            await asyncio.gather(self._flush_task, return_exceptions=True)
            self._flush_task = None
        if not self._started and not any(result.outputs for result in execution.ranks):
            return
        self._execution = execution
        self._outputs = {result.rank: result.outputs for result in execution.ranks}
        payload = execution.as_dict()
        payload["execution_id"] = self._execution_id
        if self._target_rank is not None:
            payload["target_rank"] = self._target_rank
        if self._started:
            self._kernel.live_updates.publish(self._snapshot(payload, execution=execution))
            self._send_display("update_display_data", self._data(payload, execution=execution))
        else:
            self._started = True
            self._send_display("display_data", self._data(payload, execution=execution))
            self._kernel.live_updates.publish(self._snapshot(payload, execution=execution))

    def _updated_execution(self) -> GroupExecution:
        assert self._execution is not None
        return GroupExecution(
            execution_count=self._execution.execution_count,
            execution_id=self._execution_id,
            ranks=tuple(
                RankExecution(
                    rank=result.rank,
                    status=result.status,
                    outputs=self._outputs[result.rank],
                    reply=result.reply,
                )
                for result in self._execution.ranks
            ),
        )

    async def _flush_after(self, delay: float) -> None:
        if delay:
            await asyncio.sleep(delay)
        self._flush_task = None
        if self._dirty:
            payload = self._live_payload()
            self._kernel.live_updates.publish(self._snapshot(payload))
            self._last_sent = asyncio.get_running_loop().time()
            self._dirty = False

    def _live_payload(self) -> dict[str, Any]:
        payload = {
            "execution_id": self._execution_id,
            "execution_count": self._execution_count,
            "status": "busy",
            "world_size": len(self._outputs),
            "ranks": [
                {
                    "rank": rank,
                    "status": "running",
                    "outputs": [output.as_dict() for output in outputs],
                }
                for rank, outputs in sorted(self._outputs.items())
            ],
        }
        if self._target_rank is not None:
            payload["target_rank"] = self._target_rank
        return payload

    def _snapshot(
        self,
        payload: dict[str, Any],
        *,
        execution: GroupExecution | None = None,
    ) -> dict[str, Any]:
        return {
            "execution_id": self._execution_id,
            "final": execution is not None,
            "data": self._data(payload, execution=execution),
            "metadata": {},
        }

    def _data(
        self,
        payload: dict[str, Any],
        *,
        execution: GroupExecution | None = None,
    ) -> dict[str, Any]:
        if execution is None:
            plain = "\n\n".join(
                f"[Rank {rank} — running]\n" + "".join(output.plain_text() for output in outputs)
                for rank, outputs in sorted(self._outputs.items())
            ).rstrip()
            return {RANK_MIME: payload, "text/plain": plain}
        return {
            RANK_MIME: payload,
            "text/html": render_html(execution, self._target_rank),
            "text/plain": render_plain(execution, self._target_rank),
        }

    def _send_display(self, message_type: str, data: Mapping[str, Any]) -> None:
        self._kernel.send_response(
            self._kernel.iopub_socket,
            message_type,
            {
                "data": dict(data),
                "metadata": {},
                "transient": {"display_id": self._display_id},
            },
        )
        self._last_sent = asyncio.get_running_loop().time()
        self._dirty = False


class _LiveRankUpdates:
    """Broadcast the latest transient rank snapshot to connected frontends."""

    def __init__(self, kernel: SPMDKernel) -> None:
        self._kernel = kernel
        self._subscribers: set[str] = set()
        self._latest: dict[str, Any] | None = None

    def handle_frontend(self, message_type: str, message: Mapping[str, Any]) -> bool:
        content = message.get("content", {})
        if not isinstance(content, Mapping):
            return False
        comm_id = content.get("comm_id")
        if not isinstance(comm_id, str) or not comm_id:
            return False

        if message_type == "comm_open":
            if content.get("target_name") != RANK_UPDATE_COMM_TARGET:
                return False
            self._subscribers.add(comm_id)
            self._send(
                comm_id,
                {
                    "method": "snapshots",
                    "snapshots": [self._latest] if self._latest is not None else [],
                },
            )
            return True

        if comm_id not in self._subscribers:
            return False
        if message_type == "comm_close":
            self._subscribers.discard(comm_id)
        elif message_type == "comm_msg":
            data = content.get("data", {})
            if isinstance(data, Mapping) and data.get("method") == "request_snapshots":
                self._send(
                    comm_id,
                    {
                        "method": "snapshots",
                        "snapshots": [self._latest] if self._latest is not None else [],
                    },
                )
        return True

    def publish(self, snapshot: dict[str, Any]) -> None:
        self._latest = snapshot
        for comm_id in tuple(self._subscribers):
            self._send(comm_id, {"method": "update", "snapshot": snapshot})

    def comm_info(self) -> dict[str, dict[str, str]]:
        return {comm_id: {"target_name": RANK_UPDATE_COMM_TARGET} for comm_id in self._subscribers}

    def reset(self) -> None:
        self._subscribers.clear()
        self._latest = None

    def _send(self, comm_id: str, data: dict[str, Any]) -> None:
        self._kernel.send_response(
            self._kernel.iopub_socket,
            "comm_msg",
            {"comm_id": comm_id, "data": data},
        )


class SPMDKernel(Kernel):
    """One notebook kernel whose executions fan out to persistent rank kernels."""

    implementation = "jupyter_distributed"
    implementation_version = "0.1.0"
    language = "python"
    language_version = "3"
    language_info: ClassVar[dict[str, Any]] = {
        "name": "python",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }
    banner = "Jupyter Distributed Python"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._debugger = DistributedDebugger(self)
        self._comms = DistributedCommRouter(self)
        self.live_updates = _LiveRankUpdates(self)
        self._process_registry = ChildProcessRegistry.from_environment()
        self.group = DistributedKernelGroup(
            int(os.environ.get("JUPYTER_DISTRIBUTED_WORLD_SIZE", "1")),
            kernel_name=os.environ.get("JUPYTER_DISTRIBUTED_BASE_KERNEL", "python3"),
            cwd=os.environ.get("JUPYTER_DISTRIBUTED_CWD") or None,
            on_debug_event=self._debugger.handle_event,
            on_comm_event=self._comms.handle_rank,
            on_rank_failure=self._comms.handle_rank_failure,
        )
        self._execution_loop: asyncio.AbstractEventLoop | None = None
        self.shell_handlers.update(
            {
                "comm_open": self.comm_open,
                "comm_msg": self.comm_msg,
                "comm_close": self.comm_close,
            }
        )

    def _handle_sigint(self, _signum: int, _frame: object) -> None:
        """Forward the ordinary Jupyter kernel interrupt to every child rank."""

        loop = self._execution_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(loop.create_task, self.group.interrupt())

    async def _ensure_started(self) -> None:
        if not self.group.ranks:
            await self.group.start()
            self._process_registry.update(self.group.ranks)

    async def _run_on_kernel_io_loop(self, callback: Callable[[], Awaitable[Any]]) -> Any:
        """Run rank-group work on the one event loop that owns its async state."""

        if _ON_KERNEL_IO_LOOP.get():
            return await callback()
        result: Future[Any] = Future()

        async def invoke() -> None:
            token = _ON_KERNEL_IO_LOOP.set(True)
            try:
                value = await callback()
            except BaseException as error:
                if not result.cancelled():
                    result.set_exception(error)
            else:
                if not result.cancelled():
                    result.set_result(value)
            finally:
                _ON_KERNEL_IO_LOOP.reset(token)

        self.io_loop.add_callback(invoke)
        return await asyncio.wrap_future(result)

    async def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Mapping[str, Any] | None = None,
        allow_stdin: bool = False,
        *,
        cell_meta: Mapping[str, Any] | None = None,
        cell_id: str | None = None,
    ) -> dict[str, Any]:
        if not _ON_KERNEL_IO_LOOP.get():
            return await self._run_on_kernel_io_loop(
                lambda: self.do_execute(
                    code,
                    silent,
                    store_history,
                    user_expressions,
                    allow_stdin,
                    cell_meta=cell_meta,
                    cell_id=cell_id,
                )
            )
        self._execution_loop = asyncio.get_running_loop()
        await self._ensure_started()
        try:
            rank_cell = parse_rank_cell(code)
        except RankMagicError as error:
            return self._error("RankMagicError", str(error), self.group.execution_count)
        target_rank = rank_cell.rank if rank_cell is not None else None
        if target_rank is not None and target_rank >= self.group.world_size:
            return self._error(
                "RankMagicError",
                f"rank must be between 0 and {self.group.world_size - 1}, got {target_rank}",
                self.group.execution_count,
            )
        executed_code = rank_cell.code if rank_cell is not None else code
        live_display = None
        if not silent:
            live_display = _LiveRankDisplay(
                self,
                self.group.execution_count + (1 if store_history else 0),
                target_rank,
            )
        try:
            execution = await self.group.execute(
                executed_code,
                silent=silent,
                store_history=store_history,
                user_expressions=user_expressions,
                on_output=live_display.update if live_display is not None else None,
                target_rank=target_rank,
            )
        except RuntimeError as error:
            return self._error(
                "DistributedKernelUnavailable",
                str(error),
                self.group.execution_count,
            )
        if live_display is not None:
            await live_display.finish(execution)
        if execution.status == "ok":
            reply_rank = target_rank if target_rank is not None else 0
            rank_reply = execution.ranks[reply_rank].reply
            payload = rank_reply.get("payload", [])
            return {
                "status": "ok",
                "execution_count": execution.execution_count,
                "payload": list(payload) if isinstance(payload, list) else [],
                "user_expressions": dict(rank_reply.get("user_expressions", {})),
            }
        failed = next(result for result in execution.ranks if result.status != "ok")
        return self._error(
            "DistributedExecutionError",
            f"cell failed on rank {failed.rank}",
            execution.execution_count,
        )

    async def do_complete(self, code: str, cursor_pos: int) -> Mapping[str, Any]:
        if not _ON_KERNEL_IO_LOOP.get():
            return await self._run_on_kernel_io_loop(lambda: self.do_complete(code, cursor_pos))
        await self._ensure_started()
        return await self.group.complete(code, cursor_pos)

    async def do_inspect(
        self,
        code: str,
        cursor_pos: int,
        detail_level: int = 0,
        omit_sections: tuple[str, ...] = (),
    ) -> Mapping[str, Any]:
        if not _ON_KERNEL_IO_LOOP.get():
            return await self._run_on_kernel_io_loop(
                lambda: self.do_inspect(code, cursor_pos, detail_level, omit_sections)
            )
        await self._ensure_started()
        return await self.group.inspect(code, cursor_pos, detail_level)

    async def do_is_complete(self, code: str) -> Mapping[str, Any]:
        if not _ON_KERNEL_IO_LOOP.get():
            return await self._run_on_kernel_io_loop(lambda: self.do_is_complete(code))
        await self._ensure_started()
        return await self.group.is_complete(code)

    async def kernel_info_request(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        """Report the selected base kernel's language and protocol metadata."""

        if not _ON_KERNEL_IO_LOOP.get():
            await self._run_on_kernel_io_loop(
                lambda: self.kernel_info_request(stream, ident, parent)
            )
            return
        if self.session is None:
            return
        await self._ensure_started()
        content = dict(await self.group.kernel_info())
        content.setdefault("status", "ok")
        content["implementation"] = self.implementation
        content["implementation_version"] = self.implementation_version
        supported_features = content.get("supported_features", ())
        debugger = bool(content.get("debugger", False)) or (
            isinstance(supported_features, (list, tuple)) and "debugger" in supported_features
        )
        self._debugger.set_available(debugger)
        content["debugger"] = debugger
        self.session.send(stream, "kernel_info_reply", content, parent, ident)

    async def do_debug_request(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        """Route one Jupyter Debug Adapter Protocol request across the ranks."""

        if not _ON_KERNEL_IO_LOOP.get():
            return await self._run_on_kernel_io_loop(lambda: self.do_debug_request(msg))
        await self._ensure_started()
        return await self._debugger.request(msg)

    async def comm_open(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        if not _ON_KERNEL_IO_LOOP.get():
            await self._run_on_kernel_io_loop(lambda: self.comm_open(stream, ident, parent))
            return
        await self._ensure_started()
        if self.live_updates.handle_frontend("comm_open", parent):
            return
        await self._comms.handle_frontend("comm_open", parent)

    async def comm_msg(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        if not _ON_KERNEL_IO_LOOP.get():
            await self._run_on_kernel_io_loop(lambda: self.comm_msg(stream, ident, parent))
            return
        await self._ensure_started()
        if self.live_updates.handle_frontend("comm_msg", parent):
            return
        await self._comms.handle_frontend("comm_msg", parent)

    async def comm_close(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        if not _ON_KERNEL_IO_LOOP.get():
            await self._run_on_kernel_io_loop(lambda: self.comm_close(stream, ident, parent))
            return
        await self._ensure_started()
        if self.live_updates.handle_frontend("comm_close", parent):
            return
        await self._comms.handle_frontend("comm_close", parent)

    async def comm_info_request(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        """Report comms across every rank so frontend state can reconnect."""

        if not _ON_KERNEL_IO_LOOP.get():
            await self._run_on_kernel_io_loop(lambda: self.comm_info_request(stream, ident, parent))
            return
        if self.session is None:
            return
        await self._ensure_started()
        target_name = parent.get("content", {}).get("target_name")
        comms = self._comms.comm_info(str(target_name) if target_name is not None else None)
        if target_name is None or target_name == RANK_UPDATE_COMM_TARGET:
            comms.update(self.live_updates.comm_info())
        content = {"status": "ok", "comms": comms}
        self.session.send(stream, "comm_info_reply", content, parent, ident)

    async def do_shutdown(self, restart: bool) -> dict[str, Any]:
        if not _ON_KERNEL_IO_LOOP.get():
            return await self._run_on_kernel_io_loop(lambda: self.do_shutdown(restart))
        ranks = self.group.ranks
        try:
            await self.group.shutdown(now=True)
            self._comms.reset()
            self.live_updates.reset()
        finally:
            health = await asyncio.gather(
                *(rank.is_alive() for rank in ranks), return_exceptions=True
            )
            if all(alive is False for alive in health):
                self._process_registry.remove()
        return {"status": "ok", "restart": restart}

    async def interrupt_group(self) -> None:
        if not _ON_KERNEL_IO_LOOP.get():
            await self._run_on_kernel_io_loop(self.interrupt_group)
            return
        await self.group.interrupt()

    async def interrupt_request(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        """Handle Jupyter's message-mode interrupt as a group operation."""

        if not _ON_KERNEL_IO_LOOP.get():
            await self._run_on_kernel_io_loop(lambda: self.interrupt_request(stream, ident, parent))
            return
        if self.session is None:
            return
        try:
            await self.group.interrupt()
            content: dict[str, Any] = {"status": "ok"}
        except Exception as error:
            content = {
                "status": "error",
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": traceback.format_exc().splitlines(),
            }
        self.session.send(stream, "interrupt_reply", content, parent, ident=ident)

    @staticmethod
    def _error(name: str, value: str, execution_count: int = 0) -> dict[str, Any]:
        return {
            "status": "error",
            "execution_count": execution_count,
            "ename": name,
            "evalue": value,
            "traceback": [],
        }


def main() -> None:
    app = IPKernelApp.instance(kernel_class=SPMDKernel)
    app.initialize()
    if app.kernel is not None and hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, app.kernel._handle_sigint)
    app.start()


if __name__ == "__main__":
    main()
