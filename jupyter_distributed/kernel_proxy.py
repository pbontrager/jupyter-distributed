"""Jupyter kernel facade for a :class:`DistributedKernelGroup`."""

from __future__ import annotations

import asyncio
import html
import os
import signal
import traceback
from collections.abc import Mapping
from typing import Any, ClassVar
from uuid import uuid4

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel

from .debugger import DistributedDebugger
from .kernel_group import DistributedKernelGroup
from .protocol import GroupExecution, RankOutput

RANK_MIME = "application/vnd.jupyter-distributed.rank+json"
_STREAM_UPDATE_INTERVAL = 0.05


def render_plain(execution: GroupExecution) -> str:
    sections: list[str] = []
    for result in execution.ranks:
        body = "".join(output.plain_text() for output in result.outputs)
        sections.append(f"[Rank {result.rank} — {result.status}]\n{body}".rstrip())
    return "\n\n".join(sections)


def render_html(execution: GroupExecution) -> str:
    sections: list[str] = ['<div class="jupyter-distributed-rank-output">']
    for result in execution.ranks:
        opened = " open" if result.rank == 0 else ""
        body = html.escape("".join(output.plain_text() for output in result.outputs))
        sections.append(
            f'<details data-rank="{result.rank}" data-status="{result.status}"{opened}>'
            f"<summary>Rank {result.rank} — {result.status}</summary>"
            f"<pre>{body}</pre></details>"
        )
    sections.append("</div>")
    return "".join(sections)


class _LiveRankDisplay:
    """Coalesce rank output events into one updating notebook display."""

    def __init__(self, kernel: SPMDKernel, execution_count: int) -> None:
        self._kernel = kernel
        self._execution_count = execution_count
        self._execution_id = uuid4().hex
        self._display_id = f"jupyter-distributed-{self._execution_id}"
        self._outputs: dict[int, tuple[RankOutput, ...]] = {
            rank: () for rank in range(kernel.group.world_size)
        }
        self._flush_task: asyncio.Task[None] | None = None
        self._last_sent = 0.0
        self._dirty = False

    def start(self) -> None:
        self._send("display_data", self._live_payload())

    def update(self, rank: int, outputs: tuple[RankOutput, ...]) -> None:
        self._outputs[rank] = outputs
        self._dirty = True
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
        payload = execution.as_dict()
        payload["execution_id"] = self._execution_id
        self._send("update_display_data", payload, execution=execution)

    async def _flush_after(self, delay: float) -> None:
        if delay:
            await asyncio.sleep(delay)
        self._flush_task = None
        if self._dirty:
            self._send("update_display_data", self._live_payload())

    def _live_payload(self) -> dict[str, Any]:
        return {
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

    def _send(
        self,
        message_type: str,
        payload: dict[str, Any],
        *,
        execution: GroupExecution | None = None,
    ) -> None:
        if execution is None:
            plain = "\n\n".join(
                f"[Rank {rank} — running]\n" + "".join(output.plain_text() for output in outputs)
                for rank, outputs in sorted(self._outputs.items())
            ).rstrip()
            data = {RANK_MIME: payload, "text/plain": plain}
        else:
            data = {
                RANK_MIME: payload,
                "text/html": render_html(execution),
                "text/plain": render_plain(execution),
            }
        self._kernel.send_response(
            self._kernel.iopub_socket,
            message_type,
            {
                "data": data,
                "metadata": {},
                "transient": {"display_id": self._display_id},
            },
        )
        self._last_sent = asyncio.get_running_loop().time()
        self._dirty = False


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
        self.group = DistributedKernelGroup(
            int(os.environ.get("JUPYTER_DISTRIBUTED_WORLD_SIZE", "1")),
            kernel_name=os.environ.get("JUPYTER_DISTRIBUTED_BASE_KERNEL", "python3"),
            cwd=os.environ.get("JUPYTER_DISTRIBUTED_CWD") or None,
            on_debug_event=self._debugger.handle_event,
        )
        self._execution_loop: asyncio.AbstractEventLoop | None = None

    def _handle_sigint(self, _signum: int, _frame: object) -> None:
        """Forward the ordinary Jupyter kernel interrupt to every child rank."""

        loop = self._execution_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(loop.create_task, self.group.interrupt())

    async def _ensure_started(self) -> None:
        if not self.group.ranks:
            await self.group.start()

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
        self._execution_loop = asyncio.get_running_loop()
        await self._ensure_started()
        live_display = None
        if not silent:
            live_display = _LiveRankDisplay(
                self,
                self.group.execution_count + (1 if store_history else 0),
            )
            live_display.start()
        execution = await self.group.execute(
            code,
            silent=silent,
            store_history=store_history,
            user_expressions=user_expressions,
            on_output=live_display.update if live_display is not None else None,
        )
        if live_display is not None:
            await live_display.finish(execution)
        if execution.status == "ok":
            return {
                "status": "ok",
                "execution_count": execution.execution_count,
                "payload": [],
                "user_expressions": dict(execution.ranks[0].reply.get("user_expressions", {})),
            }
        failed = next(result for result in execution.ranks if result.status != "ok")
        return self._error(
            "DistributedExecutionError",
            f"cell failed on rank {failed.rank}",
            execution.execution_count,
        )

    async def do_complete(self, code: str, cursor_pos: int) -> Mapping[str, Any]:
        await self._ensure_started()
        return await self.group.complete(code, cursor_pos)

    async def do_inspect(
        self,
        code: str,
        cursor_pos: int,
        detail_level: int = 0,
        omit_sections: tuple[str, ...] = (),
    ) -> Mapping[str, Any]:
        await self._ensure_started()
        return await self.group.inspect(code, cursor_pos, detail_level)

    async def do_is_complete(self, code: str) -> Mapping[str, Any]:
        await self._ensure_started()
        return await self.group.is_complete(code)

    async def kernel_info_request(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        """Report the selected base kernel's language and protocol metadata."""

        if self.session is None:
            return
        await self._ensure_started()
        content = dict(await self.group.kernel_info())
        content.setdefault("status", "ok")
        supported_features = content.get("supported_features", ())
        debugger = bool(content.get("debugger", False)) or (
            isinstance(supported_features, (list, tuple)) and "debugger" in supported_features
        )
        self._debugger.set_available(debugger)
        content["debugger"] = debugger
        self.session.send(stream, "kernel_info_reply", content, parent, ident)

    async def do_debug_request(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        """Route one Jupyter Debug Adapter Protocol request across the ranks."""

        await self._ensure_started()
        return await self._debugger.request(msg)

    async def do_shutdown(self, restart: bool) -> dict[str, Any]:
        await self.group.shutdown(now=True)
        return {"status": "ok", "restart": restart}

    async def interrupt_group(self) -> None:
        await self.group.interrupt()

    async def interrupt_request(self, stream: Any, ident: Any, parent: Mapping[str, Any]) -> None:
        """Handle Jupyter's message-mode interrupt as a group operation."""

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
