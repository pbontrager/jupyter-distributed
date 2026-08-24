"""Jupyter kernel facade for a :class:`DistributedKernelGroup`."""

from __future__ import annotations

import asyncio
import html
import os
import signal
import traceback
from collections.abc import Mapping
from typing import Any, ClassVar

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel

from .kernel_group import DistributedKernelGroup
from .protocol import GroupExecution

RANK_MIME = "application/vnd.jupyter-distributed.rank+json"


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
        self.group = DistributedKernelGroup(
            int(os.environ.get("JUPYTER_DISTRIBUTED_WORLD_SIZE", "1")),
            kernel_name=os.environ.get("JUPYTER_DISTRIBUTED_BASE_KERNEL", "python3"),
            cwd=os.environ.get("JUPYTER_DISTRIBUTED_CWD") or None,
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
        execution = await self.group.execute(
            code,
            silent=silent,
            store_history=store_history,
            user_expressions=user_expressions,
        )
        if not silent:
            data = {
                RANK_MIME: execution.as_dict(),
                "text/html": render_html(execution),
                "text/plain": render_plain(execution),
            }
            self.send_response(
                self.iopub_socket,
                "display_data",
                {"data": data, "metadata": {}},
            )
        if execution.status == "ok":
            return {
                "status": "ok",
                "execution_count": execution.execution_count,
                "payload": [],
                "user_expressions": dict(
                    execution.ranks[0].reply.get("user_expressions", {})
                ),
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

    async def kernel_info_request(
        self, stream: Any, ident: Any, parent: Mapping[str, Any]
    ) -> None:
        """Report the selected base kernel's language and protocol metadata."""

        if self.session is None:
            return
        await self._ensure_started()
        content = dict(await self.group.kernel_info())
        content.setdefault("status", "ok")
        content["debugger"] = False
        self.session.send(stream, "kernel_info_reply", content, parent, ident)

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
