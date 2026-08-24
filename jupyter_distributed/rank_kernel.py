"""Control-plane wrapper around one ordinary Jupyter kernel process."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from jupyter_client import AsyncKernelManager
from jupyter_client.asynchronous.client import AsyncKernelClient

from .protocol import RankExecution, RankOutput

_OUTPUT_TYPES = {"stream", "display_data", "execute_result", "error"}


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
    ) -> None:
        self.rank = rank
        self.env = dict(env)
        self.cwd = cwd
        self.ready_timeout = ready_timeout
        self.manager = AsyncKernelManager(kernel_name=kernel_name)
        self.client: AsyncKernelClient | None = None

    async def start(self) -> None:
        await self.manager.start_kernel(env=self.env, cwd=self.cwd)
        client = self.manager.client()
        self.client = client
        client.start_channels()
        try:
            await client.wait_for_ready(timeout=self.ready_timeout)
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
        reply_task = asyncio.create_task(self._shell_reply(message_id))
        outputs: list[RankOutput] = []
        try:
            while True:
                message = await client.get_iopub_msg(timeout=None)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                message_type = message.get("msg_type")
                if message_type in _OUTPUT_TYPES:
                    outputs.append(
                        RankOutput(
                            rank=self.rank,
                            kind=message_type,
                            content=message.get("content", {}),
                        )
                    )
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
        status = reply.get("content", {}).get("status", "error")
        if status not in {"ok", "error", "aborted"}:
            status = "error"
        return RankExecution(
            rank=self.rank,
            status=status,
            outputs=tuple(outputs),
            reply=reply.get("content", {}),
        )

    async def request(
        self, message_type: str, *args: Any, **kwargs: Any
    ) -> Mapping[str, Any]:
        """Send a simple shell request and return its matching content."""

        client = self._client()
        sender = getattr(client, message_type)
        message_id = sender(*args, **kwargs)
        reply = await self._shell_reply(message_id)
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
