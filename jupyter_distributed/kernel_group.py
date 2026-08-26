"""Logical distributed kernel composed of persistent Jupyter kernels."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .protocol import GroupExecution, GroupStatus
from .rank_kernel import CommEventCallback, DebugEventCallback, OutputCallback, RankKernel


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class DistributedKernelGroup:
    """Manage N child kernels as one persistent SPMD execution context."""

    def __init__(
        self,
        world_size: int = 1,
        *,
        kernel_name: str = "python3",
        master_addr: str = "127.0.0.1",
        master_port: int | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        on_debug_event: DebugEventCallback | None = None,
        on_comm_event: CommEventCallback | None = None,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        self.world_size = world_size
        self.kernel_name = kernel_name
        self.master_addr = master_addr
        self.master_port = master_port
        self.base_env = {**os.environ, **(env or {})}
        self.cwd = cwd
        self.on_debug_event = on_debug_event
        self.on_comm_event = on_comm_event
        self._ranks: list[RankKernel] = []
        self._state: Literal["stopped", "starting", "idle", "busy", "restarting"] = "stopped"
        self._execution_count = 0
        self._execution_lock = asyncio.Lock()

    @property
    def ranks(self) -> tuple[RankKernel, ...]:
        return tuple(self._ranks)

    @property
    def execution_count(self) -> int:
        return self._execution_count

    async def start(self) -> None:
        if self._ranks:
            raise RuntimeError("kernel group is already started")
        self._state = "starting"
        port = self.master_port or _free_port(self.master_addr)
        self.master_port = port
        ranks = [
            RankKernel(
                rank,
                self._rank_env(rank, port),
                kernel_name=self.kernel_name,
                cwd=self.cwd,
                on_debug_event=self.on_debug_event,
                on_comm_event=self.on_comm_event,
            )
            for rank in range(self.world_size)
        ]
        self._ranks = ranks
        results = await asyncio.gather(*(rank.start() for rank in ranks), return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            await asyncio.gather(
                *(rank.shutdown(now=True) for rank in ranks), return_exceptions=True
            )
            self._ranks = []
            self._state = "stopped"
            raise RuntimeError("failed to start distributed kernel group") from errors[0]
        self._state = "idle"

    async def execute(
        self,
        code: str,
        *,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Mapping[str, Any] | None = None,
        on_output: OutputCallback | None = None,
        target_rank: int | None = None,
    ) -> GroupExecution:
        async with self._execution_lock:
            self._require_started()
            if target_rank is not None and (
                isinstance(target_rank, bool) or not 0 <= target_rank < self.world_size
            ):
                raise ValueError(
                    f"rank must be between 0 and {self.world_size - 1}, got {target_rank}"
                )
            self._state = "busy"
            if store_history:
                self._execution_count += 1
            try:
                results = await asyncio.gather(
                    *(
                        rank.execute(
                            code if target_rank is None or rank.rank == target_rank else "pass",
                            silent=silent,
                            store_history=store_history,
                            user_expressions=(
                                user_expressions
                                if target_rank is None or rank.rank == target_rank
                                else None
                            ),
                            on_output=(
                                on_output
                                if target_rank is None or rank.rank == target_rank
                                else None
                            ),
                        )
                        for rank in self._ranks
                    )
                )
                return GroupExecution(execution_count=self._execution_count, ranks=tuple(results))
            finally:
                self._state = "idle" if self._ranks else "stopped"

    async def complete(self, code: str, cursor_pos: int) -> Mapping[str, Any]:
        self._require_started()
        return await self._ranks[0].request("complete", code, cursor_pos)

    async def inspect(self, code: str, cursor_pos: int, detail_level: int = 0) -> Mapping[str, Any]:
        self._require_started()
        return await self._ranks[0].request("inspect", code, cursor_pos, detail_level)

    async def kernel_info(self) -> Mapping[str, Any]:
        self._require_started()
        return await self._ranks[0].request("kernel_info")

    async def is_complete(self, code: str) -> Mapping[str, Any]:
        self._require_started()
        return await self._ranks[0].request("is_complete", code)

    async def debug(
        self, requests: Mapping[int, Mapping[str, Any]]
    ) -> dict[int, Mapping[str, Any]]:
        """Send rank-specific debugger requests concurrently."""

        self._require_started()
        ranks = {rank.rank: rank for rank in self._ranks}
        requested = [(rank, content) for rank, content in requests.items() if rank in ranks]
        replies = await asyncio.gather(
            *(ranks[rank].debug_request(content) for rank, content in requested)
        )
        return {rank: reply for (rank, _content), reply in zip(requested, replies, strict=True)}

    async def send_comm(
        self,
        ranks: Sequence[int],
        message_type: str,
        content: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        buffers: list[Any] | None = None,
        parent: Mapping[str, Any] | None = None,
    ) -> None:
        """Send one opaque comm message to selected child kernels."""

        self._require_started()
        rank_kernels = {rank.rank: rank for rank in self._ranks}
        for rank in ranks:
            child = rank_kernels.get(rank)
            if child is not None:
                child.send_comm(
                    message_type,
                    content,
                    metadata=metadata,
                    buffers=buffers,
                    parent=parent,
                )

    async def interrupt(self) -> None:
        await asyncio.gather(*(rank.interrupt() for rank in self._ranks), return_exceptions=True)

    async def restart(self, world_size: int | None = None) -> None:
        new_world_size = self.world_size if world_size is None else world_size
        if new_world_size < 1:
            raise ValueError("world_size must be at least 1")
        self._state = "restarting"
        await self.shutdown(now=True)
        self.world_size = new_world_size
        self.master_port = None
        self._execution_count = 0
        await self.start()

    async def shutdown(self, *, now: bool = False) -> None:
        ranks, self._ranks = self._ranks, []
        await asyncio.gather(*(rank.shutdown(now=now) for rank in ranks), return_exceptions=True)
        self._state = "stopped"

    async def status(self) -> GroupStatus:
        alive = await asyncio.gather(*(rank.is_alive() for rank in self._ranks))
        return GroupStatus(
            state=self._state,
            world_size=self.world_size,
            alive_ranks=tuple(rank.rank for rank, is_alive in zip(self._ranks, alive) if is_alive),
        )

    def _rank_env(self, rank: int, port: int) -> dict[str, str]:
        coordinator_host = (
            f"[{self.master_addr}]"
            if ":" in self.master_addr and not self.master_addr.startswith("[")
            else self.master_addr
        )
        env = {
            **self.base_env,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_WORLD_SIZE": str(self.world_size),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(port),
            "JAX_COORDINATOR_ADDRESS": f"{coordinator_host}:{port}",
            "JAX_PROCESS_ID": str(rank),
            "JAX_NUM_PROCESSES": str(self.world_size),
        }
        if self.world_size > 1:
            env["PYTHONBREAKPOINT"] = "jupyter_distributed.breakpoint.distributed_breakpoint"
        return env

    def _require_started(self) -> None:
        if not self._ranks:
            raise RuntimeError("kernel group is not started")

    async def __aenter__(self) -> DistributedKernelGroup:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.shutdown(now=True)
