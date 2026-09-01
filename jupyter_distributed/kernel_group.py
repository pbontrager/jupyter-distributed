"""Logical distributed kernel composed of persistent Jupyter kernels."""

from __future__ import annotations

import asyncio
import inspect
import os
import socket
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .protocol import GroupExecution, GroupStatus, RankExecution, RankOutput, RankOutputPatch
from .rank_kernel import (
    CommEventCallback,
    DebugEventCallback,
    FailureCallback,
    OutputCallback,
    RankKernel,
    RankKernelFailure,
)


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
        on_rank_failure: FailureCallback | None = None,
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
        self.on_rank_failure = on_rank_failure
        self._ranks: list[RankKernel] = []
        self._state: Literal[
            "stopped", "starting", "idle", "busy", "restarting", "restart_required"
        ] = "stopped"
        self._failure: str | None = None
        self._execution_count = 0
        self._execution_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()

    @property
    def ranks(self) -> tuple[RankKernel, ...]:
        return tuple(self._ranks)

    @property
    def execution_count(self) -> int:
        return self._execution_count

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._state in {"idle", "busy"} and self._ranks:
                return
            if self._state == "restart_required":
                raise RuntimeError("kernel group must be restarted after a rank failure")
            await self._start_unlocked()

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
            self._require_usable()
            if target_rank is not None and (
                isinstance(target_rank, bool) or not 0 <= target_rank < self.world_size
            ):
                raise ValueError(
                    f"rank must be between 0 and {self.world_size - 1}, got {target_rank}"
                )
            self._state = "busy"
            if store_history:
                self._execution_count += 1
            latest_outputs: dict[int, tuple[RankOutput, ...]] = {
                rank.rank: () for rank in self._ranks
            }

            async def capture_output(
                rank: int,
                outputs: tuple[RankOutput, ...],
                patches: tuple[RankOutputPatch, ...],
            ) -> None:
                latest_outputs[rank] = outputs
                if on_output is not None:
                    notified = on_output(rank, outputs, patches)
                    if inspect.isawaitable(notified):
                        await notified

            tasks = {
                rank.rank: asyncio.create_task(
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
                            capture_output
                            if target_rank is None or rank.rank == target_rank
                            else None
                        ),
                    )
                )
                for rank in self._ranks
            }
            try:
                done, pending = await asyncio.wait(
                    tasks.values(), return_when=asyncio.FIRST_EXCEPTION
                )
                failures = [
                    task.exception()
                    for task in done
                    if not task.cancelled() and task.exception() is not None
                ]
                if not failures:
                    results = await asyncio.gather(*tasks.values())
                    return GroupExecution(
                        execution_count=self._execution_count, ranks=tuple(results)
                    )

                first_failure = failures[0]
                assert first_failure is not None
                failed_rank = (
                    first_failure.rank if isinstance(first_failure, RankKernelFailure) else -1
                )
                self._mark_restart_required(failed_rank, first_failure)
                await self.interrupt(exclude={failed_rank})
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                results: list[RankExecution] = []
                for rank in self._ranks:
                    task = tasks[rank.rank]
                    if task.done() and not task.cancelled() and task.exception() is None:
                        results.append(task.result())
                        continue
                    error = task.exception() if task.done() and not task.cancelled() else None
                    outputs = latest_outputs[rank.rank]
                    if isinstance(error, RankKernelFailure):
                        outputs = error.outputs
                    if error is not None:
                        outputs = (
                            *outputs,
                            RankOutput(
                                rank.rank,
                                "error",
                                {
                                    "ename": "RankProcessError",
                                    "evalue": str(error),
                                    "traceback": [],
                                },
                            ),
                        )
                    results.append(
                        RankExecution(
                            rank.rank,
                            "error" if error is not None else "aborted",
                            outputs,
                        )
                    )
                return GroupExecution(execution_count=self._execution_count, ranks=tuple(results))
            finally:
                if self._state == "busy":
                    self._state = "idle" if self._ranks else "stopped"

    async def complete(self, code: str, cursor_pos: int) -> Mapping[str, Any]:
        self._require_usable()
        return await self._ranks[0].request("complete", code, cursor_pos)

    async def inspect(self, code: str, cursor_pos: int, detail_level: int = 0) -> Mapping[str, Any]:
        self._require_usable()
        return await self._ranks[0].request("inspect", code, cursor_pos, detail_level)

    async def kernel_info(self) -> Mapping[str, Any]:
        self._require_usable()
        return await self._ranks[0].request("kernel_info")

    async def is_complete(self, code: str) -> Mapping[str, Any]:
        self._require_usable()
        return await self._ranks[0].request("is_complete", code)

    async def debug(
        self, requests: Mapping[int, Mapping[str, Any]]
    ) -> dict[int, Mapping[str, Any]]:
        """Send rank-specific debugger requests concurrently."""

        self._require_usable()
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

        self._require_usable()
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

    async def interrupt(self, *, exclude: set[int] | None = None) -> None:
        excluded = exclude or set()
        await asyncio.gather(
            *(rank.interrupt() for rank in self._ranks if rank.rank not in excluded),
            return_exceptions=True,
        )

    async def restart(self, world_size: int | None = None) -> None:
        new_world_size = self.world_size if world_size is None else world_size
        if new_world_size < 1:
            raise ValueError("world_size must be at least 1")
        await self.interrupt()
        async with self._execution_lock, self._lifecycle_lock:
            self._state = "restarting"
            await self._shutdown_unlocked(now=True)
            self.world_size = new_world_size
            self.master_port = None
            self._execution_count = 0
            await self._start_unlocked()

    async def shutdown(self, *, now: bool = False) -> None:
        await self.interrupt()
        async with self._execution_lock, self._lifecycle_lock:
            await self._shutdown_unlocked(now=now)

    async def status(self) -> GroupStatus:
        alive = await asyncio.gather(
            *(rank.is_alive() for rank in self._ranks), return_exceptions=True
        )
        return GroupStatus(
            state=self._state,
            world_size=self.world_size,
            alive_ranks=tuple(
                rank.rank
                for rank, is_alive in zip(self._ranks, alive, strict=True)
                if is_alive is True
            ),
            failure=self._failure,
        )

    async def _start_unlocked(self) -> None:
        if self._ranks:
            raise RuntimeError("kernel group is already started")
        self._state = "starting"
        self._failure = None
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
                on_failure=self._rank_failed,
            )
            for rank in range(self.world_size)
        ]
        results = await asyncio.gather(*(rank.start() for rank in ranks), return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        health = await asyncio.gather(*(rank.is_alive() for rank in ranks), return_exceptions=True)
        errors.extend(
            failure
            for rank in ranks
            if (failure := getattr(rank, "failure_reason", None)) is not None
        )
        errors.extend(
            RuntimeError(f"rank {rank.rank} exited during startup")
            for rank, alive in zip(ranks, health, strict=True)
            if alive is not True
        )
        if errors:
            await asyncio.gather(
                *(rank.shutdown(now=True) for rank in ranks), return_exceptions=True
            )
            self._state = "stopped"
            raise RuntimeError("failed to start distributed kernel group") from errors[0]
        self._ranks = ranks
        self._state = "idle"

    async def _shutdown_unlocked(self, *, now: bool) -> None:
        ranks, self._ranks = self._ranks, []
        await asyncio.gather(*(rank.shutdown(now=now) for rank in ranks), return_exceptions=True)
        self._failure = None
        self._state = "stopped"

    def _rank_failed(self, rank: int, error: BaseException) -> None:
        if self._state in {"stopped", "restarting"}:
            return
        self._mark_restart_required(rank, error)
        asyncio.create_task(self.interrupt(exclude={rank}))

    def _mark_restart_required(self, rank: int, error: BaseException) -> None:
        if self._state != "restart_required":
            label = f"rank {rank}" if rank >= 0 else "rank execution"
            self._failure = f"{label} failed: {error}"
            self._state = "restart_required"
            if self.on_rank_failure is not None:
                notified = self.on_rank_failure(rank, error)
                if inspect.isawaitable(notified):
                    asyncio.create_task(notified)

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

    def _require_usable(self) -> None:
        if self._state == "restart_required":
            raise RuntimeError(self._failure or "kernel group must be restarted")
        if not self._ranks:
            raise RuntimeError("kernel group is not started")

    async def __aenter__(self) -> DistributedKernelGroup:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.shutdown(now=True)
