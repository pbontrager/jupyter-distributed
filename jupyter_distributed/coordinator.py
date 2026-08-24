"""Server-owned lifecycle for distributed views of ordinary Jupyter kernels."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DistributedKernelState:
    """Launch information retained while a logical kernel is distributed."""

    kernel_name: str
    original_argv: tuple[str, ...]
    original_interrupt_mode: str
    original_env: dict[str, str] | None
    original_cwd: str | None
    world_size: int = 1


class DistributedKernelCoordinator:
    """Switch managed kernels between direct and distributed execution.

    The Jupyter Server continues to expose the original kernel ID and kernel
    name. For ``world_size > 1`` its process is restarted as an internal proxy,
    which launches copies of the originally selected kernelspec.
    """

    def __init__(self, kernel_manager: Any) -> None:
        self.kernel_manager = kernel_manager
        self._states: dict[str, DistributedKernelState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def describe(self, kernel_id: str) -> dict[str, Any]:
        kernel = self._kernel(kernel_id)
        state = self._states.get(kernel_id)
        return {
            "kernel_id": kernel_id,
            "kernel_name": state.kernel_name if state else kernel.kernel_name,
            "world_size": state.world_size if state else 1,
            "distributed": bool(state and state.world_size > 1),
        }

    async def set_world_size(self, kernel_id: str, world_size: int) -> dict[str, Any]:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("world_size must be an integer")
        if world_size < 1:
            raise ValueError("world_size must be greater than zero")

        lock = self._locks.setdefault(kernel_id, asyncio.Lock())
        async with lock:
            kernel = self._kernel(kernel_id)
            state = self._states.get(kernel_id)
            if state is None:
                state = self._capture_state(kernel)
                self._states[kernel_id] = state
            if state.world_size == world_size:
                return self.describe(kernel_id)

            if world_size == 1:
                self._configure_direct(kernel, state)
            else:
                self._configure_proxy(kernel, state, world_size)

            try:
                await self.kernel_manager.restart_kernel(kernel_id)
            except BaseException:
                if state.world_size == 1:
                    self._configure_direct(kernel, state)
                else:
                    self._configure_proxy(kernel, state, state.world_size)
                raise

            state.world_size = world_size
            return self.describe(kernel_id)

    def forget(self, kernel_id: str) -> None:
        self._states.pop(kernel_id, None)
        self._locks.pop(kernel_id, None)

    def _kernel(self, kernel_id: str) -> Any:
        return self.kernel_manager.get_kernel(kernel_id)

    @staticmethod
    def _capture_state(kernel: Any) -> DistributedKernelState:
        launch_args = getattr(kernel, "_launch_args", {})
        launch_env = launch_args.get("env") if isinstance(launch_args, Mapping) else None
        return DistributedKernelState(
            kernel_name=str(kernel.kernel_name),
            original_argv=tuple(kernel.kernel_spec.argv),
            original_interrupt_mode=str(kernel.kernel_spec.interrupt_mode),
            original_env=dict(launch_env) if isinstance(launch_env, Mapping) else None,
            original_cwd=(
                str(launch_args["cwd"])
                if isinstance(launch_args, Mapping) and launch_args.get("cwd")
                else None
            ),
        )

    @staticmethod
    def _configure_proxy(
        kernel: Any, state: DistributedKernelState, world_size: int
    ) -> None:
        kernel.kernel_spec.argv = [
            sys.executable,
            "-m",
            "jupyter_distributed.kernel",
            "-f",
            "{connection_file}",
        ]
        kernel.kernel_spec.interrupt_mode = "message"
        launch_args = kernel._launch_args
        launch_args["env"] = {
            **os.environ,
            **(state.original_env or {}),
            "JUPYTER_DISTRIBUTED_BASE_KERNEL": state.kernel_name,
            "JUPYTER_DISTRIBUTED_WORLD_SIZE": str(world_size),
        }
        if state.original_cwd is not None:
            launch_args["env"]["JUPYTER_DISTRIBUTED_CWD"] = state.original_cwd

    @staticmethod
    def _configure_direct(kernel: Any, state: DistributedKernelState) -> None:
        kernel.kernel_spec.argv = list(state.original_argv)
        kernel.kernel_spec.interrupt_mode = state.original_interrupt_mode
        launch_args = kernel._launch_args
        if state.original_env is None:
            launch_args.pop("env", None)
        else:
            launch_args["env"] = dict(state.original_env)


__all__ = ["DistributedKernelCoordinator", "DistributedKernelState"]
