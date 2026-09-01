"""Server-owned lifecycle for distributed views of ordinary Jupyter kernels."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jupyter_core.paths import jupyter_runtime_dir

from .process_registry import OrphanProcessReaper


@dataclass(slots=True)
class DistributedKernelState:
    """Launch information retained while a logical kernel is proxy-managed."""

    kernel_name: str
    original_argv: tuple[str, ...]
    original_interrupt_mode: str
    original_env: dict[str, str] | None
    original_cwd: str | None
    world_size: int = 1
    proxied: bool = False


@dataclass(slots=True)
class PreparedKernelRestart:
    """A proxy launch configuration awaiting a client-managed restart."""

    token: str
    world_size: int


class KernelLaunchAdapter:
    """Isolate the Jupyter kernel-manager details needed to replace a launch."""

    def capture(self, kernel: Any) -> DistributedKernelState:
        launch_args = self._launch_args(kernel)
        launch_env = launch_args.get("env")
        return DistributedKernelState(
            kernel_name=str(kernel.kernel_name),
            original_argv=tuple(kernel.kernel_spec.argv),
            original_interrupt_mode=str(kernel.kernel_spec.interrupt_mode),
            original_env=dict(launch_env) if isinstance(launch_env, Mapping) else None,
            original_cwd=str(launch_args["cwd"]) if launch_args.get("cwd") else None,
        )

    def configure_proxy(
        self,
        kernel: Any,
        state: DistributedKernelState,
        world_size: int,
        registry_file: Path,
    ) -> None:
        kernel.kernel_spec.argv = [
            sys.executable,
            "-m",
            "jupyter_distributed.kernel",
            "-f",
            "{connection_file}",
        ]
        kernel.kernel_spec.interrupt_mode = "message"
        launch_args = self._launch_args(kernel)
        launch_args["env"] = {
            **os.environ,
            **(state.original_env or {}),
            "JUPYTER_DISTRIBUTED_BASE_KERNEL": state.kernel_name,
            "JUPYTER_DISTRIBUTED_WORLD_SIZE": str(world_size),
            "JUPYTER_DISTRIBUTED_REGISTRY_FILE": str(registry_file),
        }
        if state.original_cwd is not None:
            launch_args["env"]["JUPYTER_DISTRIBUTED_CWD"] = state.original_cwd

    def configure_direct(self, kernel: Any, state: DistributedKernelState) -> None:
        kernel.kernel_spec.argv = list(state.original_argv)
        kernel.kernel_spec.interrupt_mode = state.original_interrupt_mode
        launch_args = self._launch_args(kernel)
        if state.original_env is None:
            launch_args.pop("env", None)
        else:
            launch_args["env"] = dict(state.original_env)

    @staticmethod
    def _launch_args(kernel: Any) -> MutableMapping[str, Any]:
        launch_args = getattr(kernel, "_launch_args", None)
        if not isinstance(launch_args, MutableMapping):
            raise RuntimeError(
                "The configured Jupyter kernel manager does not expose mutable launch arguments"
            )
        return launch_args


class DistributedKernelCoordinator:
    """Run managed notebook kernels through one persistent proxy architecture.

    The Jupyter Server continues to expose the original kernel ID and kernel
    name. The process is restarted as an internal proxy, which launches one or
    more copies of the originally selected kernelspec.
    """

    def __init__(
        self,
        kernel_manager: Any,
        *,
        launch_adapter: KernelLaunchAdapter | None = None,
        registry_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.kernel_manager = kernel_manager
        self.launch_adapter = launch_adapter or KernelLaunchAdapter()
        self.reaper = OrphanProcessReaper(
            registry_dir or Path(jupyter_runtime_dir()) / "jupyter-distributed"
        )
        self._states: dict[str, DistributedKernelState] = {}
        self._prepared_restarts: dict[str, PreparedKernelRestart] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def describe(self, kernel_id: str) -> dict[str, Any]:
        kernel = self._kernel(kernel_id)
        state = self._states.get(kernel_id)
        return {
            "kernel_id": kernel_id,
            "kernel_name": state.kernel_name if state else kernel.kernel_name,
            "world_size": state.world_size if state else 1,
            "distributed": bool(state and state.world_size > 1),
            "proxied": bool(state and state.proxied),
        }

    async def set_world_size(self, kernel_id: str, world_size: int) -> dict[str, Any]:
        self._validate_world_size(world_size)

        lock = self._locks.setdefault(kernel_id, asyncio.Lock())
        async with lock:
            await self.reaper.reap()
            kernel = self._kernel(kernel_id)
            state = self._state(kernel_id, kernel)
            self._cancel_prepared_restart(kernel_id, kernel, state)
            if state.proxied and state.world_size == world_size:
                return self.describe(kernel_id)

            previous_world_size = state.world_size
            previously_proxied = state.proxied
            self.launch_adapter.configure_proxy(
                kernel, state, world_size, self.reaper.path_for(kernel_id)
            )

            try:
                await self.kernel_manager.restart_kernel(kernel_id)
            except BaseException:
                if previously_proxied:
                    self.launch_adapter.configure_proxy(
                        kernel,
                        state,
                        previous_world_size,
                        self.reaper.path_for(kernel_id),
                    )
                else:
                    self.launch_adapter.configure_direct(kernel, state)
                raise

            state.world_size = world_size
            state.proxied = True
            return self.describe(kernel_id)

    async def prepare_world_size(self, kernel_id: str, world_size: int) -> dict[str, Any]:
        """Configure the next launch for a restart performed by the client."""

        self._validate_world_size(world_size)
        lock = self._locks.setdefault(kernel_id, asyncio.Lock())
        async with lock:
            await self.reaper.reap()
            kernel = self._kernel(kernel_id)
            state = self._state(kernel_id, kernel)
            self._cancel_prepared_restart(kernel_id, kernel, state)
            if state.proxied and state.world_size == world_size:
                return {**self.describe(kernel_id), "restart_required": False}

            self.launch_adapter.configure_proxy(
                kernel, state, world_size, self.reaper.path_for(kernel_id)
            )
            prepared = PreparedKernelRestart(str(uuid.uuid4()), world_size)
            self._prepared_restarts[kernel_id] = prepared
            return {
                **self.describe(kernel_id),
                "world_size": world_size,
                "proxied": True,
                "restart_required": True,
                "restart_token": prepared.token,
            }

    async def finish_prepared_restart(
        self, kernel_id: str, token: str, *, commit: bool
    ) -> dict[str, Any]:
        """Commit or roll back a client-managed kernel restart."""

        lock = self._locks.setdefault(kernel_id, asyncio.Lock())
        async with lock:
            kernel = self._kernel(kernel_id)
            state = self._state(kernel_id, kernel)
            prepared = self._prepared_restarts.get(kernel_id)
            if prepared is None or prepared.token != token:
                raise ValueError("Unknown or expired kernel restart token")
            if commit:
                state.world_size = prepared.world_size
                state.proxied = True
                self._prepared_restarts.pop(kernel_id, None)
            else:
                self._cancel_prepared_restart(kernel_id, kernel, state)
            return self.describe(kernel_id)

    def forget(self, kernel_id: str) -> None:
        self._states.pop(kernel_id, None)
        self._prepared_restarts.pop(kernel_id, None)
        self._locks.pop(kernel_id, None)

    def _state(self, kernel_id: str, kernel: Any) -> DistributedKernelState:
        state = self._states.get(kernel_id)
        if state is None:
            state = self.launch_adapter.capture(kernel)
            self._states[kernel_id] = state
        return state

    def _cancel_prepared_restart(
        self, kernel_id: str, kernel: Any, state: DistributedKernelState
    ) -> None:
        if self._prepared_restarts.pop(kernel_id, None) is None:
            return
        if state.proxied:
            self.launch_adapter.configure_proxy(
                kernel, state, state.world_size, self.reaper.path_for(kernel_id)
            )
        else:
            self.launch_adapter.configure_direct(kernel, state)

    @staticmethod
    def _validate_world_size(world_size: int) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("world_size must be an integer")
        if world_size < 1:
            raise ValueError("world_size must be greater than zero")

    def _kernel(self, kernel_id: str) -> Any:
        return self.kernel_manager.get_kernel(kernel_id)


__all__ = ["DistributedKernelCoordinator", "DistributedKernelState", "KernelLaunchAdapter"]
