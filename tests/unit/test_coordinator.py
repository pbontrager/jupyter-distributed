from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from jupyter_distributed.coordinator import DistributedKernelCoordinator, KernelLaunchAdapter


@dataclass
class FakeSpec:
    argv: list[str] = field(
        default_factory=lambda: ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    )
    interrupt_mode: str = "signal"


@dataclass
class FakeSession:
    session: str = "manager-session"

    def clone(self) -> FakeSession:
        return FakeSession(self.session)


@dataclass
class FakeKernel:
    kernel_name: str = "custom-python"
    kernel_spec: FakeSpec = field(default_factory=FakeSpec)
    _launch_args: dict[str, Any] = field(default_factory=lambda: {"cwd": "/notebooks"})
    ready_checks: int = 0
    session: FakeSession = field(default_factory=FakeSession)

    def client(self, **kwargs: Any) -> FakeKernelClient:
        assert kwargs["session"] is not None
        return FakeKernelClient(self)


class FakeKernelClient:
    def __init__(self, kernel: FakeKernel) -> None:
        self.kernel = kernel

    def start_channels(self) -> None:
        pass

    def kernel_info(self) -> str:
        return "ready-request"

    async def get_shell_msg(self, timeout: float) -> dict[str, Any]:
        assert timeout == 1
        self.kernel.ready_checks += 1
        return {
            "msg_type": "kernel_info_reply",
            "parent_header": {"msg_id": "ready-request"},
        }

    def stop_channels(self) -> None:
        pass


class FakeKernelManager:
    def __init__(self) -> None:
        self.kernel = FakeKernel()
        self.restarts = 0

    def get_kernel(self, kernel_id: str) -> FakeKernel:
        if kernel_id != "kernel-id":
            raise KeyError(kernel_id)
        return self.kernel

    async def restart_kernel(self, kernel_id: str) -> None:
        assert kernel_id == "kernel-id"
        self.restarts += 1


@pytest.mark.asyncio
async def test_keeps_every_process_count_on_the_internal_proxy(tmp_path: Any) -> None:
    manager = FakeKernelManager()
    coordinator = DistributedKernelCoordinator(manager, registry_dir=tmp_path)

    single = await coordinator.set_world_size("kernel-id", 1)

    assert single["distributed"] is False
    assert single["proxied"] is True
    assert manager.kernel.kernel_spec.argv[1:3] == ["-m", "jupyter_distributed.kernel"]
    assert manager.kernel.kernel_spec.interrupt_mode == "message"
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_WORLD_SIZE"] == "1"

    distributed = await coordinator.set_world_size("kernel-id", 4)

    assert distributed["kernel_name"] == "custom-python"
    assert distributed["world_size"] == 4
    assert distributed["proxied"] is True
    assert manager.kernel.kernel_name == "custom-python"
    assert manager.kernel.kernel_spec.argv[1:3] == ["-m", "jupyter_distributed.kernel"]
    assert manager.kernel.kernel_spec.interrupt_mode == "message"
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_BASE_KERNEL"] == (
        "custom-python"
    )
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_WORLD_SIZE"] == "4"
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_CWD"] == "/notebooks"
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_REGISTRY_FILE"] == str(
        tmp_path / "kernel-id.json"
    )
    assert manager.kernel.ready_checks == 2

    single_again = await coordinator.set_world_size("kernel-id", 1)

    assert single_again["distributed"] is False
    assert single_again["proxied"] is True
    assert manager.kernel.kernel_spec.argv[1:3] == ["-m", "jupyter_distributed.kernel"]
    assert manager.kernel.kernel_spec.interrupt_mode == "message"
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_WORLD_SIZE"] == "1"
    assert manager.restarts == 3
    assert manager.kernel.ready_checks == 3


@pytest.mark.asyncio
async def test_rejects_invalid_world_sizes_without_restart() -> None:
    manager = FakeKernelManager()
    coordinator = DistributedKernelCoordinator(manager)

    for value in (True, 0, -1, 1.5):
        with pytest.raises((TypeError, ValueError)):
            await coordinator.set_world_size("kernel-id", value)  # type: ignore[arg-type]

    assert manager.restarts == 0


@pytest.mark.asyncio
async def test_accepts_arbitrary_positive_process_count() -> None:
    manager = FakeKernelManager()
    coordinator = DistributedKernelCoordinator(manager)

    model = await coordinator.set_world_size("kernel-id", 257)

    assert model["world_size"] == 257
    assert model["proxied"] is True
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_WORLD_SIZE"] == "257"


@pytest.mark.asyncio
async def test_client_managed_restart_can_be_committed_or_rolled_back(tmp_path: Any) -> None:
    manager = FakeKernelManager()
    coordinator = DistributedKernelCoordinator(manager, registry_dir=tmp_path)

    prepared = await coordinator.prepare_world_size("kernel-id", 2)

    assert prepared["restart_required"] is True
    assert prepared["world_size"] == 2
    assert manager.restarts == 0
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_WORLD_SIZE"] == "2"
    assert coordinator.describe("kernel-id")["proxied"] is False

    rolled_back = await coordinator.finish_prepared_restart(
        "kernel-id", prepared["restart_token"], commit=False
    )

    assert rolled_back["proxied"] is False
    assert manager.kernel.kernel_spec.argv == [
        "python",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]

    prepared = await coordinator.prepare_world_size("kernel-id", 3)
    committed = await coordinator.finish_prepared_restart(
        "kernel-id", prepared["restart_token"], commit=True
    )

    assert committed["proxied"] is True
    assert committed["world_size"] == 3
    assert manager.restarts == 0
    assert manager.kernel.ready_checks == 1

    unchanged = await coordinator.prepare_world_size("kernel-id", 3)
    assert unchanged["restart_required"] is False


def test_launch_adapter_rejects_incompatible_kernel_manager() -> None:
    kernel = FakeKernel()
    kernel._launch_args = None  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="mutable launch arguments"):
        KernelLaunchAdapter().capture(kernel)
