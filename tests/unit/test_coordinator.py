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
class FakeKernel:
    kernel_name: str = "custom-python"
    kernel_spec: FakeSpec = field(default_factory=FakeSpec)
    _launch_args: dict[str, Any] = field(default_factory=lambda: {"cwd": "/notebooks"})


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

    single_again = await coordinator.set_world_size("kernel-id", 1)

    assert single_again["distributed"] is False
    assert single_again["proxied"] is True
    assert manager.kernel.kernel_spec.argv[1:3] == ["-m", "jupyter_distributed.kernel"]
    assert manager.kernel.kernel_spec.interrupt_mode == "message"
    assert manager.kernel._launch_args["env"]["JUPYTER_DISTRIBUTED_WORLD_SIZE"] == "1"
    assert manager.restarts == 3


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


def test_launch_adapter_rejects_incompatible_kernel_manager() -> None:
    kernel = FakeKernel()
    kernel._launch_args = None  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="mutable launch arguments"):
        KernelLaunchAdapter().capture(kernel)
