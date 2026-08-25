from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[2]
DEMOS = ROOT / "demos"
CPU_DEMOS = ("debugging.ipynb",)
GPU_DEMOS = ("fsdp_training.ipynb", "tp_chat.ipynb")


def gpu_test(required_devices: int) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        torch.cuda.device_count() < required_devices,
        reason=f"demo requires at least {required_devices} CUDA devices",
    )


def _convert_and_run(notebook: Path, tmp_path: Path) -> None:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    world_size = document["metadata"]["jupyter_distributed"]["world_size"]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "python",
            "--output",
            notebook.stem,
            "--output-dir",
            str(tmp_path),
            str(notebook),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    script = tmp_path / f"{notebook.stem}.py"
    # The debugging demo should exercise its forward pass without waiting for
    # an interactive debugger in automated test processes.
    environment = {**os.environ, "PYTHONBREAKPOINT": "0"}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--node-rank=0",
            f"--nproc-per-node={world_size}",
            "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            str(script),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_demo_is_exercised() -> None:
    assert {path.name for path in DEMOS.glob("*.ipynb")} == set(CPU_DEMOS + GPU_DEMOS)


def test_transformers_tp_api_matches_demo() -> None:
    from transformers import Qwen3Config
    from transformers.distributed import DistributedConfig

    distributed_config = DistributedConfig(tp_size=2)

    assert distributed_config.tp_size == 2
    assert Qwen3Config().base_model_tp_plan


@pytest.mark.parametrize("name", CPU_DEMOS)
def test_cpu_demo(name: str, tmp_path: Path) -> None:
    _convert_and_run(DEMOS / name, tmp_path)


@gpu_test(required_devices=2)
@pytest.mark.parametrize("name", GPU_DEMOS)
def test_gpu_demo(name: str, tmp_path: Path) -> None:
    _convert_and_run(DEMOS / name, tmp_path)
