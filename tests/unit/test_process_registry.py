from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil

from jupyter_distributed.process_registry import OrphanProcessReaper


async def test_reaps_verified_child_when_proxy_is_gone(tmp_path: Path) -> None:
    directory = tmp_path / "registry"
    directory.mkdir()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        record = {
            "version": 1,
            "proxy": {"pid": 999_999_999, "create_time": 0},
            "children": [
                {
                    "rank": 0,
                    "pid": child.pid,
                    "pgid": os.getpgid(child.pid),
                    "create_time": psutil.Process(child.pid).create_time(),
                }
            ],
        }
        path = directory / "kernel-id.json"
        path.write_text(json.dumps(record), encoding="utf-8")

        assert await OrphanProcessReaper(directory).reap() == 1
        child.wait(timeout=5)
        assert not path.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


async def test_does_not_signal_reused_pid_identity(tmp_path: Path) -> None:
    directory = tmp_path / "registry"
    directory.mkdir()
    current = psutil.Process()
    path = directory / "kernel-id.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "proxy": {"pid": 999_999_999, "create_time": 0},
                "children": [{"pid": current.pid, "create_time": current.create_time() - 10}],
            }
        ),
        encoding="utf-8",
    )

    assert await OrphanProcessReaper(directory).reap() == 0
    assert current.is_running()
    assert not path.exists()
