"""Cross-process registry used to reap rank kernels after a proxy crash."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import psutil


def _process_identity(pid: int) -> dict[str, int | float] | None:
    try:
        process = psutil.Process(pid)
        return {"pid": pid, "create_time": process.create_time()}
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def _same_process(record: Mapping[str, Any]) -> psutil.Process | None:
    try:
        pid = int(record["pid"])
        create_time = float(record["create_time"])
        process = psutil.Process(pid)
        if abs(process.create_time() - create_time) > 0.01:
            return None
        return process
    except (KeyError, TypeError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None


class ChildProcessRegistry:
    """Atomically publish local child process identities for the server."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None

    @classmethod
    def from_environment(cls) -> ChildProcessRegistry:
        return cls(os.environ.get("JUPYTER_DISTRIBUTED_REGISTRY_FILE"))

    def update(self, ranks: Iterable[Any]) -> None:
        if self.path is None:
            return
        children: list[dict[str, int | float]] = []
        for rank in ranks:
            provisioner = getattr(rank.manager, "provisioner", None)
            pid = getattr(provisioner, "pid", None)
            if not isinstance(pid, int):
                continue
            identity = _process_identity(pid)
            if identity is None:
                continue
            pgid = getattr(provisioner, "pgid", None)
            if isinstance(pgid, int):
                identity["pgid"] = pgid
            identity["rank"] = int(rank.rank)
            children.append(identity)

        proxy = _process_identity(os.getpid())
        if proxy is None:
            return
        payload = {"version": 1, "proxy": proxy, "children": children}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.path)

    def remove(self) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


class OrphanProcessReaper:
    """Reap registered child groups whose owning proxy no longer exists."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def path_for(self, kernel_id: str) -> Path:
        return self.directory / f"{kernel_id}.json"

    async def reap(self) -> int:
        return await asyncio.to_thread(self._reap_sync)

    def _reap_sync(self) -> int:
        with self._lock:
            return self._reap_locked()

    def _reap_locked(self) -> int:
        if not self.directory.exists():
            return 0
        reaped = 0
        for path in self.directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    path.unlink(missing_ok=True)
                    continue
                proxy = payload.get("proxy", {})
                if not isinstance(proxy, Mapping):
                    path.unlink(missing_ok=True)
                    continue
                if _same_process(proxy) is not None:
                    continue
                children = payload.get("children", [])
                if not isinstance(children, list):
                    path.unlink(missing_ok=True)
                    continue
                retry = False
                for child in children:
                    if not isinstance(child, Mapping):
                        continue
                    outcome = self._terminate(child)
                    if outcome is True:
                        reaped += 1
                    elif outcome is None:
                        retry = True
                if not retry:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # Malformed state is not trustworthy authority to kill a process.
                path.unlink(missing_ok=True)
        return reaped

    @staticmethod
    def _terminate(record: Mapping[str, Any]) -> bool | None:
        process = _same_process(record)
        if process is None:
            return False
        try:
            descendants = process.children(recursive=True)
            pgid = record.get("pgid")
            if os.name == "posix" and isinstance(pgid, int) and pgid == process.pid:
                os.killpg(pgid, signal.SIGTERM)
            else:
                for descendant in descendants:
                    descendant.terminate()
                process.terminate()
            _gone, alive = psutil.wait_procs([process, *descendants], timeout=2)
            if alive:
                if os.name == "posix" and isinstance(pgid, int) and pgid == process.pid:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    for remaining in alive:
                        remaining.kill()
            return True
        except (ProcessLookupError, psutil.NoSuchProcess):
            return True
        except (PermissionError, psutil.AccessDenied):
            return None


__all__ = ["ChildProcessRegistry", "OrphanProcessReaper"]
