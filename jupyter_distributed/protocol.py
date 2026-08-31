"""Small, JSON-friendly types shared by the runtime and kernel proxy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

OutputKind = Literal["stream", "display_data", "execute_result", "error"]
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(value: object) -> str:
    """Remove terminal control sequences from a plain-text fallback."""

    return _ANSI_ESCAPE.sub("", str(value))


@dataclass(frozen=True, slots=True)
class RankOutput:
    """One output emitted by one rank."""

    rank: int
    kind: OutputKind
    content: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "type": self.kind, "content": dict(self.content)}

    def plain_text(self) -> str:
        if self.kind == "stream":
            return strip_ansi(self.content.get("text", ""))
        if self.kind in {"display_data", "execute_result"}:
            data = self.content.get("data", {})
            if isinstance(data, Mapping) and "text/plain" in data:
                return strip_ansi(data["text/plain"])
            if isinstance(data, Mapping):
                return "<" + ", ".join(sorted(str(key) for key in data)) + ">"
            return str(data)
        traceback = self.content.get("traceback")
        if isinstance(traceback, list):
            return "\n".join(strip_ansi(line) for line in traceback)
        return f"{self.content.get('ename', 'Error')}: {self.content.get('evalue', '')}"


@dataclass(frozen=True, slots=True)
class RankExecution:
    rank: int
    status: Literal["ok", "error", "aborted"]
    outputs: tuple[RankOutput, ...] = ()
    reply: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "status": self.status,
            "outputs": [output.as_dict() for output in self.outputs],
        }


@dataclass(frozen=True, slots=True)
class GroupExecution:
    execution_count: int
    ranks: tuple[RankExecution, ...]
    execution_id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def status(self) -> Literal["ok", "error", "aborted"]:
        statuses = {result.status for result in self.ranks}
        if "error" in statuses:
            return "error"
        if "aborted" in statuses:
            return "aborted"
        return "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "execution_count": self.execution_count,
            "status": self.status,
            "world_size": len(self.ranks),
            "ranks": [result.as_dict() for result in self.ranks],
        }


@dataclass(frozen=True, slots=True)
class GroupStatus:
    state: Literal["stopped", "starting", "idle", "busy", "restarting", "restart_required"]
    world_size: int
    alive_ranks: tuple[int, ...]
    failure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "world_size": self.world_size,
            "alive_ranks": list(self.alive_ranks),
            "failure": self.failure,
        }
