"""Parsing for the proxy-level ``%%rank N`` cell directive."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_RANK_MAGIC = re.compile(r"%%rank[ \t]+([0-9]+)[ \t]*")


class RankMagicError(ValueError):
    """Raised when a cell starts with an invalid rank directive."""


@dataclass(frozen=True, slots=True)
class RankCell:
    """A cell body targeted at one zero-based process rank."""

    rank: int
    code: str


def parse_rank_cell(code: str) -> RankCell | None:
    """Parse a leading ``%%rank N`` line, returning the target and cell body."""
    lines = code.splitlines(keepends=True)
    if not lines:
        return None
    header = lines[0].rstrip("\r\n").strip()
    if not re.match(r"%%rank(?:[ \t]|$)", header):
        return None
    match = _RANK_MAGIC.fullmatch(header)
    if match is None:
        raise RankMagicError("Usage: %%rank N, where N is a non-negative integer")
    return RankCell(rank=int(match.group(1)), code="".join(lines[1:]))


def register_single_process_rank_magic(ipython: Any) -> None:
    """Register ``%%rank`` for an ordinary one-process IPython kernel."""
    current = ipython.magics_manager.magics["cell"].get("rank")
    if current is not None:
        return

    def rank(line: str, cell: str) -> None:
        from IPython.core.error import UsageError

        try:
            parsed = parse_rank_cell(f"%%rank {line}\n{cell}")
        except RankMagicError as error:
            raise UsageError(str(error)) from error
        assert parsed is not None
        if parsed.rank != 0:
            raise UsageError(f"rank must be between 0 and 0, got {parsed.rank}")
        ipython.run_cell(parsed.code)

    ipython.register_magic_function(rank, magic_kind="cell", magic_name="rank")


def load_ipython_extension(ipython: Any) -> None:
    """Load the single-process compatibility magic as an IPython extension."""
    register_single_process_rank_magic(ipython)


__all__ = [
    "RankCell",
    "RankMagicError",
    "load_ipython_extension",
    "parse_rank_cell",
    "register_single_process_rank_magic",
]
