"""Parsing for the proxy-level ``%%rank N`` cell directive."""

from __future__ import annotations

import re
from dataclasses import dataclass

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


__all__ = ["RankCell", "RankMagicError", "parse_rank_cell"]
