"""Safe behavior for ordinary ``breakpoint()`` in a distributed kernel."""

from __future__ import annotations

import os
from typing import NoReturn


class DistributedBreakpointError(RuntimeError):
    """Raised instead of starting competing interactive debuggers."""


def distributed_breakpoint(*args: object, **kwargs: object) -> NoReturn:
    """Refuse an uncoordinated multi-rank ``pdb`` session.

    Notebook gutter breakpoints use the coordinated Jupyter debug protocol.
    Python's built-in hook would instead start several ``pdb`` sessions that
    race for the same stdin stream, so it continues to fail explicitly.
    """

    rank = os.environ.get("RANK", "?")
    world_size = os.environ.get("WORLD_SIZE", "?")
    raise DistributedBreakpointError(
        "breakpoint() is disabled for this Jupyter Distributed kernel "
        f"(rank {rank}/{world_size}); use an explicitly coordinated debugger"
    )
