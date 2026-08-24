"""Safe behavior for ordinary ``breakpoint()`` in a distributed kernel."""

from __future__ import annotations

import os
from typing import NoReturn


class DistributedBreakpointError(RuntimeError):
    """Raised instead of starting competing interactive debuggers."""


def distributed_breakpoint(*args: object, **kwargs: object) -> NoReturn:
    """Refuse an uncoordinated multi-rank debugger session.

    A later control-plane debugger can replace this hook.  Failing explicitly is
    safer than allowing several pdb instances to race for the same stdin stream.
    """

    rank = os.environ.get("RANK", "?")
    world_size = os.environ.get("WORLD_SIZE", "?")
    raise DistributedBreakpointError(
        "breakpoint() is disabled for this Jupyter Distributed kernel "
        f"(rank {rank}/{world_size}); use an explicitly coordinated debugger"
    )
