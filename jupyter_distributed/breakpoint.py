"""Fallback behavior for ``breakpoint()`` before a debugger is attached."""

from __future__ import annotations

import os
import sys

import debugpy


def distributed_breakpoint() -> None:
    """Stop in the caller when attached, otherwise emit an actionable warning."""

    if debugpy.is_client_connected():
        # debugpy's public helper stops at its caller. Calling it through this
        # hook would expose this module as the top frame, so use the same
        # internal primitive with the actual ``breakpoint()`` caller instead.
        from debugpy.server.api import _settrace

        frame = sys._getframe().f_back
        _settrace(
            suspend=True,
            trace_only_current_thread=True,
            patch_multiprocessing=False,
            stop_at_frame=frame,
        )
        return

    rank = os.environ.get("RANK", "?")
    world_size = os.environ.get("WORLD_SIZE", "?")
    print(
        "breakpoint() was ignored by Jupyter Distributed because the debugger "
        f"is not active (rank {rank}/{world_size}); enable the notebook debugger "
        "and run the cell again",
        file=sys.stderr,
    )


__all__ = ["distributed_breakpoint"]
