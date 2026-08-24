import pytest
from jupyter_distributed.breakpoint import (
    DistributedBreakpointError,
    distributed_breakpoint,
)


def test_distributed_breakpoint_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")

    with pytest.raises(DistributedBreakpointError, match=r"rank 2/4"):
        distributed_breakpoint()
