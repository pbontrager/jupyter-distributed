from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupyter_distributed.rank_magic import (
    RankMagicError,
    parse_rank_cell,
    register_single_process_rank_magic,
)


def test_parses_rank_and_preserves_nested_cell_magic() -> None:
    parsed = parse_rank_cell("%%rank 3\n%%ai model\nExplain this notebook.\n")

    assert parsed is not None
    assert parsed.rank == 3
    assert parsed.code == "%%ai model\nExplain this notebook.\n"


@pytest.mark.parametrize(
    "code",
    [
        "%%rank\nvalue",
        "%%rank -1\nvalue",
        "%%rank one\nvalue",
        "%%rank 1 extra\nvalue",
    ],
)
def test_rejects_invalid_rank_magic(code: str) -> None:
    with pytest.raises(RankMagicError, match="Usage: %%rank N"):
        parse_rank_cell(code)


def test_leaves_other_cells_untouched() -> None:
    assert parse_rank_cell("value = '%%rank 1'") is None


class FakeIPython:
    def __init__(self) -> None:
        self.magics_manager = SimpleNamespace(magics={"cell": {}})
        self.executed: list[str] = []
        self.registrations = 0

    def register_magic_function(self, function, *, magic_kind: str, magic_name: str) -> None:
        assert magic_kind == "cell"
        self.magics_manager.magics["cell"][magic_name] = function
        self.registrations += 1

    def run_cell(self, code: str) -> None:
        self.executed.append(code)


def test_single_process_magic_runs_rank_zero_and_is_idempotent() -> None:
    ipython = FakeIPython()

    register_single_process_rank_magic(ipython)
    register_single_process_rank_magic(ipython)
    ipython.magics_manager.magics["cell"]["rank"]("0", "value = 42")

    assert ipython.registrations == 1
    assert ipython.executed == ["value = 42"]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("1", "rank must be between 0 and 0, got 1"),
        ("not-a-rank", "Usage: %%rank N"),
    ],
)
def test_single_process_magic_rejects_unavailable_or_invalid_rank(line: str, message: str) -> None:
    from IPython.core.error import UsageError

    ipython = FakeIPython()
    register_single_process_rank_magic(ipython)

    with pytest.raises(UsageError, match=message):
        ipython.magics_manager.magics["cell"]["rank"](line, "value = 42")

    assert ipython.executed == []
