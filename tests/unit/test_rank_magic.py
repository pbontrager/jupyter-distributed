from __future__ import annotations

import pytest

from jupyter_distributed.rank_magic import RankMagicError, parse_rank_cell


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
