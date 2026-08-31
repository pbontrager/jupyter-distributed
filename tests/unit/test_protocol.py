from jupyter_distributed.kernel_proxy import RANK_MIME, render_html, render_plain
from jupyter_distributed.protocol import GroupExecution, RankExecution, RankOutput


def test_group_execution_aggregates_rank_errors() -> None:
    execution = GroupExecution(
        execution_count=3,
        execution_id="cell-3",
        ranks=(
            RankExecution(
                rank=0,
                status="ok",
                outputs=(RankOutput(0, "stream", {"text": "hello\n"}),),
            ),
            RankExecution(
                rank=1,
                status="error",
                outputs=(
                    RankOutput(
                        1,
                        "error",
                        {"ename": "ValueError", "evalue": "bad", "traceback": []},
                    ),
                ),
            ),
        ),
    )

    payload = execution.as_dict()
    assert payload["execution_id"] == "cell-3"
    assert payload["status"] == "error"
    assert payload["ranks"][1]["outputs"][0]["rank"] == 1
    assert RANK_MIME == "application/vnd.jupyter-distributed.rank+json"
    assert "[Rank 0 — ok]" in render_plain(execution)
    assert 'data-rank="1" data-status="error"' in render_html(execution)
    assert "Rank 0" not in render_plain(execution, target_rank=1)
    assert "[Rank 1 — error]" in render_plain(execution, target_rank=1)
    assert 'data-rank="0"' not in render_html(execution, target_rank=1)


def test_rich_output_has_a_plain_fallback() -> None:
    output = RankOutput(
        0,
        "display_data",
        {"data": {"image/png": "bytes", "text/plain": "a plot"}},
    )
    assert output.plain_text() == "a plot"


def test_plain_fallback_strips_ansi_sequences() -> None:
    output = RankOutput(
        0,
        "error",
        {
            "ename": "IndexError",
            "evalue": "bad index",
            "traceback": ["\x1b[0;31mIndexError\x1b[0m: bad index"],
        },
    )
    assert output.plain_text() == "IndexError: bad index"
