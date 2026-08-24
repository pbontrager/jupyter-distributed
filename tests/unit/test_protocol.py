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


def test_rich_output_has_a_plain_fallback() -> None:
    output = RankOutput(
        0,
        "display_data",
        {"data": {"image/png": "bytes", "text/plain": "a plot"}},
    )
    assert output.plain_text() == "a plot"
