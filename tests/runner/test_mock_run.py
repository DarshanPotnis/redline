"""The runner end to end through Inspect's eval_set on mockllm (the `mock_run` fixture)."""

from collections import Counter

from inspect_ai.log import read_eval_log

from conftest import CONTEXT_LIMITED, MockRun
from redline.runner.manifest import load


def test_both_ladders_stop_after_two_empty_levels(mock_run: MockRun) -> None:
    for ladder in mock_run.manifest.ladders:
        assert [level.level for level in ladder.levels] == [1, 2, 3]
        assert [level.solved for level in ladder.levels] == [2, 0, 0]
        assert ladder.stop_reason == "early_stop"
    assert not any(path.name == "L4" for path in mock_run.run_dir.rglob("L4"))


def test_levels_record_their_logs_and_context_limits(mock_run: MockRun) -> None:
    direct = next(ladder for ladder in mock_run.manifest.ladders if ladder.strategy == "direct")
    assert [level.context_limited for level in direct.levels] == [0, 1, 0]
    for level in direct.levels:
        assert level.status == "success" and level.log is not None
        log = read_eval_log(mock_run.run_dir / level.log, header_only=True)
        assert log.eval.task_args["levels"] == level.level
        assert log.eval.model == "mockllm/model"


def test_a_restart_reuses_every_complete_log(mock_run: MockRun) -> None:
    assert len(mock_run.calls) == mock_run.calls_first_run  # no model call the second time
    assert mock_run.resumed.ladders == mock_run.manifest.ladders
    assert len(list(mock_run.run_dir.rglob("*.eval"))) == 6
    assert len(load(mock_run.run_dir).sessions) == 2


def test_each_sample_was_asked_once_per_strategy(mock_run: MockRun) -> None:
    counts = Counter(mock_run.calls)
    # one call per sample and strategy; the context-limited direct sample adds a forcing call
    assert counts[CONTEXT_LIMITED] == 3
    assert {count for sample, count in counts.items() if sample != CONTEXT_LIMITED} == {2}
