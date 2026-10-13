"""Per-level aggregation, the chart's points, and the written report for the mock run."""

from dataclasses import dataclass

import pytest

from conftest import MockRun
from redline.analysis import build_report
from redline.analysis.chart import points
from redline.analysis.ladder import LevelRow, health_rows, level_rows
from redline.analysis.report import ReportPaths

MODELS = ("mockllm/model",)
STRATEGIES = ("direct", "z3_translate")


@dataclass(frozen=True)
class Built:
    paths: ReportPaths
    markdown: str


@pytest.fixture(scope="module")
def built(mock_run: MockRun, tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = tmp_path_factory.mktemp("analysis")
    paths = build_report(mock_run.run_dir, root / "data", root / "reports")
    return Built(paths, paths.markdown.read_text())


def test_level_rows_exclude_and_count_context_limited_samples(built: Built) -> None:
    rows = {(r.strategy, r.level): r for r in level_rows(built.paths.parquet, MODELS, STRATEGIES)}
    assert sorted(rows) == [(s, level) for s in STRATEGIES for level in (1, 2, 3)]  # no L4
    first = rows[("direct", 1)]
    assert (first.n, first.solved, first.solved_rate, first.cell_accuracy) == (2, 2, 1.0, 1.0)
    assert first.interval == pytest.approx((0.3424, 1.0), abs=1e-4)
    limited = rows[("direct", 2)]
    assert (limited.n, limited.context_limited, limited.solved) == (1, 1, 0)


def test_health_rows(built: Built) -> None:
    health = {row.strategy: row for row in health_rows(built.paths.parquet, MODELS, STRATEGIES)}
    assert (health["direct"].samples, health["direct"].context_limited) == (6, 1)
    assert health["z3_translate"].context_limited == 0
    assert health["direct"].format_ok == pytest.approx(2 / 5)


def test_chart_points_skip_unscored_levels() -> None:
    def row(level: int, n: int) -> LevelRow:
        return LevelRow("m", "s", level, n, 5 - n, 0, None, None, None, None, None)

    rows = [row(1, 5), row(2, 0), row(3, 4), LevelRow("m", "t", 1, 5, 0, 1, 0, 0, 0, 0, 0)]
    assert [point.level for point in points(rows, "m", "s")] == [1, 3]


def test_the_report_and_chart_are_written(built: Built) -> None:
    assert built.paths.chart.name == "mock_ladder.png"
    assert built.paths.chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "](figures/mock_ladder.png)" in built.markdown
    assert (
        "**DEV seeds 0-1: for tuning and pipeline checks, not a reported result.**"
        in built.markdown
    )
    assert "| model | direct | 2 | 3x3 | 1 | 1 | 0/1 |" in built.markdown
    assert "| model | direct | L1 2/2, L2 0/2, L3 0/2 | early_stop |" in built.markdown
    assert "Context-limited samples: **1 of 12**" in built.markdown
    assert "| 4 | 4x4 |" not in built.markdown
