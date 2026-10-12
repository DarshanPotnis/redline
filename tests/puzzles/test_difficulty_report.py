"""The difficulty report has one table row per level, with its seeds stated."""

from redline.puzzles.logic_grid.difficulty_report import build_report


def test_report_has_one_row_per_level() -> None:
    report = build_report({1: range(2), 2: range(2)})
    rows = [line for line in report.splitlines() if line[:3] in ("| 1", "| 2")]
    assert [row.split(" | ")[1] for row in rows] == ["3x2", "3x3"]
    assert "seeds 0-1 at levels 1-2" in report
