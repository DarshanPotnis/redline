"""Per-level and per-ladder summaries of a run's Parquet file, aggregated in DuckDB.

Context-limited samples are harness failures (the context window, not the budget, cut them off),
so every rate here excludes them and their count is reported next to it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from redline.puzzles.logic_grid import LADDER
from redline.stats import wilson_interval

_LEVELS = """
SELECT model, strategy, level,
       count(*) FILTER (WHERE NOT context_limited) AS n,
       count(*) FILTER (WHERE context_limited) AS context_limited,
       count(*) FILTER (WHERE solved AND NOT context_limited) AS solved,
       avg(cell_accuracy) FILTER (WHERE NOT context_limited) AS cell_accuracy,
       avg(format_ok::DOUBLE) FILTER (WHERE NOT context_limited) AS format_ok,
       avg(forced_answer::DOUBLE) FILTER (WHERE NOT context_limited) AS forced_answer,
       avg(output_tokens) FILTER (WHERE NOT context_limited) AS output_tokens,
       avg(seconds) FILTER (WHERE NOT context_limited) AS seconds
FROM read_parquet($path)
GROUP BY model, strategy, level
"""

_HEALTH = """
SELECT model, strategy,
       count(*) AS samples,
       count(*) FILTER (WHERE context_limited) AS context_limited,
       avg(format_ok::DOUBLE) FILTER (WHERE NOT context_limited) AS format_ok,
       avg(forced_answer::DOUBLE) FILTER (WHERE NOT context_limited) AS forced_answer,
       avg(truncated::DOUBLE) FILTER (WHERE NOT context_limited) AS truncated,
       sum(seconds) AS seconds
FROM read_parquet($path)
GROUP BY model, strategy
"""


def chance_cell_accuracy(level: int) -> float:
    """Expected cell accuracy of a uniformly random valid grid at this level.

    Each attribute is a random permutation of its values, so every cell is right with
    probability 1/houses, whatever the number of attributes. A sample whose answer fails to
    parse scores 0, so a strategy's mean can fall below this.
    """
    houses, _ = LADDER[level]
    return 1 / houses


@dataclass(frozen=True)
class LevelRow:
    model: str
    strategy: str
    level: int
    n: int  # samples scored, context-limited ones excluded
    context_limited: int
    solved: int
    cell_accuracy: float | None  # None when n == 0
    format_ok: float | None
    forced_answer: float | None
    output_tokens: float | None
    seconds: float | None

    @property
    def solved_rate(self) -> float | None:
        return self.solved / self.n if self.n else None

    @property
    def chance(self) -> float:
        """Cell accuracy a random valid grid would get here."""
        return chance_cell_accuracy(self.level)

    @property
    def interval(self) -> tuple[float, float] | None:
        """95% Wilson interval of the solved rate."""
        return wilson_interval(self.solved, self.n) if self.n else None


@dataclass(frozen=True)
class HealthRow:
    model: str
    strategy: str
    samples: int  # context-limited ones included
    context_limited: int
    format_ok: float | None
    forced_answer: float | None
    truncated: float | None
    seconds: float  # summed sample working time


def level_rows(parquet: Path, models: Sequence[str], strategies: Sequence[str]) -> list[LevelRow]:
    """One row per (model, strategy, level) that ran, in the run's model and strategy order."""
    rows = [LevelRow(*row) for row in _query(_LEVELS, parquet)]
    return sorted(
        rows, key=lambda r: (_rank(r.model, models), _rank(r.strategy, strategies), r.level)
    )


def health_rows(parquet: Path, models: Sequence[str], strategies: Sequence[str]) -> list[HealthRow]:
    rows = [HealthRow(*row) for row in _query(_HEALTH, parquet)]
    return sorted(rows, key=lambda r: (_rank(r.model, models), _rank(r.strategy, strategies)))


def _query(sql: str, parquet: Path) -> list[tuple[Any, ...]]:
    with duckdb.connect() as connection:
        return connection.execute(sql, {"path": str(parquet)}).fetchall()


def _rank(item: str, order: Sequence[str]) -> int:
    return order.index(item) if item in order else len(order)
