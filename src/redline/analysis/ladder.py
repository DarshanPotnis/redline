"""Per-level and per-ladder summaries of a run's Parquet file, aggregated in DuckDB.

Context-limited samples are harness failures (the context window, not the budget, cut them off),
so every rate here excludes them and their count is reported next to it.

Besides the solved rate, each generator has one secondary score with a chance baseline: the
logic grid's cell accuracy (1/houses for a random valid grid), and Reasoning Gym's partial credit
(for knights and knaves, the expected score of random roles).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from redline.puzzles.logic_grid import LADDER
from redline.puzzles.reasoning_gym import ReasoningGymGenerator
from redline.puzzles.registry import get_generator
from redline.scoring.reasoning_gym import chance_knights_knaves
from redline.stats import wilson_interval

# score: a sample's secondary score. Each generator fills one of the two columns.
_LEVELS = """
SELECT generator, model, strategy, level,
       count(*) FILTER (WHERE NOT context_limited) AS n,
       count(*) FILTER (WHERE context_limited) AS context_limited,
       count(*) FILTER (WHERE solved AND NOT context_limited) AS solved,
       avg(coalesce(cell_accuracy, rg_score)) FILTER (WHERE NOT context_limited) AS score,
       avg(format_ok::DOUBLE) FILTER (WHERE NOT context_limited) AS format_ok,
       avg(forced_answer::DOUBLE) FILTER (WHERE NOT context_limited) AS forced_answer,
       avg(output_tokens) FILTER (WHERE NOT context_limited) AS output_tokens,
       avg(seconds) FILTER (WHERE NOT context_limited) AS seconds
FROM read_parquet($path)
GROUP BY generator, model, strategy, level
"""

_HEALTH = """
SELECT generator, model, strategy,
       count(*) AS samples,
       count(*) FILTER (WHERE context_limited) AS context_limited,
       avg(format_ok::DOUBLE) FILTER (WHERE NOT context_limited) AS format_ok,
       avg(forced_answer::DOUBLE) FILTER (WHERE NOT context_limited) AS forced_answer,
       avg(truncated::DOUBLE) FILTER (WHERE NOT context_limited) AS truncated,
       sum(seconds) AS seconds
FROM read_parquet($path)
GROUP BY generator, model, strategy
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
class Secondary:
    """A generator's secondary score: its names, and what chance would score at each level."""

    title: str  # the table column
    axis: str  # the chart's y label
    chance_label: str
    chance: Callable[[int], float | None]


def secondary(generator: str) -> Secondary:
    if generator == "logic_grid":
        return Secondary(
            "Cell acc.", "mean cell accuracy", "chance (random valid grid)", chance_cell_accuracy
        )
    source = get_generator(generator)
    if generator == "reasoning_gym/knights_knaves" and isinstance(source, ReasoningGymGenerator):
        ladder = source.ladder
        return Secondary(
            "RG score",
            "mean RG score",
            "chance (random roles)",
            lambda level: chance_knights_knaves(ladder[level]["n_people"]),
        )
    return Secondary("RG score", "mean RG score", "chance", lambda level: None)


@dataclass(frozen=True)
class LevelRow:
    generator: str
    model: str
    strategy: str
    level: int
    n: int  # samples scored, context-limited ones excluded
    context_limited: int
    solved: int
    score: float | None  # the generator's secondary score; None when n == 0
    format_ok: float | None
    forced_answer: float | None
    output_tokens: float | None
    seconds: float | None

    @property
    def solved_rate(self) -> float | None:
        return self.solved / self.n if self.n else None

    @property
    def chance(self) -> float | None:
        """The secondary score chance would get here."""
        return secondary(self.generator).chance(self.level)

    @property
    def interval(self) -> tuple[float, float] | None:
        """95% Wilson interval of the solved rate."""
        return wilson_interval(self.solved, self.n) if self.n else None


@dataclass(frozen=True)
class HealthRow:
    generator: str
    model: str
    strategy: str
    samples: int  # context-limited ones included
    context_limited: int
    format_ok: float | None
    forced_answer: float | None
    truncated: float | None
    seconds: float  # summed sample working time


def level_rows(parquet: Path, models: Sequence[str], strategies: Sequence[str]) -> list[LevelRow]:
    """One row per (generator, model, strategy, level) that ran, in the run's order."""
    rows = [LevelRow(*row) for row in _query(_LEVELS, parquet)]
    return sorted(
        rows,
        key=lambda r: (r.generator, _rank(r.model, models), _rank(r.strategy, strategies), r.level),
    )


def health_rows(parquet: Path, models: Sequence[str], strategies: Sequence[str]) -> list[HealthRow]:
    rows = [HealthRow(*row) for row in _query(_HEALTH, parquet)]
    return sorted(
        rows, key=lambda r: (r.generator, _rank(r.model, models), _rank(r.strategy, strategies))
    )


def _query(sql: str, parquet: Path) -> list[tuple[Any, ...]]:
    with duckdb.connect() as connection:
        return connection.execute(sql, {"path": str(parquet)}).fetchall()


def _rank(item: str, order: Sequence[str]) -> int:
    return order.index(item) if item in order else len(order)
