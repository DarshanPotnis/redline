"""Logic-grid generator: seeded puzzles with exactly one solution and a minimal clue set.

`generate(level, seed)` uses `rng = random.Random(seed)` and never the global random module:
1. Pick the attributes (name first, the rest drawn by rng) and their values, then a solution.
2. Take every true clue of that solution except `at` clues, and shuffle them.
3. Add clues one at a time until the puzzle has exactly one solution.
4. Prune: in rng order, drop every clue whose removal keeps the solution unique.
5. Shuffle the final clue order.

Puzzles are fully relational by design: `at` clues ("Alice lives in house 2") never enter the
candidate pool, so positions are pinned only through relations and `not_at` clues. Uniqueness is
still always reachable, because a value's `not_at` clues alone pin its house. `at` stays in the
clue schema for the 1d translations.

Solutions are counted by backtracking up to a search space of 1,000,000 (levels 1-4) and by Z3
above it. Every counter is exact and the generator only branches on "count == 1", so the choice
of counter never changes the puzzle.
"""

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from redline.puzzles.base import PuzzleGenerator, PuzzleInstance
from redline.puzzles.logic_grid.clues import true_clues
from redline.puzzles.logic_grid.model import GENERATOR_NAME, Clue, LogicGridPuzzle, Solution, Spec
from redline.puzzles.logic_grid.solve_backtrack import count_solutions
from redline.puzzles.logic_grid.solve_z3 import count_solutions_z3
from redline.puzzles.logic_grid.vocab import VOCAB

# level -> (houses, attributes)
LADDER: Final[Mapping[int, tuple[int, int]]] = {
    1: (3, 2),
    2: (3, 3),
    3: (4, 3),
    4: (4, 4),
    5: (5, 4),
    6: (6, 5),
    7: (6, 6),
}
BACKTRACKING_MAX_SEARCH_SPACE = 1_000_000

# (spec, clues, cap) -> the number of solutions, counting stops at cap
type Counter = Callable[[Spec, Sequence[Clue], int], int]


def generate(level: int, seed: int, *, counter: Counter | None = None) -> LogicGridPuzzle:
    """The puzzle for (level, seed): the same inputs give the same puzzle in any process.

    `counter` defaults to `default_counter(spec)`; any exact counter gives the same puzzle.
    """
    if level not in LADDER:
        raise ValueError(f"unknown level {level}; levels are {sorted(LADDER)}")
    if seed < 0:
        raise ValueError("seed must be non-negative: random.Random(-n) repeats random.Random(n)")
    rng = random.Random(seed)
    spec = draw_spec(rng, level)
    solution = draw_solution(rng, spec)
    count = counter if counter is not None else default_counter(spec)
    candidates: list[Clue] = [clue for clue in true_clues(spec, solution) if clue.kind != "at"]
    rng.shuffle(candidates)
    clues = _prune(rng, spec, count, _shortest_unique_prefix(spec, count, candidates))
    rng.shuffle(clues)
    return LogicGridPuzzle(spec=spec, clues=tuple(clues), solution=solution, level=level, seed=seed)


def default_counter(spec: Spec) -> Counter:
    """Backtracking within its range (levels 1-4), Z3 above it."""
    if spec.search_space <= BACKTRACKING_MAX_SEARCH_SPACE:
        return count_solutions
    return count_solutions_z3


def draw_spec(rng: random.Random, level: int) -> Spec:
    """Name plus rng-drawn attributes and values, kept in vocabulary order."""
    houses, n_attributes = LADDER[level]
    others = sorted(rng.sample(range(1, len(VOCAB)), n_attributes - 1))
    attributes: list[tuple[str, tuple[str, ...]]] = []
    for attribute in (VOCAB[0], *(VOCAB[i] for i in others)):
        picked = sorted(rng.sample(range(len(attribute.values)), houses))
        attributes.append((attribute.name, tuple(attribute.values[i] for i in picked)))
    return Spec(houses=houses, attributes=tuple(attributes))


def draw_solution(rng: random.Random, spec: Spec) -> Solution:
    """A uniformly random assignment: each attribute's values shuffled across the houses."""
    columns: dict[str, tuple[str, ...]] = {}
    for name, values in spec.attributes:
        column = list(values)
        rng.shuffle(column)
        columns[name] = tuple(column)
    return Solution(columns)


def _is_unique(spec: Spec, count: Counter, clues: Sequence[Clue]) -> bool:
    # The generator's only branch point: keep it "count == 1" so every exact counter agrees.
    return count(spec, clues, 2) == 1


def _shortest_unique_prefix(spec: Spec, count: Counter, candidates: list[Clue]) -> list[Clue]:
    for size in range(1, len(candidates) + 1):
        if _is_unique(spec, count, candidates[:size]):
            return candidates[:size]
    raise RuntimeError("unreachable: the not_at clues alone pin every value")


def _prune(rng: random.Random, spec: Spec, count: Counter, clues: list[Clue]) -> list[Clue]:
    """Drop redundant clues in rng order; what remains is minimal (every clue is needed)."""
    keep = [True] * len(clues)
    order = list(range(len(clues)))
    rng.shuffle(order)
    for i in order:
        keep[i] = False
        remaining = [clue for clue, kept in zip(clues, keep, strict=True) if kept]
        if not _is_unique(spec, count, remaining):
            keep[i] = True
    return [clue for clue, kept in zip(clues, keep, strict=True) if kept]


class LogicGridGenerator:
    """The logic-grid source as a `PuzzleGenerator`."""

    name: str = GENERATOR_NAME
    levels: Sequence[int] = tuple(LADDER)

    def generate(self, level: int, seed: int) -> PuzzleInstance:
        return generate(level, seed).to_instance()


GENERATOR: Final[PuzzleGenerator] = LogicGridGenerator()
