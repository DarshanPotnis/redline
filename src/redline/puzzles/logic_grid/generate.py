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
"""

import random
from collections.abc import Mapping, Sequence
from typing import Final

from redline.puzzles.base import PuzzleGenerator, PuzzleInstance
from redline.puzzles.logic_grid.clues import true_clues
from redline.puzzles.logic_grid.model import GENERATOR_NAME, Clue, LogicGridPuzzle, Solution, Spec
from redline.puzzles.logic_grid.solve_backtrack import count_solutions
from redline.puzzles.logic_grid.vocab import VOCAB

# level -> (houses, attributes). 1b adds levels 5-6.
LADDER: Final[Mapping[int, tuple[int, int]]] = {1: (3, 2), 2: (3, 3), 3: (4, 3), 4: (4, 4)}


def generate(level: int, seed: int) -> LogicGridPuzzle:
    """The puzzle for (level, seed): the same inputs give the same puzzle in any process."""
    if level not in LADDER:
        raise ValueError(f"unknown level {level}; levels are {sorted(LADDER)}")
    if seed < 0:
        raise ValueError("seed must be non-negative: random.Random(-n) repeats random.Random(n)")
    rng = random.Random(seed)
    spec = draw_spec(rng, level)
    solution = draw_solution(rng, spec)
    candidates: list[Clue] = [clue for clue in true_clues(spec, solution) if clue.kind != "at"]
    rng.shuffle(candidates)
    clues = _prune(rng, spec, _shortest_unique_prefix(spec, candidates))
    rng.shuffle(clues)
    return LogicGridPuzzle(spec=spec, clues=tuple(clues), solution=solution, level=level, seed=seed)


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


def _is_unique(spec: Spec, clues: Sequence[Clue]) -> bool:
    return count_solutions(spec, clues, cap=2) == 1


def _shortest_unique_prefix(spec: Spec, candidates: list[Clue]) -> list[Clue]:
    for size in range(1, len(candidates) + 1):
        if _is_unique(spec, candidates[:size]):
            return candidates[:size]
    raise RuntimeError("unreachable: the not_at clues alone pin every value")


def _prune(rng: random.Random, spec: Spec, clues: list[Clue]) -> list[Clue]:
    """Drop redundant clues in rng order; what remains is minimal (every clue is needed)."""
    keep = [True] * len(clues)
    order = list(range(len(clues)))
    rng.shuffle(order)
    for i in order:
        keep[i] = False
        if not _is_unique(spec, [clue for clue, kept in zip(clues, keep, strict=True) if kept]):
            keep[i] = True
    return [clue for clue, kept in zip(clues, keep, strict=True) if kept]


class LogicGridGenerator:
    """The logic-grid source as a `PuzzleGenerator`."""

    name: str = GENERATOR_NAME
    levels: Sequence[int] = tuple(LADDER)

    def generate(self, level: int, seed: int) -> PuzzleInstance:
        return generate(level, seed).to_instance()


GENERATOR: Final[PuzzleGenerator] = LogicGridGenerator()
