"""Brute-force solution counter. Test oracle only: obviously correct, too slow for generation."""

import itertools
from collections.abc import Iterator, Sequence

from redline.puzzles.logic_grid.clues import holds
from redline.puzzles.logic_grid.model import Clue, Solution, Spec

MAX_SEARCH_SPACE = 1_000_000


def count_solutions_bruteforce(spec: Spec, clues: Sequence[Clue]) -> int:
    """Count the assignments that satisfy every clue by trying every assignment."""
    if spec.search_space > MAX_SEARCH_SPACE:
        raise ValueError(
            f"search space {spec.search_space:,} exceeds {MAX_SEARCH_SPACE:,}; "
            "the brute-force counter is a test oracle for small puzzles only"
        )
    return sum(all(holds(clue, solution) for clue in clues) for solution in _assignments(spec))


def _assignments(spec: Spec) -> Iterator[Solution]:
    columns = [itertools.permutations(values) for _, values in spec.attributes]
    for assignment in itertools.product(*columns):
        # Built without validation: every assignment is a valid solution by construction.
        yield Solution.model_construct(dict(zip(spec.attribute_names, assignment, strict=True)))
