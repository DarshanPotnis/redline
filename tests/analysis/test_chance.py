"""The chance baseline, checked against the grader over every valid grid."""

import itertools
import statistics

import pytest
from pydantic import TypeAdapter

from redline.analysis.ladder import chance_cell_accuracy
from redline.puzzles.logic_grid import GENERATOR, LADDER
from redline.puzzles.logic_grid.model import Clue, Solution, Spec
from redline.scoring.logic_grid import grade

CLUES = TypeAdapter(list[Clue])


@pytest.mark.parametrize("level", [1, 2, 3])
def test_chance_is_the_mean_cell_accuracy_of_every_valid_grid(level: int) -> None:
    # Brute force: grade each grid that makes every attribute a permutation of its values.
    puzzle = GENERATOR.generate(level, 0)
    spec = Spec.model_validate(puzzle.instance["spec"])
    clues = CLUES.validate_python(puzzle.instance["clues"])
    solution = Solution.model_validate(puzzle.answer)
    houses = range(1, spec.houses + 1)
    columns = [itertools.permutations(values) for _, values in spec.attributes]
    accuracies = [
        grade(
            {
                h: {
                    name: column[h - 1]
                    for name, column in zip(spec.attribute_names, grid, strict=True)
                }
                for h in houses
            },
            solution,
            clues,
        ).cell_accuracy
        for grid in itertools.product(*columns)
    ]
    assert statistics.fmean(accuracies) == pytest.approx(chance_cell_accuracy(level))
    assert chance_cell_accuracy(level) == 1 / LADDER[level][0]
