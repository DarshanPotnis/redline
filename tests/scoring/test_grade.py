"""grade(): uniqueness makes "solved" and "breaks no clue" the same verdict for any valid grid."""

import functools

from hypothesis import given
from hypothesis import strategies as st

from redline.puzzles.logic_grid import LogicGridPuzzle, generate
from redline.scoring.logic_grid import Grid, ParseError, grade

seeds = st.integers(min_value=0, max_value=2**32 - 1)


@functools.cache
def puzzle(level: int, seed: int) -> LogicGridPuzzle:
    return generate(level, seed)


@st.composite
def valid_grids(draw: st.DrawFn, puzzle: LogicGridPuzzle) -> Grid:
    """The solution, the solution with up to 3 random swaps, or a random permutation grid."""
    spec = puzzle.spec
    if draw(st.booleans()):
        columns = {name: list(puzzle.solution.root[name]) for name in spec.attribute_names}
        for _ in range(draw(st.integers(min_value=0, max_value=3))):
            name = draw(st.sampled_from(spec.attribute_names))
            i, j = (draw(st.integers(min_value=0, max_value=spec.houses - 1)) for _ in range(2))
            columns[name][i], columns[name][j] = columns[name][j], columns[name][i]
    else:
        columns = {name: draw(st.permutations(values)) for name, values in spec.attributes}
    return {
        house: {name: columns[name][house - 1] for name in spec.attribute_names}
        for house in range(1, spec.houses + 1)
    }


@given(st.sampled_from([1, 2, 3]), seeds, st.data())
def test_valid_grid_is_solved_exactly_when_it_breaks_no_clue(
    level: int, seed: int, data: st.DataObject
) -> None:
    p = puzzle(level, seed)
    result = grade(data.draw(valid_grids(p)), p.solution, p.clues)
    assert result.format_ok
    assert result.valid_grid
    assert result.solved == (result.broken_clues == ())
    assert result.solved == (result.cell_accuracy == 1.0)


def test_a_format_error_scores_zero() -> None:
    p = puzzle(1, 0)
    result = grade(ParseError("no_json", "nothing"), p.solution, p.clues)
    assert (result.format_ok, result.valid_grid, result.solved) == (False, False, False)
    assert result.cell_accuracy == 0.0
    assert result.broken_clues is None
    assert result.parse_error == "no_json: nothing"
