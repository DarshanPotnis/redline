"""Z3 solution counter: the generator's counter above the backtracking range, and a third
independent oracle in the differential tests.

One Int per (attribute, value) holding its house in 1..N, Distinct per attribute, and one
constraint per clue. Counting solves, blocks the assignment it found, and solves again.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple, assert_never

import z3

from redline.puzzles.logic_grid.model import (
    AtClue,
    Clue,
    ImmediatelyLeftClue,
    LeftOfClue,
    NextToClue,
    NotAtClue,
    NotSameHouseClue,
    Ref,
    SameHouseClue,
    Solution,
    Spec,
)


class Encoding(NamedTuple):
    variables: dict[tuple[str, str], Any]  # (attribute, value) -> Int: the house holding it
    structure: list[Any]  # house ranges, plus one Distinct per attribute
    clues: list[Any]  # one constraint per clue, in clue order


def count_solutions_z3(spec: Spec, clues: Sequence[Clue], cap: int | None = None) -> int:
    """Count the solutions satisfying every clue, stopping once `cap` are found (None: no cap)."""
    if cap is not None and cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    # Z3's main context on purpose: counts are exact whatever state the context carries, and a
    # fresh context per call costs about 4 ms. difficulty.measure() needs fresh contexts.
    encoding = encode(spec, clues, z3.main_ctx())
    solver = z3.Solver()
    solver.add(*encoding.structure, *encoding.clues)
    return count_models(solver, list(encoding.variables.values()), cap)


def encode(spec: Spec, clues: Sequence[Clue], ctx: z3.Context) -> Encoding:
    variables = {
        (name, value): z3.Int(f"{name}={value}", ctx)
        for name, values in spec.attributes
        for value in values
    }
    structure: list[Any] = []
    for name, values in spec.attributes:
        column = [variables[name, value] for value in values]
        structure += [z3.And(house >= 1, house <= spec.houses) for house in column]
        structure.append(z3.Distinct(*column))
    return Encoding(variables, structure, [_constraint(clue, variables) for clue in clues])


def solutions_z3(spec: Spec, clues: Sequence[Clue], cap: int) -> list[Solution]:
    """Up to `cap` actual solutions, found like count_solutions_z3 (solve, block, repeat)."""
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    encoding = encode(spec, clues, z3.main_ctx())
    solver = z3.Solver()
    solver.add(*encoding.structure, *encoding.clues)
    found: list[Solution] = []
    while len(found) < cap:
        result = solver.check()
        if result == z3.unsat:
            break
        if result != z3.sat:
            raise RuntimeError(f"z3 returned {result}: {solver.reason_unknown()}")
        model = solver.model()
        houses = {
            key: int(model.eval(variable, model_completion=True).as_long())
            for key, variable in encoding.variables.items()
        }
        found.append(_solution(spec, houses))
        solver.add(
            z3.Or([v != model.eval(v, model_completion=True) for v in encoding.variables.values()])
        )
    return found


def _solution(spec: Spec, houses: dict[tuple[str, str], int]) -> Solution:
    """Each attribute's values ordered by the house that holds them."""
    return Solution(
        {
            name: tuple(sorted(values, key=lambda value: houses[name, value]))
            for name, values in spec.attributes
        }
    )


def count_models(solver: z3.Solver, variables: Sequence[Any], cap: int | None) -> int:
    """Solve, block the model found, and repeat until `cap` models or unsat."""
    found = 0
    while cap is None or found < cap:
        result = solver.check()
        if result == z3.unsat:
            break
        if result != z3.sat:
            # The generator relies on exact counts, so an inconclusive answer is an error.
            raise RuntimeError(f"z3 returned {result}: {solver.reason_unknown()}")
        found += 1
        model = solver.model()
        solver.add(z3.Or([v != model.eval(v, model_completion=True) for v in variables]))
    return found


def _constraint(clue: Clue, variables: dict[tuple[str, str], Any]) -> Any:
    match clue:
        case AtClue():
            return _house(variables, clue.ref) == clue.house
        case NotAtClue():
            return _house(variables, clue.ref) != clue.house
        case SameHouseClue():
            return _house(variables, clue.a) == _house(variables, clue.b)
        case NotSameHouseClue():
            return _house(variables, clue.a) != _house(variables, clue.b)
        case LeftOfClue():
            return _house(variables, clue.a) < _house(variables, clue.b)
        case ImmediatelyLeftClue():
            return _house(variables, clue.a) + 1 == _house(variables, clue.b)
        case NextToClue():
            a, b = _house(variables, clue.a), _house(variables, clue.b)
            return z3.Or(a - b == 1, b - a == 1)
        case _:
            assert_never(clue)


def _house(variables: dict[tuple[str, str], Any], ref: Ref) -> Any:
    try:
        return variables[ref.attribute, ref.value]
    except KeyError:
        raise ValueError(
            f"clue mentions {ref.attribute}={ref.value!r}, which is not in the spec"
        ) from None
