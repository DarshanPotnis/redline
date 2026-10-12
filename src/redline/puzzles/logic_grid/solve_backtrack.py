"""Backtracking solution counter, used by the generator.

Assign one attribute's permutation at a time. Each clue is indexed by the last attribute it
mentions and checked as soon as that attribute is assigned. The search stops at `cap` solutions.
"""

import itertools
from collections.abc import Callable, Sequence
from typing import assert_never

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
    Spec,
    clue_refs,
)

# table[x][y]: whether the clue holds with its first ref in house x and its second in y (0-based).
type _Table = tuple[tuple[bool, ...], ...]
# (attribute of first ref, its value, attribute of second ref, its value, table), all indices.
type _Check = tuple[int, int, int, int, _Table]


def count_solutions(spec: Spec, clues: Sequence[Clue], cap: int | None = None) -> int:
    """Count the solutions satisfying every clue, stopping once `cap` are found (None: no cap)."""
    if cap is not None and cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    n_attributes = len(spec.attributes)
    checks: list[list[_Check]] = [[] for _ in range(n_attributes)]
    for clue in clues:
        check = _compile(spec, clue)
        checks[max(check[0], check[2])].append(check)
    # A permutation p places value v of an attribute in house p[v].
    permutations = list(itertools.permutations(range(spec.houses)))
    assigned: list[tuple[int, ...]] = [() for _ in range(n_attributes)]
    found = 0

    def search(depth: int) -> bool:
        """Assign attribute `depth` onward; return True once the cap is reached."""
        nonlocal found
        if depth == n_attributes:
            found += 1
            return found == cap
        for permutation in permutations:
            assigned[depth] = permutation
            # A plain loop rather than all(...): this is the hot path, and a generator per
            # candidate permutation made level-4 generation about 3x slower.
            for a1, v1, a2, v2, table in checks[depth]:
                if not table[assigned[a1][v1]][assigned[a2][v2]]:
                    break
            else:
                if search(depth + 1):
                    return True
        return False

    search(0)
    return found


def _compile(spec: Spec, clue: Clue) -> _Check:
    first, second = clue_refs(clue)
    a1, v1 = _locate(spec, first)
    a2, v2 = _locate(spec, second)
    return a1, v1, a2, v2, _table(clue, spec.houses)


def _locate(spec: Spec, ref: Ref) -> tuple[int, int]:
    for index, (name, values) in enumerate(spec.attributes):
        if name == ref.attribute and ref.value in values:
            return index, values.index(ref.value)
    raise ValueError(f"clue mentions {ref.attribute}={ref.value!r}, which is not in the spec")


def _table(clue: Clue, houses: int) -> _Table:
    # Written independently of clues.holds on purpose: the brute-force oracle uses holds, so the
    # differential test cross-checks the two implementations.
    match clue:
        case AtClue():
            target = clue.house - 1
            return _tabulate(houses, lambda x, _: x == target)
        case NotAtClue():
            target = clue.house - 1
            return _tabulate(houses, lambda x, _: x != target)
        case SameHouseClue():
            return _tabulate(houses, lambda x, y: x == y)
        case NotSameHouseClue():
            return _tabulate(houses, lambda x, y: x != y)
        case LeftOfClue():
            return _tabulate(houses, lambda x, y: x < y)
        case ImmediatelyLeftClue():
            return _tabulate(houses, lambda x, y: x + 1 == y)
        case NextToClue():
            return _tabulate(houses, lambda x, y: abs(x - y) == 1)
        case _:
            assert_never(clue)


def _tabulate(houses: int, relation: Callable[[int, int], bool]) -> _Table:
    return tuple(tuple(relation(x, y) for y in range(houses)) for x in range(houses))
