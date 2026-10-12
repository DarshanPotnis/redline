"""Clue semantics: whether a clue holds in a solution, and every clue that holds."""

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
    Solution,
    Spec,
)


def holds(clue: Clue, solution: Solution) -> bool:
    """Whether `clue` is true of `solution`. Left means a lower house number."""
    match clue:
        case AtClue():
            return solution.house_of(clue.ref) == clue.house
        case NotAtClue():
            return solution.house_of(clue.ref) != clue.house
        case SameHouseClue():
            return solution.house_of(clue.a) == solution.house_of(clue.b)
        case NotSameHouseClue():
            return solution.house_of(clue.a) != solution.house_of(clue.b)
        case LeftOfClue():
            return solution.house_of(clue.a) < solution.house_of(clue.b)
        case ImmediatelyLeftClue():
            return solution.house_of(clue.a) + 1 == solution.house_of(clue.b)
        case NextToClue():
            return abs(solution.house_of(clue.a) - solution.house_of(clue.b)) == 1
        case _:
            assert_never(clue)


def true_clues(spec: Spec, solution: Solution) -> list[Clue]:
    """Every non-tautological clue that holds in `solution`, in a fixed order.

    Same-house and not-same-house clues only pair different attributes: within one attribute
    they are always false or always true.
    """
    refs = [
        Ref(attribute=name, value=value) for name, values in spec.attributes for value in values
    ]
    houses = [solution.house_of(ref) for ref in refs]
    placed = list(zip(refs, houses, strict=True))
    clues: list[Clue] = [AtClue(ref=ref, house=house) for ref, house in placed]
    clues += [
        NotAtClue(ref=ref, house=other)
        for ref, house in placed
        for other in range(1, spec.houses + 1)
        if other != house
    ]
    for i, (a, house_a) in enumerate(placed):
        for b, house_b in placed[i + 1 :]:
            if a.attribute != b.attribute:
                same = house_a == house_b
                clues.append(SameHouseClue(a=a, b=b) if same else NotSameHouseClue(a=a, b=b))
    for a, house_a in placed:
        for b, house_b in placed:
            if house_a < house_b:
                clues.append(LeftOfClue(a=a, b=b))
            if house_a + 1 == house_b:
                clues.append(ImmediatelyLeftClue(a=a, b=b))
    for i, (a, house_a) in enumerate(placed):
        for b, house_b in placed[i + 1 :]:
            if abs(house_a - house_b) == 1:
                clues.append(NextToClue(a=a, b=b))
    return clues
