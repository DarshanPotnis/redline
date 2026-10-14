"""Three independent counters (brute force, backtracking, Z3) agree on random clue sets, plus
solver edge cases."""

import random
from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from redline.puzzles.logic_grid.clues import true_clues
from redline.puzzles.logic_grid.generate import draw_solution, draw_spec
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
)
from redline.puzzles.logic_grid.solve_backtrack import count_solutions
from redline.puzzles.logic_grid.solve_bruteforce import count_solutions_bruteforce
from redline.puzzles.logic_grid.solve_z3 import count_solutions_z3, entails
from redline.puzzles.logic_grid.vocab import VOCAB

HOUSE_KINDS = [AtClue, NotAtClue]
PAIR_KINDS = [SameHouseClue, NotSameHouseClue, LeftOfClue, ImmediatelyLeftClue, NextToClue]
Z3_CAP = 50  # every Z3 comparison counts to at most 50


@st.composite
def any_clue(draw: st.DrawFn, spec: Spec) -> Clue:
    """Any clue over the spec's values, true or false, degenerate pairs (a == b) included."""
    refs = st.sampled_from(
        [Ref(attribute=name, value=value) for name, values in spec.attributes for value in values]
    )
    if draw(st.booleans()):
        kind = draw(st.sampled_from(HOUSE_KINDS))
        return kind(ref=draw(refs), house=draw(st.integers(1, spec.houses)))
    pair_kind = draw(st.sampled_from(PAIR_KINDS))
    return pair_kind(a=draw(refs), b=draw(refs))


@st.composite
def clue_sets(draw: st.DrawFn, levels: tuple[int, ...] = (1, 2, 3)) -> tuple[Spec, list[Clue]]:
    """Random subsets of one solution's true clues plus a few arbitrary ones (0, 1 or many
    solutions). The default levels are the ones the brute-force oracle can handle."""
    level = draw(st.sampled_from(levels))
    rng = random.Random(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    spec = draw_spec(rng, level)
    pool = true_clues(spec, draw_solution(rng, spec))
    picked = draw(st.lists(st.sampled_from(pool), max_size=12))
    extra = draw(st.lists(any_clue(spec), max_size=3))
    return spec, picked + extra


@given(clue_sets())
def test_backtracking_matches_brute_force(case: tuple[Spec, list[Clue]]) -> None:
    spec, clues = case
    assert count_solutions(spec, clues) == count_solutions_bruteforce(spec, clues)


@given(clue_sets(), st.integers(min_value=1, max_value=5))
def test_cap_stops_the_count(case: tuple[Spec, list[Clue]], cap: int) -> None:
    spec, clues = case
    assert count_solutions(spec, clues, cap=cap) == min(cap, count_solutions(spec, clues))


@given(clue_sets(levels=(1, 2, 3, 4)))
def test_z3_matches_backtracking(case: tuple[Spec, list[Clue]]) -> None:
    spec, clues = case
    # Backtracking is capped too: uncapped, it can enumerate all 331,776 level-4 candidates, and
    # count_solutions(cap=c) == min(c, full count) is tested above.
    assert count_solutions_z3(spec, clues, cap=Z3_CAP) == count_solutions(spec, clues, cap=Z3_CAP)


@given(clue_sets())
def test_z3_matches_brute_force(case: tuple[Spec, list[Clue]]) -> None:
    spec, clues = case
    expected = min(Z3_CAP, count_solutions_bruteforce(spec, clues))
    assert count_solutions_z3(spec, clues, cap=Z3_CAP) == expected


@given(clue_sets(levels=(1, 2)), st.data())
def test_entailment_matches_brute_force(case: tuple[Spec, list[Clue]], data: st.DataObject) -> None:
    # The premises entail a clue exactly when adding it leaves their solutions unchanged
    # (vacuously so when they have none).
    spec, premises = case
    conclusion = data.draw(any_clue(spec))
    expected = count_solutions_bruteforce(spec, premises) == count_solutions_bruteforce(
        spec, [*premises, conclusion]
    )
    assert entails(spec, premises, conclusion) == expected


@pytest.mark.parametrize("level", [1, 2])
def test_no_clues_count_the_whole_search_space(level: int) -> None:
    spec = draw_spec(random.Random(0), level)
    assert count_solutions(spec, []) == count_solutions_bruteforce(spec, []) == spec.search_space


def test_z3_without_clues_counts_the_whole_search_space() -> None:
    spec = draw_spec(random.Random(0), 1)
    assert count_solutions_z3(spec, []) == spec.search_space == 36


def test_brute_force_refuses_large_search_spaces() -> None:
    spec = Spec(houses=5, attributes=tuple((a.name, a.values[:5]) for a in VOCAB[:3]))
    assert spec.search_space == 1_728_000
    with pytest.raises(ValueError, match="search space"):
        count_solutions_bruteforce(spec, [])


@pytest.mark.parametrize("counter", [count_solutions, count_solutions_z3])
def test_cap_must_be_positive(counter: Callable[..., int]) -> None:
    spec = draw_spec(random.Random(0), 1)
    with pytest.raises(ValueError, match="cap"):
        counter(spec, [], cap=0)


@pytest.mark.parametrize("counter", [count_solutions, count_solutions_z3])
def test_unknown_ref_is_rejected(counter: Callable[..., int]) -> None:
    spec = draw_spec(random.Random(0), 1)
    clue = AtClue(ref=Ref(attribute="pet", value="unicorn"), house=1)
    with pytest.raises(ValueError, match="unicorn"):
        counter(spec, [clue])
