"""Rendering: one numbered line per clue, every value named, readable sentences."""

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from redline.puzzles.logic_grid import generate, render
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
)
from redline.puzzles.logic_grid.render import clue_sentence

levels = st.sampled_from([1, 2, 3, 4])  # the fast levels; 5-7 are covered by slow tests
seeds = st.integers(min_value=0, max_value=2**32 - 1)


@given(levels, seeds)
def test_each_clue_is_one_numbered_line(level: int, seed: int) -> None:
    p = generate(level, seed)
    numbered = [line for line in render(p).splitlines() if re.match(r"\d+\. ", line)]
    assert numbered == [f"{n}. {clue_sentence(clue)}" for n, clue in enumerate(p.clues, start=1)]


@given(levels, seeds)
def test_every_value_is_named(level: int, seed: int) -> None:
    p = generate(level, seed)
    text = render(p)
    assert all(value in text for _, values in p.spec.attributes for value in values)


ALICE = Ref(attribute="name", value="Alice")
CAT = Ref(attribute="pet", value="cat")
RED = Ref(attribute="color", value="red")


@pytest.mark.parametrize(
    ("clue", "sentence"),
    [
        (AtClue(ref=CAT, house=2), "The person who keeps the cat lives in house 2."),
        (NotAtClue(ref=ALICE, house=1), "Alice does not live in house 1."),
        (SameHouseClue(a=ALICE, b=CAT), "Alice is the person who keeps the cat."),
        (NotSameHouseClue(a=RED, b=ALICE), "The person in the red house is not Alice."),
        (
            LeftOfClue(a=ALICE, b=RED),
            "Alice lives somewhere to the left of the person in the red house.",
        ),
        (
            ImmediatelyLeftClue(a=CAT, b=ALICE),
            "The person who keeps the cat lives directly left of Alice.",
        ),
        (
            NextToClue(a=ALICE, b=CAT),
            "Alice and the person who keeps the cat live next to each other.",
        ),
    ],
)
def test_clue_sentences(clue: Clue, sentence: str) -> None:
    assert clue_sentence(clue) == sentence
