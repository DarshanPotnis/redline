"""The translation format: every clue is {kind, a, b}, converting losslessly to the Clue model."""

import pytest
from pydantic import ValidationError

from redline.puzzles.logic_grid.model import (
    CLUE_KINDS,
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
from redline.strategies.clue_format import House, TranslatedClue, from_clue, to_clue

ALICE = Ref(attribute="name", value="Alice")
CAT = Ref(attribute="pet", value="cat")
EVERY_KIND: list[Clue] = [
    AtClue(ref=CAT, house=2),
    NotAtClue(ref=ALICE, house=1),
    SameHouseClue(a=ALICE, b=CAT),
    NotSameHouseClue(a=CAT, b=ALICE),
    LeftOfClue(a=ALICE, b=CAT),
    ImmediatelyLeftClue(a=CAT, b=ALICE),
    NextToClue(a=ALICE, b=CAT),
]


def test_every_kind_is_covered() -> None:
    assert sorted(clue.kind for clue in EVERY_KIND) == sorted(CLUE_KINDS)


@pytest.mark.parametrize("clue", EVERY_KIND, ids=lambda clue: clue.kind)
def test_round_trip(clue: Clue) -> None:
    translated = from_clue(clue)
    assert to_clue(translated) == clue
    assert from_clue(to_clue(translated)) == translated
    assert set(translated.model_dump()) == {"kind", "a", "b"}


def test_positions_put_the_house_in_b() -> None:
    assert from_clue(NotAtClue(ref=ALICE, house=3)).model_dump(mode="json") == {
        "kind": "not_at",
        "a": {"attribute": "name", "value": "Alice"},
        "b": {"house": 3},
    }


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(
            {"kind": "not_at", "ref": {"attribute": "name", "value": "Alice"}, "house": 1},
            id="internal-ref-house-shape",
        ),
        pytest.param(
            {"kind": "at", "a": {"attribute": "name", "value": "Alice"}, "b": CAT.model_dump()},
            id="at-with-a-value",
        ),
        pytest.param(
            {"kind": "same_house", "a": ALICE.model_dump(), "b": {"house": 2}},
            id="same-house-with-a-house",
        ),
        pytest.param({"kind": "at", "a": ALICE.model_dump(), "b": {"house": 0}}, id="house-zero"),
        pytest.param({"a": ALICE.model_dump(), "b": CAT.model_dump()}, id="no-kind"),
    ],
)
def test_other_shapes_are_rejected(item: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TranslatedClue.model_validate(item)


def test_a_house_is_only_a_house() -> None:
    with pytest.raises(ValidationError):
        House.model_validate({"house": 2, "attribute": "pet"})
