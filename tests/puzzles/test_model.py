"""JSON round trips and schema validation for the logic-grid model."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from redline.puzzles.logic_grid import LADDER, LogicGridPuzzle, generate
from redline.puzzles.logic_grid.model import Clue, NotAtClue, Ref, Solution, Spec

levels = st.sampled_from(sorted(LADDER))
seeds = st.integers(min_value=0, max_value=2**32 - 1)
CLUE = TypeAdapter(Clue)
KINDS = [
    "at",
    "not_at",
    "same_house",
    "not_same_house",
    "left_of",
    "immediately_left",
    "next_to",
]


@given(levels, seeds)
def test_puzzle_json_round_trip(level: int, seed: int) -> None:
    p = generate(level, seed)
    assert LogicGridPuzzle.model_validate_json(p.model_dump_json()) == p


@given(levels, seeds)
def test_clue_json_validates_against_the_schema(level: int, seed: int) -> None:
    for clue in generate(level, seed).clues:
        assert CLUE.validate_json(CLUE.dump_json(clue)) == clue


def test_clue_schema_discriminates_on_the_seven_kinds() -> None:
    schema = CLUE.json_schema()
    assert schema["discriminator"]["propertyName"] == "kind"
    assert sorted(schema["discriminator"]["mapping"]) == sorted(KINDS)


CAT = '{"attribute": "pet", "value": "cat"}'
BOB = '{"attribute": "name", "value": "Bob"}'


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(f'{{"kind": "between", "a": {CAT}, "b": {BOB}}}', id="unknown-kind"),
        pytest.param(f'{{"ref": {CAT}, "house": 2}}', id="missing-kind"),
        pytest.param(f'{{"kind": "at", "ref": {CAT}, "house": 0}}', id="house-0"),
        pytest.param(f'{{"kind": "at", "ref": {CAT}, "house": 2, "x": 1}}', id="extra-field"),
        pytest.param(f'{{"kind": "left_of", "a": {CAT}}}', id="missing-field"),
    ],
)
def test_malformed_clue_json_is_rejected(payload: str) -> None:
    with pytest.raises(ValidationError):
        CLUE.validate_json(payload)


def test_inconsistent_puzzles_are_rejected() -> None:
    p = generate(2, 0)
    name, values = p.spec.attributes[1]
    ref = Ref(attribute=name, value=values[0])
    unknown = Ref(attribute=name, value="unicorn")
    shuffled = dict(p.solution.root)
    shuffled[name] = (values[0], values[0], values[1])
    bad_parts = [
        {"clues": (*p.clues, NotAtClue(ref=unknown, house=1))},
        {"clues": (*p.clues, NotAtClue(ref=ref, house=p.spec.houses + 1))},
        {"solution": Solution(shuffled)},
    ]
    for bad in bad_parts:
        with pytest.raises(ValidationError):
            LogicGridPuzzle(**{**dict(p), **bad})


@pytest.mark.parametrize(
    "attributes",
    [
        pytest.param((("name", ("Alice", "Bob")), ("pet", ("Alice", "cat"))), id="shared-value"),
        pytest.param((("name", ("Alice", "Bob")), ("pet", ("cat",))), id="short-attribute"),
        pytest.param((("name", ("Alice", "Bob")), ("name", ("cat", "dog"))), id="same-name"),
        pytest.param((), id="no-attributes"),
    ],
)
def test_malformed_specs_are_rejected(attributes: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
    with pytest.raises(ValidationError):
        Spec(houses=2, attributes=attributes)
