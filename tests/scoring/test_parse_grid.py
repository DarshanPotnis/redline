"""parse_grid: lenient on the form of the answer, strict on its content."""

import json

import pytest

from redline.puzzles.logic_grid.model import Solution, Spec
from redline.scoring.logic_grid import ParseError, grade, parse_grid

SPEC = Spec(
    houses=3,
    attributes=(("name", ("Alice", "Bob", "Carol")), ("color", ("red", "green", "blue"))),
)
ANSWER = {
    "1": {"name": "Alice", "color": "red"},
    "2": {"name": "Bob", "color": "green"},
    "3": {"name": "Carol", "color": "blue"},
}
GRID = {
    1: {"name": "Alice", "color": "red"},
    2: {"name": "Bob", "color": "green"},
    3: {"name": "Carol", "color": "blue"},
}
WRONG = {
    "1": {"name": "Bob", "color": "red"},
    "2": {"name": "Alice", "color": "green"},
    "3": {"name": "Carol", "color": "blue"},
}


def with_house(house: str, cells: object) -> str:
    return json.dumps({**ANSWER, house: cells})


def raw_object(pairs: list[tuple[str, object]]) -> str:
    """A JSON object that repeats a key exactly, which json.dumps of a dict cannot write."""
    return "{" + ", ".join(f"{json.dumps(key)}: {json.dumps(value)}" for key, value in pairs) + "}"


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(json.dumps(ANSWER), id="unfenced"),
        pytest.param(f"```json\n{json.dumps(ANSWER, indent=2)}\n```", id="fenced"),
        pytest.param(f"Here is the grid: {json.dumps(ANSWER)} Hope that helps!", id="prose"),
        pytest.param(
            f"First guess: {json.dumps(WRONG)}\nCorrected:\n```json\n{json.dumps(ANSWER)}\n```",
            id="two-blocks-last-wins",
        ),
        pytest.param(
            json.dumps({f"House {house}": cells for house, cells in ANSWER.items()}),
            id="house-keys",
        ),
        pytest.param(
            json.dumps({" house 1 ": ANSWER["1"], "HOUSE 2": ANSWER["2"], "3": ANSWER["3"]}),
            id="house-keys-any-case",
        ),
        pytest.param(
            json.dumps(
                {
                    house: {" NAME ": cells["name"].upper(), "Color": f" {cells['color'].title()}"}
                    for house, cells in ANSWER.items()
                }
            ),
            id="wrong-case",
        ),
        pytest.param(
            json.dumps(
                {
                    house: {"Name": cells["name"], "Favorite color": cells["color"]}
                    for house, cells in ANSWER.items()
                }
            ),
            id="attribute-labels",
        ),
    ],
)
def test_accepts_the_answer_in_any_form(response: str) -> None:
    assert parse_grid(response, SPEC) == GRID


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        pytest.param("I believe Alice keeps the cat.", "no_json", id="garbage"),
        pytest.param("{not json at all", "no_json", id="broken-json"),
        pytest.param(
            json.dumps({"1": ANSWER["1"], "2": ANSWER["2"]}), "missing_house", id="missing-house"
        ),
        pytest.param(with_house("4", ANSWER["1"]), "unknown_house", id="unknown-house"),
        pytest.param(
            json.dumps({**ANSWER, "House 1": ANSWER["1"]}), "duplicate_house", id="duplicate-house"
        ),
        pytest.param(
            raw_object(
                [("1", ANSWER["1"]), ("1", ANSWER["1"]), ("2", ANSWER["2"]), ("3", ANSWER["3"])]
            ),
            "duplicate_house",
            id="duplicate-house-exact-key",
        ),
        pytest.param(with_house("2", ["Bob", "green"]), "house_not_object", id="house-not-object"),
        pytest.param(with_house("2", {"name": "Bob"}), "missing_attribute", id="missing-attribute"),
        pytest.param(
            with_house("2", {"name": "Bob", "color": "green", "age": "40"}),
            "unknown_attribute",
            id="unknown-attribute",
        ),
        pytest.param(
            with_house("2", {"name": "Bob", "Name": "Bob", "color": "green"}),
            "duplicate_attribute",
            id="duplicate-attribute",
        ),
        pytest.param(
            with_house("2", {"name": "Bob", "color": "purple"}), "unknown_value", id="unknown-value"
        ),
        pytest.param(
            with_house("2", {"name": "Bob", "color": "Carol"}),
            "unknown_value",
            id="value-from-another-attribute",
        ),
        pytest.param(
            with_house("2", {"name": 2, "color": "green"}), "unknown_value", id="not-a-string"
        ),
        pytest.param(
            f"{json.dumps(ANSWER)} My confidence: {json.dumps({'confidence': 0.9})}",
            "unknown_house",
            id="last-object-wins-even-if-not-a-grid",
        ),
    ],
)
def test_rejects_wrong_content_with_a_reason(response: str, reason: str) -> None:
    result = parse_grid(response, SPEC)
    assert isinstance(result, ParseError)
    assert result.reason == reason


def test_duplicate_values_parse_but_are_not_a_valid_grid() -> None:
    response = with_house("2", {"name": "Bob", "color": "red"})  # red twice, green nowhere
    parsed = parse_grid(response, SPEC)
    assert not isinstance(parsed, ParseError)
    result = grade(
        parsed, Solution({"name": ("Alice", "Bob", "Carol"), "color": ("red", "green", "blue")}), []
    )
    assert result.format_ok
    assert not result.valid_grid
    assert not result.solved
    assert result.broken_clues is None
    assert result.cell_accuracy == 5 / 6
