"""z3_translate: the model translates, Z3 solves, and only a unique solution becomes an answer."""

import asyncio
import json

import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput

from redline.puzzles.logic_grid import LogicGridPuzzle, generate
from redline.puzzles.logic_grid.model import (
    CLUE_KINDS,
    AtClue,
    Clue,
    LeftOfClue,
    NotAtClue,
    clue_refs,
)
from redline.puzzles.logic_grid.render import answer_json
from redline.puzzles.logic_grid.solve_bruteforce import count_solutions_bruteforce
from redline.puzzles.logic_grid.vocab import VOCAB
from redline.strategies import z3_translate
from redline.strategies.base import BudgetedModel, StrategyResult
from redline.strategies.clue_format import translation_json

PUZZLES = [(level, seed) for level in (1, 2, 3) for seed in (0, 1)]


def clues_json(clues: list[Clue] | tuple[Clue, ...]) -> str:
    """What a perfect translator writes: the clues in the translation format."""
    return translation_json(clues)


def stored(clues: list[Clue] | tuple[Clue, ...]) -> list[dict[str, object]]:
    """How the strategy records translated clues: the internal Clue model's JSON."""
    return [clue.model_dump(mode="json") for clue in clues]


def translate(puzzle: LogicGridPuzzle, reply: str) -> StrategyResult:
    async def model(messages: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        return ModelOutput.from_content("fake", reply)

    public = puzzle.to_instance().public()
    return asyncio.run(z3_translate.run(public, BudgetedModel(model)))


EXAMPLE_CLUES = [clue for _, clue in z3_translate.EXAMPLE]


def test_the_worked_example_parses_through_the_real_parser() -> None:
    shown = z3_translate.CLUE_FORMAT.split("Translation:\n", 1)[1]
    translation = z3_translate.parse_translation(shown, z3_translate.EXAMPLE_SPEC)
    assert (translation.error, translation.malformed) == (None, ())
    assert list(translation.clues) == EXAMPLE_CLUES


def test_the_worked_example_is_a_real_puzzle_with_the_easily_confused_kinds() -> None:
    assert count_solutions_bruteforce(z3_translate.EXAMPLE_SPEC, EXAMPLE_CLUES) == 1
    assert {"at", "not_at", "immediately_left", "same_house"} <= {c.kind for c in EXAMPLE_CLUES}
    for sentence, clue in z3_translate.EXAMPLE:
        assert all(ref.value in sentence for ref in clue_refs(clue))
        if isinstance(clue, AtClue | NotAtClue):
            assert f"house {clue.house}" in sentence


def test_the_worked_example_names_nothing_from_our_vocabulary() -> None:
    # Models copy what they see; the example's names must never be a valid answer.
    ours = {word.casefold() for a in VOCAB for word in (a.name, a.label, *a.values)}
    theirs = {
        word.casefold()
        for name, values in z3_translate.EXAMPLE_SPEC.attributes
        for word in (name, *values)
    }
    for mine in theirs:
        for word in ours:
            assert mine not in word and word not in mine, (mine, word)


def test_the_prompt_lists_all_seven_kinds() -> None:
    assert [kind for kind, _ in z3_translate.KINDS] == list(CLUE_KINDS)
    assert "<" not in z3_translate.CLUE_FORMAT  # no placeholders to mimic


@pytest.mark.parametrize(("level", "seed"), PUZZLES)
def test_the_exact_clues_give_the_unique_answer(level: int, seed: int) -> None:
    p = generate(level, seed)
    result = translate(p, f"Translation:\n```json\n{clues_json(p.clues)}\n```")
    assert result.artifacts["translation_status"] == "unique"
    assert json.loads(result.final_text) == json.loads(answer_json(p.solution))


@pytest.mark.parametrize(("level", "seed"), PUZZLES)
def test_dropping_any_clue_is_ambiguous(level: int, seed: int) -> None:
    # Clue sets are minimal, so every single omission leaves more than one grid.
    p = generate(level, seed)
    for i in range(len(p.clues)):
        status, grid = z3_translate.solve_translation(p.spec, p.clues[:i] + p.clues[i + 1 :])
        assert (status, grid) == ("ambiguous", None)


def test_contradictory_clues_are_unsat_and_answer_nothing() -> None:
    p = generate(2, 0)
    a, b = p.spec.attributes[0][1][:2]
    name = p.spec.attributes[0][0]
    ref_a, ref_b = (
        {"attribute": name, "value": a},
        {"attribute": name, "value": b},
    )
    clash = [
        {"kind": "left_of", "a": ref_a, "b": ref_b},
        {"kind": "left_of", "a": ref_b, "b": ref_a},
    ]
    result = translate(p, json.dumps(clash))
    assert result.artifacts["translation_status"] == "unsat"
    assert result.final_text == ""


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("I would rather just solve it: Alice is in house 1.", id="garbage"),
        pytest.param('[{"kind": "between", "a": 1}]', id="unknown-kind"),
        pytest.param(
            '[{"kind": "at", "a": {"attribute": "pet", "value": "unicorn"}, "b": {"house": 1}}]',
            id="unknown-value",
        ),
        pytest.param(
            '[{"kind": "at", "ref": {"attribute": "name", "value": "Alice"}, "house": 1}]',
            id="internal-shape",
        ),
    ],
)
def test_unusable_translations_are_parse_errors(reply: str) -> None:
    # Nothing reaches Z3: no JSON list, or a list with no well-formed clue in it.
    result = translate(generate(1, 0), reply)
    assert result.artifacts["translation_status"] == "parse_error"
    assert result.artifacts["translated_clues"] is None
    assert result.final_text == ""


def test_a_house_outside_the_puzzle_is_malformed() -> None:
    p = generate(1, 0)
    name, values = p.spec.attributes[0]
    reply = json.dumps(
        [{"kind": "at", "a": {"attribute": name, "value": values[0]}, "b": {"house": 9}}]
    )
    result = translate(p, reply)
    assert result.artifacts["translation_status"] == "parse_error"
    assert result.artifacts["malformed"] == [{"item": 1, "reason": "house 9 is not one of 1..3"}]


# The internal {ref, house} shape, which the translation format no longer accepts: the format is
# uniform {kind, a, b} (the pilot's {a, b: {house}} not_at is now the valid shape).
OLD_SHAPE_NOT_AT = {"kind": "not_at", "ref": {"attribute": "name", "value": "Bob"}, "house": 3}


def test_a_malformed_item_is_dropped_and_counted() -> None:
    p = generate(3, 0)
    reply = json.dumps([*json.loads(clues_json(p.clues)), OLD_SHAPE_NOT_AT])
    result = translate(p, reply)
    assert result.artifacts["translation_status"] == "unique"  # the real clues still suffice
    assert result.final_text == answer_json(p.solution)
    assert result.artifacts["translated_clues"] == stored(p.clues)
    [malformed] = result.artifacts["malformed"]
    assert malformed["item"] == len(p.clues) + 1
    assert "not a clue" in malformed["reason"]


def test_a_malformed_item_in_place_of_a_needed_clue_leaves_it_ambiguous() -> None:
    p = generate(3, 0)
    reply = json.dumps([OLD_SHAPE_NOT_AT, *json.loads(clues_json(p.clues[1:]))])
    result = translate(p, reply)
    assert result.artifacts["translation_status"] == "ambiguous"
    assert [entry["item"] for entry in result.artifacts["malformed"]] == [1]
    assert result.final_text == ""


def test_spelling_is_matched_like_the_grid_parser() -> None:
    p = generate(2, 1)
    shouty = json.loads(clues_json(p.clues))
    for clue in shouty:
        for field in ("a", "b"):
            if "attribute" in clue[field]:  # a value; b may be a house
                clue[field] = {key: f"  {value.upper()} " for key, value in clue[field].items()}
    result = translate(p, json.dumps(shouty))
    assert result.artifacts["translation_status"] == "unique"
    assert result.artifacts["translated_clues"] == stored(p.clues)


def test_the_prompt_carries_the_puzzle_the_spellings_and_the_format() -> None:
    public = generate(2, 1).to_instance().public()
    system, user = z3_translate.build_messages(public)
    assert system.role == "system" and "never solve" in system.text
    assert user.text.startswith(public.text)
    assert z3_translate.CLUE_FORMAT in user.text
    assert user.text.endswith(z3_translate.INSTRUCTION)
    for name, values in public.structure["attributes"]:
        assert f"- {name}: {', '.join(values)}" in user.text


def test_a_flipped_clue_still_parses() -> None:
    # The strategy does not judge content; a misread clue is the scorer's diagnosis to make.
    p = generate(3, 0)
    flipped = [
        LeftOfClue(a=clue.b, b=clue.a) if isinstance(clue, LeftOfClue) else clue for clue in p.clues
    ]
    result = translate(p, clues_json(flipped))
    assert result.artifacts["translation_status"] in ("unsat", "unique", "ambiguous")
