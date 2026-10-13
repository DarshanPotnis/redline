"""z3_translate: the model translates, Z3 solves, and only a unique solution becomes an answer."""

import asyncio
import json

import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from pydantic import TypeAdapter

from redline.puzzles.logic_grid import LogicGridPuzzle, generate
from redline.puzzles.logic_grid.model import Clue, LeftOfClue
from redline.puzzles.logic_grid.render import answer_json
from redline.puzzles.logic_grid.vocab import VOCAB
from redline.strategies import z3_translate
from redline.strategies.base import BudgetedModel, StrategyResult

PUZZLES = [(level, seed) for level in (1, 2, 3) for seed in (0, 1)]


def clues_json(clues: list[Clue] | tuple[Clue, ...]) -> str:
    return json.dumps([clue.model_dump(mode="json") for clue in clues])


def translate(puzzle: LogicGridPuzzle, reply: str) -> StrategyResult:
    async def model(messages: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        return ModelOutput.from_content("fake", reply)

    public = puzzle.to_instance().public()
    return asyncio.run(z3_translate.run(public, BudgetedModel(model)))


def test_the_examples_are_valid_clues_with_placeholder_names() -> None:
    examples = TypeAdapter(list[Clue]).validate_python(
        [example for _, example in z3_translate.EXAMPLES]
    )
    assert sorted(clue.kind for clue in examples) == sorted(
        ["at", "not_at", "same_house", "not_same_house", "left_of", "immediately_left", "next_to"]
    )
    vocabulary = {a.name for a in VOCAB} | {v for a in VOCAB for v in a.values}
    for clue in examples:
        for ref in [getattr(clue, field) for field in ("ref", "a", "b") if hasattr(clue, field)]:
            assert ref.attribute not in vocabulary and ref.value not in vocabulary


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
            '[{"kind": "at", "ref": {"attribute": "pet", "value": "unicorn"}, "house": 1}]',
            id="unknown-value",
        ),
    ],
)
def test_unusable_translations_are_parse_errors(reply: str) -> None:
    result = translate(generate(1, 0), reply)
    assert result.artifacts["translation_status"] == "parse_error"
    assert result.artifacts["translated_clues"] is None
    assert result.final_text == ""


def test_a_house_outside_the_puzzle_is_a_parse_error() -> None:
    p = generate(1, 0)
    name, values = p.spec.attributes[0]
    reply = json.dumps([{"kind": "at", "ref": {"attribute": name, "value": values[0]}, "house": 9}])
    assert translate(p, reply).artifacts["translation_status"] == "parse_error"


def test_spelling_is_matched_like_the_grid_parser() -> None:
    p = generate(2, 1)
    shouty = json.loads(clues_json(p.clues))
    for clue in shouty:
        for field in ("ref", "a", "b"):
            if field in clue:
                clue[field] = {key: f"  {value.upper()} " for key, value in clue[field].items()}
    result = translate(p, json.dumps(shouty))
    assert result.artifacts["translation_status"] == "unique"
    assert result.artifacts["translated_clues"] == json.loads(clues_json(p.clues))


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
