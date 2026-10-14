"""Z3-translate strategy: the model only translates the clues into JSON, and Z3 solves them.

One attempt; retrying with solver feedback is the Phase 2 agentic version. Only a unique solution
becomes an answer: an unsat or ambiguous translation answers nothing rather than guess.

The model writes every clue as {"kind", "a", "b"} (strategies/clue_format.py), taught by one
worked example from a made-up puzzle. The translation is parsed item by item. A malformed item
(not in that format, or naming a value or house the puzzle doesn't have) is dropped and recorded,
and the rest go to Z3, so the status is what Z3 sees. It is a parse error only when nothing
reaches Z3: no JSON list, or no well-formed item.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from pydantic import JsonValue, ValidationError

from redline.jsonscan import last_json
from redline.puzzles.base import PublicPuzzle
from redline.puzzles.logic_grid.model import (
    AtClue,
    Clue,
    ImmediatelyLeftClue,
    NotAtClue,
    Ref,
    SameHouseClue,
    Solution,
    Spec,
)
from redline.puzzles.logic_grid.render import answer_json
from redline.puzzles.logic_grid.solve_z3 import solutions_z3
from redline.strategies.base import BudgetedModel, StrategyResult
from redline.strategies.clue_format import TranslatedClue, to_clue, translation_json

type Status = Literal["parse_error", "unsat", "ambiguous", "unique"]

SYSTEM = (
    "You translate logic-grid puzzle clues into a JSON format. You never solve the puzzle: a "
    "separate solver works out the answer from your translation."
)
INSTRUCTION = "Translate every clue into this JSON list. Do not solve the puzzle."
FORCE_TRANSLATION = "Output only the JSON list of clues now."


KINDS: tuple[tuple[str, str], ...] = (
    ("at", "a lives in house b"),
    ("not_at", "a does not live in house b"),
    ("same_house", "a and b are the same person"),
    ("not_same_house", "a and b are different people"),
    ("left_of", "a lives somewhere to the left of b"),
    ("immediately_left", "a lives directly left of b"),
    ("next_to", "a and b live next to each other"),
)

# One worked example: a two-house puzzle whose attributes and values appear in no real puzzle (a
# test checks), as English clues followed by their complete translation. Small models copy what
# they see: one-line examples leaked their placeholder values into answers, and field lines full
# of <value> placeholders were mimicked structurally (1e.1), so they see a real input and output.
_CANOE = Ref(attribute="boat", value="canoe")
_KAYAK = Ref(attribute="boat", value="kayak")
_BERET = Ref(attribute="hat", value="beret")
_FEDORA = Ref(attribute="hat", value="fedora")
EXAMPLE_SPEC = Spec(
    houses=2, attributes=(("boat", ("canoe", "kayak")), ("hat", ("beret", "fedora")))
)
EXAMPLE: tuple[tuple[str, Clue], ...] = (
    ("The person with the canoe lives in house 2.", AtClue(ref=_CANOE, house=2)),
    ("The person with the beret does not live in house 2.", NotAtClue(ref=_BERET, house=2)),
    (
        "The person with the kayak lives directly left of the person with the fedora.",
        ImmediatelyLeftClue(a=_KAYAK, b=_FEDORA),
    ),
    (
        "The person with the canoe is the person with the fedora.",
        SameHouseClue(a=_CANOE, b=_FEDORA),
    ),
)
CLUE_FORMAT = "\n".join(
    [
        "Write the clues as a JSON list with one object per clue. Every object has exactly three "
        'fields: "kind", "a" and "b". "a" is always a value from the puzzle, written with its '
        'attribute. "b" is a value too, except in the kinds at and not_at, where "b" is a house '
        "number.",
        "The seven kinds:",
        *(f"- {kind}: {meaning}." for kind, meaning in KINDS),
        "",
        "A worked example, from a different puzzle with 2 houses (its attributes, boat and hat, "
        "are not in your puzzle):",
        *(f"- {name}: {', '.join(values)}" for name, values in EXAMPLE_SPEC.attributes),
        "Clues:",
        *(f"{n}. {sentence}" for n, (sentence, _) in enumerate(EXAMPLE, start=1)),
        "Translation:",
        translation_json([clue for _, clue in EXAMPLE], one_per_line=True),
    ]
)


@dataclass(frozen=True)
class Translation:
    clues: tuple[Clue, ...]  # the well-formed items, in the spec's spelling, in order
    malformed: tuple[tuple[int, str], ...]  # (1-based item number in the model's list, why)
    error: str | None = None  # why nothing reached Z3; None when some clue did


def build_messages(puzzle: PublicPuzzle) -> list[ChatMessage]:
    spec = Spec.model_validate(puzzle.structure)
    spellings = "\n".join(f"- {name}: {', '.join(values)}" for name, values in spec.attributes)
    return [
        ChatMessageSystem(content=SYSTEM),
        ChatMessageUser(
            content=f"{puzzle.text}\n\n{CLUE_FORMAT}\n\n"
            f"Use exactly these attribute names and values:\n{spellings}\n\n{INSTRUCTION}"
        ),
    ]


async def run(puzzle: PublicPuzzle, model: BudgetedModel) -> StrategyResult:
    spec = Spec.model_validate(puzzle.structure)
    output = await model.generate(build_messages(puzzle), forcing_message=FORCE_TRANSLATION)
    translation = parse_translation(output.completion, spec)
    malformed: list[JsonValue] = [
        {"item": item, "reason": reason} for item, reason in translation.malformed
    ]
    if translation.error is not None:
        return StrategyResult(
            final_text="",
            artifacts={
                "translation_status": "parse_error",
                "translated_clues": None,
                "malformed": malformed,
                "parse_error": translation.error,
            },
        )
    status, grid = solve_translation(spec, translation.clues)
    return StrategyResult(
        final_text=answer_json(grid) if grid is not None else "",
        artifacts={
            "translation_status": status,
            "translated_clues": [clue.model_dump(mode="json") for clue in translation.clues],
            "malformed": malformed,
            "parse_error": None,
        },
    )


def solve_translation(spec: Spec, clues: Sequence[Clue]) -> tuple[Status, Solution | None]:
    """Z3 on the translation: unique gives the grid; unsat and ambiguous give nothing."""
    found = solutions_z3(spec, clues, cap=2)
    if len(found) == 1:
        return "unique", found[0]
    return ("unsat" if not found else "ambiguous"), None


def parse_translation(text: str, spec: Spec) -> Translation:
    """The last JSON list in `text`, item by item, with refs in the spec's own spelling."""
    raw = last_json(text, "[")
    if raw is None:
        return Translation((), (), "no JSON list in the response")
    clues: list[Clue] = []
    malformed: list[tuple[int, str]] = []
    for number, item in enumerate(raw, start=1):
        clue = _parse_item(item, spec)
        if isinstance(clue, str):
            malformed.append((number, clue))
        else:
            clues.append(clue)
    error = None if clues else f"no well-formed clue in the list ({len(malformed)} malformed)"
    return Translation(tuple(clues), tuple(malformed), error)


def _parse_item(item: Any, spec: Spec) -> Clue | str:
    """One clue in the spec's spelling, or why the item isn't one. Items must be in the
    translation format ({"kind", "a", "b"}); the internal {"ref", "house"} shape is malformed."""
    try:
        translated = TranslatedClue.model_validate(item)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        return f"not a clue ({error.error_count()} errors; {where}: {first['msg']})"
    return _in_spec(to_clue(translated), spec)


def _in_spec(clue: Clue, spec: Spec) -> Clue | str:
    fields = ("ref",) if isinstance(clue, AtClue | NotAtClue) else ("a", "b")
    refs: dict[str, Ref] = {}
    for field in fields:
        ref: Ref = getattr(clue, field)
        known = _known_ref(ref, spec)
        if known is None:
            return f"unknown value {ref.attribute}={ref.value!r}"
        refs[field] = known
    if isinstance(clue, AtClue | NotAtClue) and not 1 <= clue.house <= spec.houses:
        return f"house {clue.house} is not one of 1..{spec.houses}"
    return clue.model_copy(update=refs)


def _known_ref(ref: Ref, spec: Spec) -> Ref | None:
    """Case-insensitive, trimmed match against the spec; None if it names nothing there."""
    for name, values in spec.attributes:
        if _fold(name) == _fold(ref.attribute):
            value = next((v for v in values if _fold(v) == _fold(ref.value)), None)
            return None if value is None else Ref(attribute=name, value=value)
    return None


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()
