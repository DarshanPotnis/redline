"""Z3-translate strategy: the model only translates the clues into JSON, and Z3 solves them.

One attempt; retrying with solver feedback is the Phase 2 agentic version. Only a unique solution
becomes an answer: an unsat or ambiguous translation answers nothing rather than guess.

The translation is parsed item by item. A malformed item (not a clue, or naming a value or house
the puzzle doesn't have) is dropped and recorded, and the rest go to Z3, so the status is what Z3
sees. It is a parse error only when nothing reaches Z3: no JSON list, or no well-formed item.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from redline.jsonscan import last_json
from redline.puzzles.base import PublicPuzzle
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
from redline.puzzles.logic_grid.render import answer_json
from redline.puzzles.logic_grid.solve_z3 import solutions_z3
from redline.strategies.base import BudgetedModel, StrategyResult

type Status = Literal["parse_error", "unsat", "ambiguous", "unique"]

SYSTEM = (
    "You translate logic-grid puzzle clues into a JSON format. You never solve the puzzle: a "
    "separate solver works out the answer from your translation."
)
INSTRUCTION = "Translate every clue into this JSON list. Do not solve the puzzle."
FORCE_TRANSLATION = "Output only the JSON list of clues now."


def _ref(attribute: str, value: str) -> dict[str, JsonValue]:
    return {"attribute": attribute, "value": value}


# One example per kind. The attributes and values are placeholders that no puzzle uses (a test
# checks), so the examples show the format without hinting at any puzzle's content.
EXAMPLES: list[tuple[str, dict[str, JsonValue]]] = [
    ("a value is in a given house", {"kind": "at", "ref": _ref("shape", "circle"), "house": 2}),
    (
        "a value is not in a given house",
        {"kind": "not_at", "ref": _ref("shape", "square"), "house": 1},
    ),
    (
        "two values belong to the same house",
        {"kind": "same_house", "a": _ref("shape", "circle"), "b": _ref("tool", "hammer")},
    ),
    (
        "two values belong to different houses",
        {"kind": "not_same_house", "a": _ref("shape", "square"), "b": _ref("tool", "saw")},
    ),
    (
        "a is somewhere to the left of b (a lower house number)",
        {"kind": "left_of", "a": _ref("tool", "saw"), "b": _ref("shape", "circle")},
    ),
    (
        "a is directly left of b (exactly one house lower)",
        {"kind": "immediately_left", "a": _ref("shape", "square"), "b": _ref("tool", "hammer")},
    ),
    (
        "a and b are in adjacent houses",
        {"kind": "next_to", "a": _ref("shape", "circle"), "b": _ref("shape", "square")},
    ),
]
# The exact fields of each kind, read from the clue models so they can't drift. The pilot's 4B
# wrote not_at as {"a": ..., "b": {"house": 3}}: the examples alone didn't pin the shape.
CLUE_MODELS: tuple[type[BaseModel], ...] = (
    AtClue,
    NotAtClue,
    SameHouseClue,
    NotSameHouseClue,
    LeftOfClue,
    ImmediatelyLeftClue,
    NextToClue,
)
_FIELD_PLACEHOLDERS = {"ref": "<value>", "a": "<value>", "b": "<value>", "house": "<house number>"}


def field_shape(model: type[BaseModel]) -> str:
    """One kind and its fields: - at: {"kind": "at", "ref": <value>, "house": <house number>}"""
    kind = model.model_fields["kind"].default
    fields = [
        f'"{name}": {_FIELD_PLACEHOLDERS[name]}' for name in model.model_fields if name != "kind"
    ]
    return f'- {kind}: {{"kind": "{kind}", {", ".join(fields)}}}'


CLUE_FORMAT = "\n".join(
    [
        "Write the clues as a JSON list with one object per clue. A value is written as "
        '{"attribute": "<attribute>", "value": "<value>"}. Each kind of clue has exactly these '
        "fields:",
        *(field_shape(model) for model in CLUE_MODELS),
        "These are the seven kinds of clue, each with an example (the attributes in the examples "
        "are placeholders, not from this puzzle):",
        *(
            f"- {example['kind']}: {meaning}. {json.dumps(example)}"
            for meaning, example in EXAMPLES
        ),
    ]
)
_CLUE: TypeAdapter[Clue] = TypeAdapter(Clue)


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
    """One clue in the spec's spelling, or why the item isn't one."""
    try:
        clue = _CLUE.validate_python(item)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        return f"not a clue ({error.error_count()} errors; {where}: {first['msg']})"
    return _in_spec(clue, spec)


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
