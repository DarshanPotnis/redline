"""The translation format z3_translate asks for: every clue is {"kind", "a", "b"}.

Five of the seven internal kinds relate two values (`a`, `b`), while at and not_at hold a value
and a house (`ref`, `house`). Models generalized the majority shape and wrote not_at as
{"a": ..., "b": {"house": 3}} (pilot-1e, 1e.1), so this format makes that the only shape: `a` is
always a value, and `b` is a value, or {"house": N} for at and not_at. It is only the strategy's
input and output format; the generator, the scorer and the stored translations use the internal
`Clue` model.
"""

import json
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

type Kind = Literal[
    "at", "not_at", "same_house", "not_same_house", "left_of", "immediately_left", "next_to"
]
POSITION_KINDS: tuple[Kind, ...] = ("at", "not_at")


class House(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    house: int = Field(ge=1)


class TranslatedClue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    a: Ref
    b: Ref | House

    @model_validator(mode="after")
    def _b_fits_the_kind(self) -> Self:
        if self.kind in POSITION_KINDS and not isinstance(self.b, House):
            raise ValueError(f'{self.kind} needs b to be a house: {{"house": N}}')
        if self.kind not in POSITION_KINDS and not isinstance(self.b, Ref):
            raise ValueError(f"{self.kind} needs b to be a value, not a house")
        return self


def to_clue(translated: TranslatedClue) -> Clue:
    a, b = translated.a, translated.b
    match translated.kind, b:
        case "at", House():
            return AtClue(ref=a, house=b.house)
        case "not_at", House():
            return NotAtClue(ref=a, house=b.house)
        case "same_house", Ref():
            return SameHouseClue(a=a, b=b)
        case "not_same_house", Ref():
            return NotSameHouseClue(a=a, b=b)
        case "left_of", Ref():
            return LeftOfClue(a=a, b=b)
        case "immediately_left", Ref():
            return ImmediatelyLeftClue(a=a, b=b)
        case "next_to", Ref():
            return NextToClue(a=a, b=b)
    raise ValueError(f"{translated.kind} cannot take b={b!r}")  # the validator rules this out


def from_clue(clue: Clue) -> TranslatedClue:
    if isinstance(clue, AtClue | NotAtClue):
        return TranslatedClue(kind=clue.kind, a=clue.ref, b=House(house=clue.house))
    return TranslatedClue(kind=clue.kind, a=clue.a, b=clue.b)


def translation_json(clues: Sequence[Clue], *, one_per_line: bool = False) -> str:
    """Clues as a JSON list in the translation format."""
    items = [json.dumps(from_clue(clue).model_dump(mode="json")) for clue in clues]
    if one_per_line:
        return "[\n" + ",\n".join(f"  {item}" for item in items) + "\n]"
    return "[" + ", ".join(items) + "]"
