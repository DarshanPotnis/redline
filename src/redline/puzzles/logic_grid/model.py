"""Logic-grid data model: specs, clues, solutions and puzzles. Houses are 1-indexed throughout."""

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from redline.puzzles.base import PuzzleInstance

GENERATOR_NAME = "logic_grid"
# The version of the puzzle text. Puzzles (spec, clues, solution) are unchanged across versions;
# only their wording changes. 1: 1a-1e (pilot-1e). 2: 1e.1, color as "favorite color" instead of
# "the person in the blue house", which read as a house position.
GENERATOR_VERSION = 2


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Ref(_Frozen):
    """One attribute value, e.g. (pet, cat): the person who keeps the cat."""

    attribute: str
    value: str


class AtClue(_Frozen):
    """`ref` lives in house `house`."""

    kind: Literal["at"] = "at"
    ref: Ref
    house: int = Field(ge=1)


class NotAtClue(_Frozen):
    """`ref` does not live in house `house`."""

    kind: Literal["not_at"] = "not_at"
    ref: Ref
    house: int = Field(ge=1)


class SameHouseClue(_Frozen):
    """`a` and `b` are the same person."""

    kind: Literal["same_house"] = "same_house"
    a: Ref
    b: Ref


class NotSameHouseClue(_Frozen):
    """`a` and `b` are different people."""

    kind: Literal["not_same_house"] = "not_same_house"
    a: Ref
    b: Ref


class LeftOfClue(_Frozen):
    """`a` lives somewhere to the left of `b`."""

    kind: Literal["left_of"] = "left_of"
    a: Ref
    b: Ref


class ImmediatelyLeftClue(_Frozen):
    """`a` lives directly left of `b`."""

    kind: Literal["immediately_left"] = "immediately_left"
    a: Ref
    b: Ref


class NextToClue(_Frozen):
    """`a` and `b` live in adjacent houses."""

    kind: Literal["next_to"] = "next_to"
    a: Ref
    b: Ref


Clue = Annotated[
    AtClue
    | NotAtClue
    | SameHouseClue
    | NotSameHouseClue
    | LeftOfClue
    | ImmediatelyLeftClue
    | NextToClue,
    Field(discriminator="kind"),
]
CLUE_KINDS: tuple[str, ...] = (
    "at",
    "not_at",
    "same_house",
    "not_same_house",
    "left_of",
    "immediately_left",
    "next_to",
)


def clue_refs(clue: Clue) -> tuple[Ref, Ref]:
    """The clue's two refs; a single-ref clue (`at`, `not_at`) repeats its ref."""
    if isinstance(clue, AtClue | NotAtClue):
        return clue.ref, clue.ref
    return clue.a, clue.b


class Spec(_Frozen):
    """The puzzle's shape: the house count and each attribute's values, in display order."""

    houses: int = Field(ge=2)
    attributes: tuple[tuple[str, tuple[str, ...]], ...]

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        names = self.attribute_names
        values = [value for _, attribute_values in self.attributes for value in attribute_values]
        if not names:
            raise ValueError("a spec needs at least one attribute")
        if len(set(names)) != len(names):
            raise ValueError("attribute names must be unique")
        if any(len(attribute_values) != self.houses for _, attribute_values in self.attributes):
            raise ValueError(f"every attribute needs exactly {self.houses} values")
        if len(set(values)) != len(values):
            raise ValueError("values must be unique across all attributes")
        return self

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.attributes)

    @property
    def search_space(self) -> int:
        """Number of candidate solutions: one permutation of the houses per attribute."""
        return math.prod(math.factorial(self.houses) for _ in self.attributes)

    def values_of(self, attribute: str) -> tuple[str, ...]:
        for name, values in self.attributes:
            if name == attribute:
                return values
        raise KeyError(attribute)


class Solution(RootModel[dict[str, tuple[str, ...]]]):
    """The answer key: attribute -> its values in house order (index 0 is house 1)."""

    model_config = ConfigDict(frozen=True)

    def house_of(self, ref: Ref) -> int:
        """The 1-indexed house holding `ref`."""
        column: tuple[str, ...] = self.root[ref.attribute]
        return column.index(ref.value) + 1


class LogicGridPuzzle(_Frozen):
    spec: Spec
    clues: tuple[Clue, ...]
    solution: Solution
    level: int = Field(ge=1)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        spec = self.spec
        if tuple(self.solution.root) != spec.attribute_names:
            raise ValueError("solution attributes must match the spec, in order")
        for name, values in spec.attributes:
            if sorted(self.solution.root[name]) != sorted(values):
                raise ValueError(f"solution for {name!r} is not a permutation of its values")
        for clue in self.clues:
            for ref in clue_refs(clue):
                if ref.attribute not in spec.attribute_names:
                    raise ValueError(f"clue mentions unknown attribute {ref.attribute!r}")
                if ref.value not in spec.values_of(ref.attribute):
                    raise ValueError(f"clue mentions unknown value {ref.value!r}")
            if isinstance(clue, AtClue | NotAtClue) and clue.house > spec.houses:
                raise ValueError(f"clue mentions house {clue.house} of {spec.houses}")
        return self

    def to_instance(self) -> PuzzleInstance:
        # Imported here because both modules import this one.
        from redline.puzzles.logic_grid.difficulty import measure, method
        from redline.puzzles.logic_grid.render import answer_format, render

        return PuzzleInstance(
            generator=GENERATOR_NAME,
            level=self.level,
            seed=self.seed,
            text=render(self),
            answer_format=answer_format(self.spec),
            structure=self.spec.model_dump(mode="json"),
            answer=self.solution.model_dump(mode="json"),
            instance={
                **self.model_dump(mode="json", include={"spec", "clues"}),
                "generator_version": GENERATOR_VERSION,
                "features_method": method(),
            },
            search_space=self.spec.search_space,
            features=measure(self),
        )
