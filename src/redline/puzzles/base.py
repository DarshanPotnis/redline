"""Shared puzzle types: the public view a solver gets, the full instance a generator returns,
and the generator protocol."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class PublicPuzzle(BaseModel):
    """Everything a solver may see. Strategies receive this and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Which generator made it, e.g. "logic_grid" or "reasoning_gym/knights_knaves". Strategies
    # word their prompts by its family (the part before "/"); it says nothing about the answer.
    generator: str
    text: str
    answer_format: str
    # Generator-specific structure, all of it already stated in `text` (for the logic grid: the
    # houses, and each attribute with its values). Never the clues, never the answer.
    structure: dict[str, JsonValue]

    @property
    def family(self) -> str:
        return self.generator.split("/", 1)[0]


class PuzzleInstance(BaseModel):
    """One generated puzzle: `public()` is the solver's view; the rest is for the verifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generator: str
    level: int
    seed: int
    text: str
    answer_format: str
    structure: dict[str, JsonValue]
    answer: JsonValue  # a grid for the logic grid; a string for Reasoning Gym
    instance: dict[str, JsonValue]
    search_space: int | None  # None when the generator doesn't know it
    # Difficulty features; empty for generators that have none.
    features: dict[str, float] = Field(default_factory=dict)

    def public(self) -> PublicPuzzle:
        return PublicPuzzle(
            generator=self.generator,
            text=self.text,
            answer_format=self.answer_format,
            structure=self.structure,
        )


class PuzzleGenerator(Protocol):
    """A seeded puzzle source: the same (level, seed) always gives the same instance."""

    name: str
    levels: Sequence[int]

    def generate(self, level: int, seed: int) -> PuzzleInstance: ...

    def describe(self, level: int) -> str:
        """A short label for the level's size, e.g. "4x3" or "4p d3 w3"."""
        ...
