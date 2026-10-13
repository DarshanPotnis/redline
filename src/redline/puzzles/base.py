"""Shared puzzle types: the public view a solver gets, the full instance a generator returns,
and the generator protocol."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class PublicPuzzle(BaseModel):
    """Everything a solver may see. Strategies receive this and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    answer_format: str


class PuzzleInstance(BaseModel):
    """One generated puzzle: `public()` is the solver's view; the rest is for the verifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generator: str
    level: int
    seed: int
    text: str
    answer_format: str
    answer: dict[str, JsonValue]
    instance: dict[str, JsonValue]
    search_space: int
    # Difficulty features; empty for generators that have none (e.g. Reasoning Gym).
    features: dict[str, float] = Field(default_factory=dict)

    def public(self) -> PublicPuzzle:
        return PublicPuzzle(text=self.text, answer_format=self.answer_format)


class PuzzleGenerator(Protocol):
    """A seeded puzzle source: the same (level, seed) always gives the same instance."""

    name: str
    levels: Sequence[int]

    def generate(self, level: int, seed: int) -> PuzzleInstance: ...
