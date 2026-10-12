"""Shared puzzle types: the instance every generator returns, and the generator protocol."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class PuzzleInstance(BaseModel):
    """One generated puzzle. Strategies only ever see `text`; the rest is for the verifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generator: str
    level: int
    seed: int
    text: str
    answer: dict[str, JsonValue]
    instance: dict[str, JsonValue]
    search_space: int
    # Difficulty features; empty for generators that have none (e.g. Reasoning Gym).
    features: dict[str, float] = Field(default_factory=dict)


class PuzzleGenerator(Protocol):
    """A seeded puzzle source: the same (level, seed) always gives the same instance."""

    name: str
    levels: Sequence[int]

    def generate(self, level: int, seed: int) -> PuzzleInstance: ...
