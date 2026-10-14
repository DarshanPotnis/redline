"""Every generator the runner, tasks and analysis know, by the name `--generator` takes."""

from collections.abc import Mapping
from typing import Final

from redline.puzzles.base import PuzzleGenerator
from redline.puzzles.logic_grid import GENERATOR as LOGIC_GRID
from redline.puzzles.reasoning_gym import KNIGHTS_KNAVES

GENERATORS: Final[Mapping[str, PuzzleGenerator]] = {
    "logic_grid": LOGIC_GRID,
    "knights_knaves": KNIGHTS_KNAVES,
}
BY_NAME: Final[Mapping[str, PuzzleGenerator]] = {g.name: g for g in GENERATORS.values()}


def get_generator(name: str) -> PuzzleGenerator:
    """By its `--generator` name ("knights_knaves") or its own name ("reasoning_gym/...")."""
    found = GENERATORS.get(name) or BY_NAME.get(name)
    if found is None:
        raise ValueError(f"unknown generator {name!r}; choose from {', '.join(GENERATORS)}")
    return found
