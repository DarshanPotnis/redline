"""Inspect task over logic-grid puzzles: one sample per (level, seed), graded by grid_verifier."""

import json
from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig

from redline.puzzles.logic_grid import GENERATOR, LADDER
from redline.scoring.logic_grid import grid_verifier
from redline.strategies import get_strategy
from redline.tasks.solver import strategy_solver

# The generation budget, identical for every strategy (the CLAUDE.md budget rule).
MAX_TOKENS = 4096


@task
def logic_grid(
    levels: str | int | Sequence[int] = "1,2,3",
    n_seeds: int = 10,
    seed_start: int = 0,
    strategy: str = "direct",
) -> Task:
    get_strategy(strategy)  # fail before building the dataset, listing the valid names
    return Task(
        dataset=logic_grid_samples(parse_levels(levels), n_seeds, seed_start),
        solver=strategy_solver(strategy),
        scorer=grid_verifier(),
        config=GenerateConfig(max_tokens=MAX_TOKENS, temperature=0.0),
    )


def logic_grid_samples(levels: Sequence[int], n_seeds: int, seed_start: int = 0) -> list[Sample]:
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    seeds = range(seed_start, seed_start + n_seeds)
    return [_sample(level, seed) for level in levels for seed in seeds]


def parse_levels(levels: str | int | Sequence[int]) -> list[int]:
    """Accept "1,2,3", a single level, or a list. Inspect's `-T levels=1` arrives as an int."""
    if isinstance(levels, int):
        parsed = [levels]
    elif isinstance(levels, str):
        parsed = [int(part) for part in levels.split(",") if part.strip()]
    else:
        parsed = list(levels)
    unknown = [level for level in parsed if level not in LADDER]
    if unknown or not parsed:
        raise ValueError(f"unknown levels {unknown or levels!r}; levels are {sorted(LADDER)}")
    return parsed


def _sample(level: int, seed: int) -> Sample:
    instance = GENERATOR.generate(level, seed)
    return Sample(
        id=f"logic_grid/L{level}/s{seed}",
        input=instance.text,
        target=json.dumps(instance.answer),
        metadata={
            "public": instance.public().model_dump(),
            "hidden": instance.instance,
            "level": level,
            "seed": seed,
            "generator": instance.generator,
            **{f"feature_{name}": value for name, value in instance.features.items()},
        },
    )
