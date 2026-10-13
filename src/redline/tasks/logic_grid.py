"""Inspect task over logic-grid puzzles: one sample per (level, seed), graded by grid_verifier."""

import json
from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from redline.puzzles.logic_grid import GENERATOR, LADDER
from redline.runtime.profiles import active_profile
from redline.scoring.logic_grid import grid_verifier
from redline.strategies import get_strategy
from redline.tasks.solver import strategy_solver


@task
def logic_grid(
    levels: str | int | Sequence[int | str] = "1,2,3",
    n_seeds: int = 10,
    seed_start: int = 0,
    strategy: str = "direct",
) -> Task:
    get_strategy(strategy)  # fail before building the dataset, listing the valid names
    # The model under evaluation must have a registered profile: it sets the sampling and the
    # connection limit, and lands in the log. An unregistered model is an error, never a default.
    profile = active_profile()
    return Task(
        dataset=logic_grid_samples(parse_levels(levels), n_seeds, seed_start),
        solver=strategy_solver(strategy),
        scorer=grid_verifier(),
        config=profile.generate_config(),
        metadata={"model_profile": profile.model_dump(mode="json")},
    )


def logic_grid_samples(levels: Sequence[int], n_seeds: int, seed_start: int = 0) -> list[Sample]:
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    seeds = range(seed_start, seed_start + n_seeds)
    return [_sample(level, seed) for level in levels for seed in seeds]


def parse_levels(levels: str | int | Sequence[int | str]) -> list[int]:
    """Accept "1,2,3", a single level, or a list. From Inspect's CLI, `-T levels=1` arrives as
    an int and `-T levels=1,2` as a list of strings."""
    if isinstance(levels, int):
        items: Sequence[int | str] = [levels]
    elif isinstance(levels, str):
        items = levels.split(",")
    else:
        items = levels
    try:
        parsed = [int(item) for item in items if str(item).strip()]
    except ValueError:
        raise ValueError(f"levels must be whole numbers, got {levels!r}") from None
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
