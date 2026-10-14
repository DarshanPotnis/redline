"""Inspect task over a Reasoning Gym generator: one sample per (level, seed), scored with RG's
checker on the extracted "Answer:" line (scoring/reasoning_gym.py).

The same model profiles, budget, solver and answering strategies as the logic grid; z3_translate
translates logic-grid clues, so it is refused here.
"""

from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from redline.puzzles.base import PuzzleGenerator
from redline.puzzles.registry import get_generator
from redline.runtime.profiles import active_profile
from redline.scoring.reasoning_gym import rg_verifier
from redline.strategies import get_strategy
from redline.tasks.logic_grid import TRANSLATING, parse_levels
from redline.tasks.solver import strategy_solver


@task
def reasoning_gym(
    task: str = "knights_knaves",
    levels: str | int | Sequence[int | str] = "1,2,3",
    n_seeds: int = 10,
    seed_start: int = 0,
    strategy: str = "direct",
) -> Task:
    get_strategy(strategy)  # fail before building the dataset, listing the valid names
    if strategy in TRANSLATING:
        raise ValueError(f"{strategy} translates logic-grid clues; it can't run on {task}")
    generator = get_generator(task)
    profile = active_profile()
    return Task(
        dataset=rg_samples(generator, parse_levels(levels, generator.levels), n_seeds, seed_start),
        solver=strategy_solver(strategy),
        scorer=rg_verifier(),
        config=profile.generate_config(),
        metadata={"model_profile": profile.model_dump(mode="json")},
    )


def rg_samples(
    generator: PuzzleGenerator, levels: Sequence[int], n_seeds: int, seed_start: int = 0
) -> list[Sample]:
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    seeds = range(seed_start, seed_start + n_seeds)
    return [_sample(generator, level, seed) for level in levels for seed in seeds]


def _sample(generator: PuzzleGenerator, level: int, seed: int) -> Sample:
    instance = generator.generate(level, seed)
    short = generator.name.removeprefix("reasoning_gym/")
    return Sample(
        id=f"{short}/L{level}/s{seed}",
        input=instance.text,
        target=str(instance.answer),
        metadata={
            "public": instance.public().model_dump(),
            "hidden": instance.instance,
            "level": level,
            "seed": seed,
            "generator": instance.generator,
            **{f"feature_{name}": value for name, value in instance.features.items()},
        },
    )
