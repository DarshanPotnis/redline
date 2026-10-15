"""Reasoning Gym tasks as Redline generators: one dataset item per (level, seed).

`generate(level, seed)` is item 0 of `reasoning_gym.create_dataset(task, size=1, seed=seed,
**ladder[level])`, which is deterministic and does not depend on `size`. The question is the
puzzle text; RG's answer and metadata (which hold the solution) stay in the hidden instance, and
the verifier rebuilds the item from (task, config, seed) to score with RG's own checker.
"""

from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any, Final

import reasoning_gym
from pydantic import JsonValue, TypeAdapter

from redline.puzzles.base import PuzzleGenerator, PuzzleInstance

# Neutral about what comes before the answer, so a strategy decides that: "End your response with
# a final line..." (1f) invited a response first, and direct reasoned for thousands of tokens.
ANSWER_FORMAT = (
    "Give your final answer as a single line of the form: Answer: <your answer>, in the format "
    "the question asks for."
)
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)

type Config = Mapping[str, int]


def rg_item(task: str, config: Config, seed: int) -> dict[str, Any]:
    """The RG dataset item for (task, config, seed)."""
    return dict(reasoning_gym.create_dataset(task, size=1, seed=seed, **config)[0])


def rg_dataset(task: str, config: Config, seed: int) -> Any:
    """The one-item RG dataset itself, whose `score_answer` is RG's checker for the task."""
    return reasoning_gym.create_dataset(task, size=1, seed=seed, **config)


class ReasoningGymGenerator:
    """A Reasoning Gym task with a difficulty ladder: level -> the task's config parameters."""

    def __init__(self, task: str, ladder: Mapping[int, Config], labels: Mapping[str, str]):
        self.task = task
        self.name = f"reasoning_gym/{task}"
        self.ladder = dict(ladder)
        self.levels: Sequence[int] = tuple(sorted(self.ladder))
        self.labels = dict(labels)  # config key -> short label template, e.g. "{}p"

    def generate(self, level: int, seed: int) -> PuzzleInstance:
        config = self.ladder[level]
        item = rg_item(self.task, config, seed)
        question = str(item["question"])
        return PuzzleInstance(
            generator=self.name,
            level=level,
            seed=seed,
            text=question,
            answer_format=ANSWER_FORMAT,
            structure={},
            answer=str(item["answer"]),
            instance={
                "task": self.task,
                "config": dict(config),
                "seed": seed,
                # RG's metadata includes the solution: hidden, like every answer key.
                "metadata": _JSON.validate_python(_jsonable(item["metadata"])),
                "rg_version": version("reasoning_gym"),
            },
            search_space=None,
            features={
                **{key: float(value) for key, value in config.items()},
                "n_chars": float(len(question)),
            },
        )

    def describe(self, level: int) -> str:
        config = self.ladder[level]
        return " ".join(
            self.labels.get(key, key + "={}").format(value) for key, value in config.items()
        )


def _jsonable(value: Any) -> Any:
    """RG metadata as plain JSON (tuples become lists)."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


# Knights and knaves: (n_people, depth_constraint, width_constraint) per level. Generation takes at
# most ~20 ms even at level 7.
KNIGHTS_KNAVES: Final[PuzzleGenerator] = ReasoningGymGenerator(
    "knights_knaves",
    {
        level: {"n_people": n, "depth_constraint": depth, "width_constraint": width}
        for level, (n, depth, width) in {
            1: (2, 2, 2),
            2: (3, 2, 2),
            3: (4, 2, 2),
            4: (4, 3, 3),
            5: (5, 3, 3),
            6: (6, 3, 3),
            7: (7, 4, 3),
        }.items()
    },
    {"n_people": "{}p", "depth_constraint": "d{}", "width_constraint": "w{}"},
)
