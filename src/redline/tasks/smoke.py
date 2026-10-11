"""Smoke task: proves the Inspect pipeline runs end to end. It measures nothing."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def smoke() -> Task:
    return Task(
        dataset=[
            Sample(input="What is 2 + 2? Answer with just the number.", target="4"),
            Sample(input="What is 3 + 5? Answer with just the number.", target="8"),
        ],
        solver=generate(),
        scorer=match(),
    )
