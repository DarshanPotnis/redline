"""grid_verifier inside a real Inspect eval, with mockllm playing three kinds of model."""

import json
from pathlib import Path

import inspect_ai
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ModelOutput, get_model

from redline.puzzles.base import PublicPuzzle
from redline.strategies import get_strategy
from redline.tasks.logic_grid import logic_grid


def answer(target: str, swap: bool = False) -> str:
    """The sample's solution in the answer format; `swap` exchanges two cells of one attribute."""
    columns = {name: list(values) for name, values in json.loads(target).items()}
    if swap:
        first = next(iter(columns))
        columns[first][0], columns[first][1] = columns[first][1], columns[first][0]
    houses = len(next(iter(columns.values())))
    return json.dumps(
        {
            str(h): {name: values[h - 1] for name, values in columns.items()}
            for h in range(1, houses + 1)
        }
    )


def prompt(sample: Sample) -> str:
    public = PublicPuzzle.model_validate((sample.metadata or {})["public"])
    return get_strategy("direct")(public)[-1].text


def test_correct_swapped_and_garbage_answers(tmp_path: Path) -> None:
    task = logic_grid(levels="2", n_seeds=3, strategy="direct")
    samples = list(task.dataset)
    replies = {
        prompt(samples[0]): answer(samples[0].target),
        prompt(samples[1]): answer(samples[1].target, swap=True),
        prompt(samples[2]): "Honestly, I think Alice keeps the cat.",
    }

    def respond(messages: list[ChatMessage], *_: object) -> ModelOutput:
        return ModelOutput.from_content(model="mockllm/model", content=replies[messages[-1].text])

    [log] = inspect_ai.eval(
        task,
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success"
    assert log.samples is not None
    scores = {sample.id: sample.scores["grid_verifier"] for sample in log.samples if sample.scores}
    correct, swapped, garbage = (scores[sample.id] for sample in samples)

    assert correct.value == {
        "format_ok": 1.0,
        "valid_grid": 1.0,
        "solved": 1.0,
        "cell_accuracy": 1.0,
    }
    assert correct.metadata == {"broken_clues": [], "n_broken": 0, "parse_error": None}

    assert isinstance(swapped.value, dict) and swapped.metadata is not None
    assert swapped.value["solved"] == 0.0
    assert swapped.value["valid_grid"] == 1.0
    assert 0 < float(swapped.value["cell_accuracy"]) < 1
    assert swapped.metadata["n_broken"] > 0

    assert isinstance(garbage.value, dict) and garbage.metadata is not None
    assert garbage.value["format_ok"] == 0.0
    assert garbage.value["cell_accuracy"] == 0.0
    assert garbage.metadata["parse_error"].startswith("no_json")
