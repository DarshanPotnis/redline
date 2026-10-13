"""Truncation-aware scoring: a cut-off generation is classified by what stopped it."""

from pathlib import Path

import inspect_ai
import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ModelOutput, ModelUsage, StopReason, get_model

from redline.puzzles.base import PublicPuzzle
from redline.scoring.logic_grid import truncation
from redline.strategies import get_strategy
from redline.tasks.logic_grid import logic_grid, logic_grid_samples


@pytest.mark.parametrize(
    ("stop_reason", "output_tokens", "expected"),
    [
        pytest.param("stop", 60, (False, False), id="finished"),
        pytest.param("max_tokens", 4096, (True, False), id="spent-the-budget"),
        pytest.param("max_tokens", 3781, (True, True), id="context-stopped-it"),
        pytest.param("model_length", 900, (True, True), id="model-length"),
        pytest.param("max_tokens", None, (True, False), id="no-usage-reported"),
    ],
)
def test_truncation(
    stop_reason: str, output_tokens: int | None, expected: tuple[bool, bool]
) -> None:
    assert truncation(stop_reason, output_tokens, max_tokens=4096) == expected


def prompt(sample: Sample) -> str:
    public = PublicPuzzle.model_validate((sample.metadata or {})["public"])
    return get_strategy("direct")(public)[-1].text


def test_the_scorer_records_why_generation_stopped(tmp_path: Path) -> None:
    samples = logic_grid_samples([1], 4)
    # (stop_reason, output_tokens) per sample: finished, spent the budget, stopped by the
    # context window at 3781 of 4096 tokens (the 1c local run), stopped by model length.
    stops: list[tuple[StopReason, int]] = [
        ("stop", 60),
        ("max_tokens", 4096),
        ("max_tokens", 3781),
        ("model_length", 900),
    ]
    replies = dict(zip((prompt(sample) for sample in samples), stops, strict=True))

    def respond(messages: list[ChatMessage], *_: object) -> ModelOutput:
        stop_reason, output_tokens = replies[messages[-1].text]
        output = ModelOutput.from_content("mockllm/model", "no answer", stop_reason=stop_reason)
        usage = ModelUsage(
            input_tokens=300, output_tokens=output_tokens, total_tokens=300 + output_tokens
        )
        return output.model_copy(update={"usage": usage})

    [log] = inspect_ai.eval(
        logic_grid,
        task_args={"levels": "1", "n_seeds": 4, "strategy": "direct"},
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success" and log.samples is not None
    scores = {sample.id: sample.scores["grid_verifier"] for sample in log.samples if sample.scores}
    finished, budget, context, length = (scores[sample.id] for sample in samples)

    for score, truncated, context_limited in [
        (finished, 0.0, 0.0),
        (budget, 1.0, 0.0),
        (context, 1.0, 1.0),
        (length, 1.0, 1.0),
    ]:
        assert isinstance(score.value, dict)
        assert (score.value["truncated"], score.value["context_limited"]) == (
            truncated,
            context_limited,
        )

    assert context.metadata is not None
    assert context.metadata["stop_reason"] == "max_tokens"
    assert (context.metadata["output_tokens"], context.metadata["max_tokens"]) == (3781, 4096)
    assert context.metadata["input_tokens"] == 300
    assert "harness failure" in (context.explanation or "")
    assert "harness failure" not in (budget.explanation or "")

    assert log.results is not None
    means = {
        score.name: score.metrics["mean"].value
        for score in log.results.scores
        if score.name in ("truncated", "context_limited")
    }
    assert means == {"truncated": 0.75, "context_limited": 0.5}
