"""Truncation-aware scoring: a cut-off generation is classified by what stopped it, judged on
the final model call."""

from pathlib import Path

import inspect_ai
import pytest
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    StopReason,
    get_model,
)

from redline.budget import ANSWER_RESERVE, TOTAL_BUDGET
from redline.scoring.harness import truncation
from redline.strategies.common import FORCE_ANSWER
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


FIRST_CAP = TOTAL_BUDGET - ANSWER_RESERVE  # 3584
type Step = tuple[StopReason, int | None]  # (stop_reason, output_tokens); None: the whole cap


def test_the_scorer_judges_truncation_on_the_final_call(tmp_path: Path) -> None:
    samples = logic_grid_samples([1], 4)
    # Per sample: (first call, forcing call). A first call stopping on max_tokens is forced.
    scripts: list[tuple[Step, Step | None]] = [
        (("stop", 60), None),  # finished
        (("max_tokens", None), ("max_tokens", None)),  # the forced answer spent its reserve
        (("max_tokens", None), ("max_tokens", 300)),  # the context window cut the forced answer
        (("model_length", 900), None),  # stopped by model length: not forced
    ]
    by_puzzle = dict(zip((str(sample.input) for sample in samples), scripts, strict=True))

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        first_user = next(message.text for message in messages if message.role == "user")
        [(first, forced)] = [steps for text, steps in by_puzzle.items() if text in first_user]
        step = forced if messages[-1].text == FORCE_ANSWER else first
        assert step is not None and config.max_tokens is not None
        stop_reason, tokens = step
        spent = config.max_tokens if tokens is None else tokens
        output = ModelOutput.from_content("mockllm/model", "no answer", stop_reason=stop_reason)
        usage = ModelUsage(input_tokens=300, output_tokens=spent, total_tokens=300 + spent)
        return output.model_copy(update={"usage": usage})

    [log] = inspect_ai.eval(
        logic_grid,
        task_args={"levels": "1", "n_seeds": 4, "strategy": "cot"},
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success" and log.samples is not None
    scores = {sample.id: sample.scores["grid_verifier"] for sample in log.samples if sample.scores}
    finished, spent, context, length = (scores[sample.id] for sample in samples)

    for score, truncated, context_limited, forced in [
        (finished, 0.0, 0.0, 0.0),
        (spent, 1.0, 0.0, 1.0),
        (context, 1.0, 1.0, 1.0),
        (length, 1.0, 1.0, 0.0),
    ]:
        assert isinstance(score.value, dict)
        assert (
            score.value["truncated"],
            score.value["context_limited"],
            score.value["forced_answer"],
        ) == (truncated, context_limited, forced)

    assert context.metadata is not None
    assert context.metadata["stop_reason"] == "max_tokens"
    assert (context.metadata["output_tokens"], context.metadata["max_tokens"]) == (300, 512)
    assert context.metadata["budget_used"] == FIRST_CAP + 300
    assert "harness failure" in (context.explanation or "")
    assert "harness failure" not in (spent.explanation or "")

    assert log.results is not None
    means = {
        score.name: score.metrics["mean"].value
        for score in log.results.scores
        if score.name in ("truncated", "context_limited", "forced_answer")
    }
    assert means == {"truncated": 0.75, "context_limited": 0.5, "forced_answer": 0.5}
