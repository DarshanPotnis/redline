"""direct is the answer-only baseline on every generator: one call capped at the 512-token
reserve, never forced; a reply that runs into the cap counts as a direct_overrun."""

from collections.abc import Callable
from typing import Any

import inspect_ai
import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, ModelUsage, get_model

from redline.budget import ANSWER_RESERVE
from redline.tasks.logic_grid import logic_grid
from redline.tasks.reasoning_gym import reasoning_gym

TASKS: dict[str, tuple[Callable[..., Any], dict[str, object], str]] = {
    "logic_grid": (logic_grid, {}, "grid_verifier"),
    "knights_knaves": (reasoning_gym, {"task": "knights_knaves"}, "rg_verifier"),
}


@pytest.mark.parametrize("generator", sorted(TASKS))
def test_an_over_long_direct_reply_overruns_without_a_forcing_call(
    generator: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    task, args, scorer = TASKS[generator]
    caps: list[int] = []

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        # A model that ignores "no reasoning" and reasons until the cap stops it.
        assert config.max_tokens is not None
        caps.append(config.max_tokens)
        reply = ModelOutput.from_content(
            "mockllm/model", "Let me reason step by step...", stop_reason="max_tokens"
        )
        usage = ModelUsage(
            input_tokens=200, output_tokens=config.max_tokens, total_tokens=200 + config.max_tokens
        )
        return reply.model_copy(update={"usage": usage})

    [log] = inspect_ai.eval(
        task,
        task_args={**args, "levels": "1", "n_seeds": 2, "strategy": "direct"},
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path_factory.mktemp(f"direct-{generator}")),
        display="none",
    )
    assert log.status == "success" and log.samples is not None
    assert caps == [ANSWER_RESERVE, ANSWER_RESERVE]  # exactly one 512-token call per sample
    for sample in log.samples:
        run = sample.metadata["strategy_run"]
        assert run["budget"] == {"total": 512, "reserve": 0, "forcing": False, "used": 512}
        assert len(run["calls"]) == 1
        assert sample.scores is not None
        value = sample.scores[scorer].value
        assert isinstance(value, dict)
        assert value["direct_overrun"] == 1.0
        assert (value["forced_answer"], value["truncated"], value["context_limited"]) == (
            0.0,
            1.0,
            0.0,
        )
        assert value["format_ok"] == 0.0
    assert log.results is not None
    overrun = next(score for score in log.results.scores if score.name == "direct_overrun")
    assert overrun.metrics["mean"].value == 1.0
