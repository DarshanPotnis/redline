"""BudgetedModel: one shared budget, budget forcing from the reserve, and every call recorded."""

import asyncio

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    StopReason,
    get_model,
)

from redline.budget import ANSWER_ONLY, ANSWER_RESERVE, REASONING, TOTAL_BUDGET, Budget
from redline.strategies.base import BudgetedModel, BudgetExhausted

FORCE = "Out of budget: answer now."
PROMPT: list[ChatMessage] = [ChatMessageUser(content="Solve the puzzle.")]


def scripted(
    *steps: tuple[str, StopReason, int | None], budget: Budget = REASONING
) -> tuple[BudgetedModel, list[int]]:
    """A BudgetedModel over mockllm that replies with the given (text, stop_reason, tokens)
    steps in order; tokens None means "spend this call's whole cap". Also returns the caps the
    model saw."""
    replies = iter(steps)
    caps: list[int] = []

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        assert config.max_tokens is not None
        caps.append(config.max_tokens)
        text, stop_reason, tokens = next(replies)
        spent = config.max_tokens if tokens is None else tokens
        output = ModelOutput.from_content("mockllm/model", text, stop_reason=stop_reason)
        usage = ModelUsage(input_tokens=20, output_tokens=spent, total_tokens=20 + spent)
        return output.model_copy(update={"usage": usage})

    model = get_model("mockllm/model", custom_outputs=respond)
    budgeted = BudgetedModel(
        lambda messages, config: model.generate(messages, config=config),
        total=budget.total,
        reserve=budget.reserve,
        forcing=budget.forcing,
    )
    return budgeted, caps


def generate(model: BudgetedModel) -> ModelOutput:
    return asyncio.run(model.generate(PROMPT, forcing_message=FORCE))


def test_a_finished_call_is_the_only_call() -> None:
    model, caps = scripted(("{}", "stop", 40))
    assert generate(model).completion == "{}"
    assert caps == [TOTAL_BUDGET - ANSWER_RESERVE] == [3584]
    assert [call.forcing for call in model.calls] == [False]
    assert model.used == 40


def test_a_truncated_call_gets_exactly_one_forcing_call() -> None:
    model, caps = scripted(("thinking...", "max_tokens", None), ('{"1": {}}', "max_tokens", None))
    assert generate(model).completion == '{"1": {}}'
    assert caps == [3584, 512]
    assert model.used == TOTAL_BUDGET
    first, forced = model.calls
    assert (first.forcing, forced.forcing) == (False, True)
    # The forcing call continues the conversation: prompt, the cut-off reasoning, the message.
    assert forced.messages == [
        ("user", "Solve the puzzle."),
        ("assistant", "thinking..."),
        ("user", FORCE),
    ]


def test_the_forcing_call_gets_whatever_the_first_call_left() -> None:
    # Stopped on max_tokens at 3,000 of 3,584 tokens (e.g. by the context window).
    model, caps = scripted(("partial", "max_tokens", 3000), ("{}", "stop", 30))
    generate(model)
    assert caps == [3584, TOTAL_BUDGET - 3000]
    assert model.used == 3030


def test_no_forcing_after_other_stops() -> None:
    model, caps = scripted(("partial", "model_length", 900))
    generate(model)
    assert caps == [3584]
    assert [call.forcing for call in model.calls] == [False]


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([("a", "max_tokens", None), ("b", "max_tokens", None)], id="spent"),
        pytest.param([("a", "stop", 3584)], id="only-the-reserve-left"),
    ],
)
def test_calls_are_refused_once_the_budget_is_spent(
    steps: list[tuple[str, StopReason, int | None]],
) -> None:
    model, _ = scripted(*steps)
    generate(model)
    assert model.remaining <= ANSWER_RESERVE
    with pytest.raises(BudgetExhausted):
        generate(model)


def test_a_call_without_usage_is_charged_its_whole_cap() -> None:
    async def no_usage(messages: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        return ModelOutput.from_content("fake", "{}")

    model = BudgetedModel(no_usage)
    generate(model)
    assert model.used == 3584
    assert model.calls[0].output_tokens is None


def test_records_hold_what_the_scorer_needs() -> None:
    model, _ = scripted(("{}", "stop", 40))
    generate(model)
    [call] = model.calls
    assert (call.max_tokens, call.stop_reason, call.output_tokens, call.input_tokens) == (
        3584,
        "stop",
        40,
        20,
    )
    assert call.text == "{}"
    assert call.messages == [("user", "Solve the puzzle.")]


def test_answer_only_is_one_call_capped_at_the_reserve_and_never_forced() -> None:
    # direct's budget: the whole sample is the 512-token reserve, in one call.
    model, caps = scripted(
        ("Let me reason this through...", "max_tokens", None), budget=ANSWER_ONLY
    )
    assert generate(model).stop_reason == "max_tokens"
    assert caps == [ANSWER_RESERVE] == [512]
    assert [call.forcing for call in model.calls] == [False]
    assert model.used == 512 and model.remaining == 0
